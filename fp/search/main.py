import logging
import random
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

from constants import BattleType
from fp.battle import Battle
from config import FoulPlayConfig
from .standard_battles import prepare_battles
from .random_battles import prepare_random_battles

from poke_engine import State as PokeEngineState, monte_carlo_tree_search, MctsResult

from fp.search.poke_engine_helpers import battle_to_poke_engine_state, poke_engine_get_damage_rolls
from fp.helpers import type_effectiveness_modifier
from data import all_move_json
from fp.search.move_priors import get_move_priorities, should_strongly_consider_switching
from fp.search.position_eval import get_position_metrics
from fp.search.switch_logic import get_switch_recommendation
from fp.strategy.scouting import UsageScout

logger = logging.getLogger(__name__)


from fp.strategy.strategic_context import StrategicContext
from fp.search.lookahead import evaluate_candidate_lines
from fp.strategy.risk import MoveRiskProfile


def select_move_from_mcts_results(
    battle: Battle,
    mcts_results: list[(MctsResult, float, int)],
    strategic_context: StrategicContext | None = None,
) -> tuple[str, dict]:
    final_policy = {}
    move_scores = {}
    move_variance = {}

    for mcts_result, sample_chance, index in mcts_results:
        this_policy = max(mcts_result.side_one, key=lambda x: x.visits)
        logger.debug(
            "Policy {}: {} visited {}% avg_score={} sample_chance_multiplier={}".format(
                index,
                this_policy.move_choice,
                round(100 * this_policy.visits / mcts_result.total_visits, 2),
                round(this_policy.total_score / this_policy.visits, 3),
                round(sample_chance, 3),
            )
        )
        for s1_option in mcts_result.side_one:
            move_name = s1_option.move_choice
            visit_percentage = s1_option.visits / mcts_result.total_visits
            weighted_visit = sample_chance * visit_percentage

            final_policy[move_name] = final_policy.get(move_name, 0) + weighted_visit

            if move_name not in move_scores:
                move_scores[move_name] = []
            move_scores[move_name].append(visit_percentage)

    final_policy = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)

    if final_policy:
        highest_percentage = final_policy[0][1]

        if highest_percentage >= 0.9:
            logger.info("HIGH CONFIDENCE: {} dominates with {:.1f}% - selecting immediately".format(
                final_policy[0][0], highest_percentage * 100
            ))
            policy_summary = {final_policy[0][0]: 1.0}
            return final_policy[0][0], policy_summary

        for move_name, visits in move_scores.items():
            if len(visits) > 1:
                mean = sum(visits) / len(visits)
                variance = sum((x - mean) ** 2 for x in visits) / len(visits)
                move_variance[move_name] = variance

        if move_variance:
            avg_variance = sum(move_variance.values()) / len(move_variance)
            if avg_variance < 0.01:
                logger.info("LOW VARIANCE: Results are consistent across samples (avg var: {:.4f})".format(avg_variance))
            else:
                logger.info("HIGH VARIANCE: Results vary across samples (avg var: {:.4f}) - position uncertain".format(avg_variance))

    final_policy = [i for i in final_policy if i[1] >= highest_percentage * 0.75]

    position_metrics = None
    risk_profiles: dict[str, MoveRiskProfile] | None = None
    line_evaluations = None
    if strategic_context:
        position_metrics = strategic_context.last_position_metrics
        if position_metrics is None:
            position_metrics = get_position_metrics(battle)
            strategic_context.update_position_metrics(position_metrics)
        candidate_moves = [policy[0] for policy in final_policy]
        risk_analyzer = strategic_context.risk_analyzer
        missing_moves = [mv for mv in candidate_moves if mv not in risk_analyzer.turn_cache]
        if missing_moves:
            risk_analyzer.evaluate_moves(battle, position_metrics, missing_moves)
        risk_profiles = {
            mv: risk_analyzer.turn_cache.get(mv)
            for mv in candidate_moves
            if mv in risk_analyzer.turn_cache
        }
        opponent_profile = strategic_context.opponent_model.get_profile()
        line_evaluations = evaluate_candidate_lines(
            battle,
            candidate_moves,
            opponent_profile,
            risk_profiles,
            position_metrics,
        )
        double_switch_bias = opponent_profile.get("double_switch_success", 0.0)
        sack_bias = opponent_profile.get("sack_timing", 0.0)
        opponent_tera_turn = opponent_profile.get("tera_turn")

        adjusted_policy = []
        for move_name, weight in final_policy:
            adjustment = 1.0
            if line_evaluations and move_name in line_evaluations:
                adjustment += line_evaluations[move_name]
            if risk_profiles and move_name in risk_profiles:
                profile = risk_profiles[move_name]
                momentum = position_metrics.get("momentum", 0.5)
                if momentum > 0.6:
                    adjustment *= max(0.4, 1.0 - profile.fail_chance)
                elif momentum < 0.4:
                    adjustment *= 1.0 + profile.expected_value / 150.0
                if profile.description == "finisher":
                    adjustment *= 1.0 + sack_bias * 0.3
            if move_name.startswith("switch"):
                adjustment *= max(0.2, 1.0 - double_switch_bias * 0.5)
            if move_name.endswith("-tera"):
                if opponent_tera_turn is None:
                    adjustment *= 0.9
                else:
                    adjustment *= 1.1
            adjusted_policy.append((move_name, max(weight * adjustment, 0.0)))

        total = sum(weight for _, weight in adjusted_policy)
        if total > 0:
            final_policy = [(move, weight / total) for move, weight in adjusted_policy]

    logger.info("Best Moves (ranked):")
    for i, policy in enumerate(final_policy):
        move_name = policy[0]
        percentage = policy[1]

        variance_note = ""
        if move_name in move_variance:
            variance_note = " (variance: {:.3f})".format(move_variance[move_name])

        logger.info(f"  {i+1}. {move_name} - {round(percentage * 100, 2)}%{variance_note}")

    policy_summary = {move: weight for move, weight in final_policy}
    choice = random.choices(final_policy, weights=[p[1] for p in final_policy])[0]
    return choice[0], policy_summary


def get_result_from_mcts(state: str, search_time_ms: int, index: int) -> MctsResult:
    logger.debug("Calling with {} state: {}".format(index, state))
    poke_engine_state = PokeEngineState.from_string(state)

    res = monte_carlo_tree_search(poke_engine_state, search_time_ms)
    logger.info("Iterations {}: {}".format(index, res.total_visits))
    return res


def calculate_position_criticality(battle: Battle, position_metrics: dict | None = None) -> float:
    criticality = 1.0

    if battle.team_preview:
        return criticality

    try:
        if position_metrics is None:
            if (
                hasattr(battle, "strategic_context")
                and battle.strategic_context
                and battle.strategic_context.last_position_metrics
            ):
                position_metrics = battle.strategic_context.last_position_metrics
            else:
                position_metrics = get_position_metrics(battle)
                if hasattr(battle, "strategic_context") and battle.strategic_context:
                    battle.strategic_context.update_position_metrics(position_metrics)

        user_alive = sum(1 for p in [battle.user.active] + battle.user.reserve if p and p.hp > 0)
        opp_alive = sum(1 for p in [battle.opponent.active] + battle.opponent.reserve if p and p.hp > 0)

        if user_alive <= 2:
            criticality *= 2.0
            logger.info("Endgame position detected (user has {} Pokemon) - increasing search time by 2x".format(user_alive))
        elif user_alive <= 3:
            criticality *= 1.5

        if opp_alive <= 2:
            criticality *= 1.3

        if battle.user.active and battle.user.active.max_hp > 0:
            hp_percentage = battle.user.active.hp / battle.user.active.max_hp
            if hp_percentage < 0.3:
                criticality *= 1.5
                logger.info("Low HP detected ({:.1f}%) - increasing search time by 1.5x".format(hp_percentage * 100))

        if position_metrics['opp_sweep_potential'] >= 0.7:
            criticality *= 1.8
            logger.info("High opponent sweep potential ({:.2f}) - increasing search time by 1.8x".format(
                position_metrics['opp_sweep_potential']
            ))

        if not position_metrics['has_win_condition']:
            criticality *= 1.3
            logger.info("No clear win condition - increasing search time by 1.3x")

        if position_metrics['momentum'] < 0.4:
            criticality *= 1.2
            logger.info("Losing momentum ({:.2f}) - increasing search time by 1.2x".format(
                position_metrics['momentum']
            ))

        criticality = min(criticality, 3.0)

    except Exception as e:
        logger.debug("Error calculating position metrics: {}".format(e))
        user_alive = sum(1 for p in [battle.user.active] + battle.user.reserve if p and p.hp > 0)
        if user_alive <= 2:
            criticality *= 2.0

    return criticality


def search_time_num_battles_randombattles(battle):
    revealed_pkmn = len(battle.opponent.reserve)
    if battle.opponent.active is not None:
        revealed_pkmn += 1

    opponent_active_num_moves = len(battle.opponent.active.moves)
    in_time_pressure = battle.time_remaining is not None and battle.time_remaining <= 60

    criticality = calculate_position_criticality(battle)
    time_manager = battle.strategic_context.time_manager
    adjusted_time = time_manager.allocate_search_time(battle, criticality)

    if (
        revealed_pkmn <= 3
        and battle.opponent.active.hp > 0
        and opponent_active_num_moves == 0
    ):
        num_battles_multiplier = 2 if in_time_pressure else 4
        return FoulPlayConfig.parallelism * num_battles_multiplier, int(adjusted_time // 2)
    else:
        num_battles_multiplier = 1 if in_time_pressure else 2
        return FoulPlayConfig.parallelism * num_battles_multiplier, adjusted_time


def search_time_num_battles_standard_battle(battle):
    opponent_active_num_moves = len(battle.opponent.active.moves)
    in_time_pressure = battle.time_remaining is not None and battle.time_remaining <= 60

    criticality = calculate_position_criticality(battle)
    time_manager = battle.strategic_context.time_manager
    adjusted_time = time_manager.allocate_search_time(battle, criticality)

    if (
        battle.team_preview
        or (battle.opponent.active.hp > 0 and opponent_active_num_moves == 0)
        or opponent_active_num_moves < 3
    ):
        num_battles_multiplier = 1 if in_time_pressure else 2
        return FoulPlayConfig.parallelism * num_battles_multiplier, adjusted_time
    else:
        return FoulPlayConfig.parallelism, adjusted_time


def check_ohko_risk(battle: Battle):
    if battle.team_preview or battle.opponent.active is None:
        return None, None

    current_hp = battle.user.active.hp
    if current_hp <= 0:
        return None, None

    ohko_threats = []

    for move in battle.opponent.active.moves:
        if move.disabled:
            continue

        try:
            _, opponent_rolls = poke_engine_get_damage_rolls(
                battle,
                "switch",
                move.name,
                False
            )

            if opponent_rolls and len(opponent_rolls) >= 2:
                min_damage = opponent_rolls[0]
                max_damage = opponent_rolls[1]

                if max_damage >= current_hp:
                    ohko_threats.append({
                        'move': move.name,
                        'min_damage': min_damage,
                        'max_damage': max_damage,
                        'current_hp': current_hp
                    })
                    logger.warning(
                        "OHKO THREAT DETECTED: {} can OHKO with {} (damage: {}-{}, current HP: {})".format(
                            battle.opponent.active.name,
                            move.name,
                            min_damage,
                            max_damage,
                            current_hp
                        )
                    )
        except Exception as e:
            logger.debug("Error calculating damage for move {}: {}".format(move.name, e))
            continue

    if ohko_threats:
        return True, ohko_threats
    return False, None


def calculate_matchup_score(our_pkmn, opp_pkmn):
    score = 0.0

    our_offensive_score = 0.0
    our_defensive_score = 0.0

    for move in our_pkmn.moves:
        move_data = all_move_json.get(move.name)
        if move_data and move_data.get('category') in ['physical', 'special']:
            move_type = move_data.get('type', 'normal')
            effectiveness = type_effectiveness_modifier(move_type, opp_pkmn.types)
            our_offensive_score = max(our_offensive_score, effectiveness)

    for our_type in our_pkmn.types:
        defensive_modifier = 1.0
        for opp_type in opp_pkmn.types:
            type_eff = type_effectiveness_modifier(opp_type, [our_type])
            defensive_modifier *= type_eff
        our_defensive_score += (2.0 - defensive_modifier)

    score = our_offensive_score * 10 + our_defensive_score * 5

    return score


def has_setup_move(pkmn):
    setup_moves = ['swordsdance', 'dragondance', 'nastyplot', 'quiverdance', 'calmmind', 'irondefense', 'agility', 'rockpolish', 'shellsmash', 'geomancy', 'bulkup', 'curse']
    return any(move.name in setup_moves for move in pkmn.moves)


def has_hazard_move(pkmn):
    hazard_moves = ['stealthrock', 'spikes', 'toxicspikes', 'stickyweb']
    return any(move.name in hazard_moves for move in pkmn.moves)


def predict_opponent_lead(opp_team: list) -> str:
    lead_scores = {}

    for opp_pkmn in opp_team:
        score = 0.0

        if has_setup_move(opp_pkmn):
            score += 20

        if has_hazard_move(opp_pkmn):
            score += 25

        weather_setters = ['pelipper', 'torkoal', 'hippowdon', 'tyranitar', 'ninetalesalola', 'politoed']
        if any(setter in opp_pkmn.name.lower() for setter in weather_setters):
            score += 30
            logger.debug("{} is weather setter - likely lead".format(opp_pkmn.name))

        common_leads = ['landorus', 'heatran', 'ferrothorn', 'clefable', 'excadrill', 'garchomp', 'gliscor']
        if any(lead in opp_pkmn.name.lower() for lead in common_leads):
            score += 10

        lead_scores[opp_pkmn.name] = score

    if lead_scores:
        predicted_lead = max(lead_scores.items(), key=lambda x: x[1])
        logger.info("Predicted opponent lead: {} (confidence: {:.1f})".format(predicted_lead[0], predicted_lead[1]))
        return predicted_lead[0]

    return opp_team[0].name if opp_team else None


def optimize_team_preview_order(battle: Battle) -> str:
    if not battle.team_preview:
        return None

    our_team = battle.user.reserve
    opp_team = battle.opponent.reserve

    if not our_team or not opp_team:
        return None

    scout_report = getattr(battle.strategic_context, "preview_report", None)
    if scout_report is None:
        try:
            scout_report = UsageScout().build_preview_report(battle)
            if battle.strategic_context:
                battle.strategic_context.preview_report = scout_report
        except Exception as exc:
            logger.debug("Unable to compute usage scouting report: {}".format(exc))
            scout_report = {}

    lead_probs = scout_report.get("leads", {}) if scout_report else {}
    threat_scores = scout_report.get("threats", {}) if scout_report else {}

    if lead_probs:
        predicted_opp_lead_name = max(lead_probs.items(), key=lambda x: x[1])[0]
        logger.info(
            "Predicted opponent lead: {} ({:.1f}%)".format(
                predicted_opp_lead_name, lead_probs[predicted_opp_lead_name] * 100
            )
        )
        logger.info(
            "Lead distribution: {}".format(
                ", ".join(
                    "{}:{:.0f}%".format(name, prob * 100)
                    for name, prob in sorted(lead_probs.items(), key=lambda x: x[1], reverse=True)
                )
            )
        )
    else:
        predicted_opp_lead_name = predict_opponent_lead(opp_team)

    weight_map = lead_probs or {
        pokemon.name: 1.0 / len([p for p in opp_team if p]) for pokemon in opp_team if pokemon
    }

    lead_scores = {}
    for our_pkmn in our_team:
        score = 0.0
        for opp_pkmn in opp_team:
            if not opp_pkmn:
                continue
            weight = weight_map.get(opp_pkmn.name, 0.0)
            matchup = calculate_matchup_score(our_pkmn, opp_pkmn)
            score += matchup * weight
            threat = threat_scores.get(opp_pkmn.name, 0.0)
            if threat:
                score += threat * matchup * 0.1
        score += our_pkmn.speed / 100.0
        lead_scores[our_pkmn.name] = score
        logger.info("Team Preview Score for {}: {:.1f}".format(our_pkmn.name, score))

    best_lead_name = max(lead_scores.items(), key=lambda x: x[1])[0]
    logger.info("Best lead selected: {} (score: {:.1f})".format(best_lead_name, lead_scores[best_lead_name]))

    for i, pkmn in enumerate(our_team):
        if pkmn.name == best_lead_name:
            return i + 1

    return 1


def find_best_move(battle: Battle) -> str:
    strategic_context = getattr(battle, "strategic_context", None)
    battle = deepcopy(battle)

    if strategic_context is None:
        strategic_context = StrategicContext()
    battle.strategic_context = strategic_context

    if battle.team_preview:
        optimal_lead_index = optimize_team_preview_order(battle)
        if optimal_lead_index:
            logger.info("Team preview: Using optimized lead (position {})".format(optimal_lead_index))
            return "switch {}".format(optimal_lead_index)

        battle.user.active = battle.user.reserve.pop(0)
        battle.opponent.active = battle.opponent.reserve.pop(0)

    battle.strategic_context.opponent_model.observe_turn(battle)
    position_metrics = get_position_metrics(battle)
    battle.strategic_context.update_position_metrics(position_metrics)
    battle.strategic_context.risk_analyzer.evaluate_moves(battle, position_metrics)
    if battle.strategic_context.risk_analyzer.turn_cache:
        logger.info("Risk/Reward snapshot:")
        ranked_risks = sorted(
            battle.strategic_context.risk_analyzer.turn_cache.values(),
            key=lambda p: p.expected_value,
            reverse=True,
        )
        for profile in ranked_risks[:3]:
            hazard_note = ""
            if profile.hazard_cost > 0:
                hazard_note = ", hazard -{:.0f}".format(profile.hazard_cost)
            logger.info(
                "  {} -> EV {:.1f} (range {:.0f}-{:.0f}), fail {:.0%}, variance {:.2f}{}".format(
                    profile.name,
                    profile.expected_value,
                    profile.min_damage,
                    profile.max_damage,
                    profile.fail_chance,
                    profile.variance,
                    hazard_note,
                )
            )

    if battle.battle_type == BattleType.RANDOM_BATTLE:
        num_battles, search_time_per_battle = search_time_num_battles_randombattles(
            battle
        )
        battles = prepare_random_battles(battle, num_battles)
    elif battle.battle_type == BattleType.BATTLE_FACTORY:
        num_battles, search_time_per_battle = search_time_num_battles_standard_battle(
            battle
        )
        battles = prepare_random_battles(battle, num_battles)
    elif battle.battle_type == BattleType.STANDARD_BATTLE:
        num_battles, search_time_per_battle = search_time_num_battles_standard_battle(
            battle
        )
        battles = prepare_battles(battle, num_battles)
    else:
        raise ValueError("Unsupported battle type: {}".format(battle.battle_type))

    has_ohko_risk, ohko_details = check_ohko_risk(battle)

    if has_ohko_risk:
        logger.warning("=" * 60)
        logger.warning("SURVIVAL ALERT: Opponent can OHKO you!")
        for threat in ohko_details:
            logger.warning("  - {} deals {}-{} damage (current HP: {})".format(
                threat['move'], threat['min_damage'], threat['max_damage'], threat['current_hp']
            ))
        alive_reserves = [p for p in battle.user.reserve if p.hp > 0]
        if alive_reserves:
            logger.warning("RECOMMENDATION: Strongly consider switching to avoid OHKO")
        logger.warning("=" * 60)

        search_time_per_battle = int(search_time_per_battle * 1.5)
        logger.info("OHKO threat detected - boosting search time by 1.5x to {}ms for deeper switch exploration".format(
            search_time_per_battle
        ))

    move_priorities = {}
    try:
        move_priorities = get_move_priorities(battle)
    except Exception as e:
        logger.debug("Error getting move priorities: {}".format(e))

    switch_rec = None
    try:
        switch_rec = get_switch_recommendation(battle)
        if switch_rec['should_switch']:
            logger.info("=" * 60)
            logger.info("SWITCH RECOMMENDATION: {}".format(switch_rec['reason']))
            logger.info("  Recommended: {} (score: {:.2f})".format(
                switch_rec['pokemon_name'],
                switch_rec['score']
            ))
            if 'switch_damage' in switch_rec:
                logger.info("  Switch-in damage: {}-{} (safe: {})".format(
                    switch_rec['switch_damage']['min'],
                    switch_rec['switch_damage']['max'],
                    switch_rec['switch_damage']['safe']
                ))
            logger.info("=" * 60)
    except Exception as e:
        logger.debug("Error getting switch recommendation: {}".format(e))

    def choose_policy_move():
        preserve_list = position_metrics.get('win_conditions', {}).get('preserve', [])
        if (
            switch_rec
            and switch_rec.get('should_switch')
            and (
                battle.user.active
                and battle.user.active.name in preserve_list
                or should_strongly_consider_switching(battle)
            )
        ):
            return "switch {}".format(switch_rec['pokemon_name'])
        if not move_priorities and battle.user.active and battle.user.active.moves:
            return battle.user.active.moves[0].name
        if not move_priorities:
            return "struggle"

        momentum = position_metrics.get('momentum', 0.5)

        def score(item):
            move_name, weight = item
            profile = battle.strategic_context.risk_analyzer.turn_cache.get(move_name)
            if profile:
                if momentum > 0.6:
                    return weight - profile.fail_chance
                else:
                    return weight + profile.expected_value / 150.0
            return weight

        return max(move_priorities.items(), key=score)[0]

    if battle.strategic_context.time_manager.should_skip_deep_search(battle):
        logger.warning("Time pressure detected - falling back to heuristic policy")
        fallback_choice = choose_policy_move()
        policy_summary = move_priorities if move_priorities else {fallback_choice: 1.0}
        battle.strategic_context.experience_tracker.record_turn(
            battle,
            fallback_choice,
            position_metrics,
            policy_summary,
            battle.strategic_context.risk_analyzer,
        )
        return fallback_choice

    logger.info("Searching for a move using MCTS...")
    logger.info(
        "Sampling {} battles at {}ms each".format(num_battles, search_time_per_battle)
    )
    with ProcessPoolExecutor(max_workers=FoulPlayConfig.parallelism) as executor:
        futures = []
        for index, (b, chance) in enumerate(battles):
            fut = executor.submit(
                get_result_from_mcts,
                battle_to_poke_engine_state(b).to_string(),
                search_time_per_battle,
                index,
            )
            futures.append((fut, chance, index))

    mcts_results = [(fut.result(), chance, index) for (fut, chance, index) in futures]
    choice, policy_summary = select_move_from_mcts_results(
        battle,
        mcts_results,
        battle.strategic_context,
    )
    logger.info("Choice: {}".format(choice))

    battle.strategic_context.experience_tracker.record_turn(
        battle,
        choice,
        position_metrics,
        policy_summary,
        battle.strategic_context.risk_analyzer,
    )

    return choice
