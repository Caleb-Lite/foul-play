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

logger = logging.getLogger(__name__)


def select_move_from_mcts_results(mcts_results: list[(MctsResult, float, int)]) -> str:
    final_policy = {}
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
            final_policy[s1_option.move_choice] = final_policy.get(
                s1_option.move_choice, 0
            ) + (sample_chance * (s1_option.visits / mcts_result.total_visits))

    final_policy = sorted(final_policy.items(), key=lambda x: x[1], reverse=True)

    # Consider all moves that are close to the best move
    highest_percentage = final_policy[0][1]
    final_policy = [i for i in final_policy if i[1] >= highest_percentage * 0.75]
    logger.info("Best Moves (ranked):")
    for i, policy in enumerate(final_policy):
        logger.info(f"  {i+1}. {policy[0]} - {round(policy[1] * 100, 2)}%")

    choice = random.choices(final_policy, weights=[p[1] for p in final_policy])[0]
    return choice[0]


def get_result_from_mcts(state: str, search_time_ms: int, index: int) -> MctsResult:
    logger.debug("Calling with {} state: {}".format(index, state))
    poke_engine_state = PokeEngineState.from_string(state)

    res = monte_carlo_tree_search(poke_engine_state, search_time_ms)
    logger.info("Iterations {}: {}".format(index, res.total_visits))
    return res


def calculate_position_criticality(battle: Battle) -> float:
    criticality = 1.0

    if battle.team_preview:
        return criticality

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

    if battle.opponent.active:
        setup_moves = ['swordsdance', 'dragondance', 'nastyplot', 'quiverdance', 'calmmind', 'irondefense', 'agility', 'rockpolish', 'shellsmash', 'geomancy']
        for move in battle.opponent.active.moves:
            if move.name in setup_moves:
                criticality *= 1.5
                logger.info("Opponent has setup move {} - increasing search time by 1.5x".format(move.name))
                break

    criticality = min(criticality, 3.0)

    return criticality


def search_time_num_battles_randombattles(battle):
    revealed_pkmn = len(battle.opponent.reserve)
    if battle.opponent.active is not None:
        revealed_pkmn += 1

    opponent_active_num_moves = len(battle.opponent.active.moves)
    in_time_pressure = battle.time_remaining is not None and battle.time_remaining <= 60

    base_time = FoulPlayConfig.search_time_ms
    criticality = calculate_position_criticality(battle)
    adjusted_time = int(base_time * criticality)

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

    base_time = FoulPlayConfig.search_time_ms
    criticality = calculate_position_criticality(battle)
    adjusted_time = int(base_time * criticality)

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


def optimize_team_preview_order(battle: Battle) -> str:
    if not battle.team_preview:
        return None

    our_team = battle.user.reserve
    opp_team = battle.opponent.reserve

    if not our_team or not opp_team:
        return None

    lead_scores = {}

    for our_pkmn in our_team:
        score = 0.0

        for opp_pkmn in opp_team:
            matchup = calculate_matchup_score(our_pkmn, opp_pkmn)
            score += matchup

        if has_setup_move(our_pkmn):
            score += 15
            logger.info("{} has setup move - bonus +15".format(our_pkmn.name))

        if has_hazard_move(our_pkmn):
            score += 10
            logger.info("{} has hazard move - bonus +10".format(our_pkmn.name))

        avg_speed = sum(p.speed for p in our_team) / len(our_team)
        if our_pkmn.speed > avg_speed:
            score += 5

        lead_scores[our_pkmn.name] = score
        logger.info("Team Preview Score for {}: {:.1f}".format(our_pkmn.name, score))

    best_lead_name = max(lead_scores.items(), key=lambda x: x[1])[0]
    logger.info("Best lead selected: {} (score: {:.1f})".format(best_lead_name, lead_scores[best_lead_name]))

    for i, pkmn in enumerate(our_team):
        if pkmn.name == best_lead_name:
            return i + 1

    return 1


def find_best_move(battle: Battle) -> str:
    battle = deepcopy(battle)

    if battle.team_preview:
        optimal_lead_index = optimize_team_preview_order(battle)
        if optimal_lead_index:
            logger.info("Team preview: Using optimized lead (position {})".format(optimal_lead_index))
            return "switch {}".format(optimal_lead_index)

        battle.user.active = battle.user.reserve.pop(0)
        battle.opponent.active = battle.opponent.reserve.pop(0)

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
    choice = select_move_from_mcts_results(mcts_results)
    logger.info("Choice: {}".format(choice))
    return choice
