import logging
from fp.battle import Battle
from fp.helpers import type_effectiveness_modifier
from fp.search.poke_engine_helpers import poke_engine_get_damage_rolls
from data import all_move_json

logger = logging.getLogger(__name__)


def calculate_defensive_matchup_score(our_pkmn, opp_pkmn) -> float:
    if not our_pkmn or not opp_pkmn:
        return 0.5

    score = 0.5

    for move in opp_pkmn.moves:
        if move.disabled or move.current_pp <= 0:
            continue

        try:
            move_data = all_move_json[move.name]
            if move_data.get('category') in ['physical', 'special']:
                move_type = move_data.get('type', 'normal')
                effectiveness = type_effectiveness_modifier(move_type, our_pkmn.types)

                if effectiveness == 0:
                    score += 0.3
                elif effectiveness < 0.5:
                    score += 0.2
                elif effectiveness == 1.0:
                    score += 0.05
                elif effectiveness >= 2.0:
                    score -= 0.2
                elif effectiveness >= 4.0:
                    score -= 0.4
        except KeyError:
            continue

    for our_type in our_pkmn.types:
        for opp_type in opp_pkmn.types:
            our_offensive_eff = type_effectiveness_modifier(our_type, [opp_type])
            if our_offensive_eff >= 2.0:
                score += 0.1

    return max(0.0, min(1.0, score))


def find_best_switch_in(battle: Battle) -> dict:
    if not battle.opponent.active:
        return None

    alive_reserves = [p for p in battle.user.reserve if p.hp > 0]

    if not alive_reserves:
        return None

    switch_scores = {}

    for pkmn in alive_reserves:
        score = calculate_defensive_matchup_score(pkmn, battle.opponent.active)

        hp_percentage = pkmn.hp / pkmn.max_hp if pkmn.max_hp > 0 else 0
        score *= (0.5 + 0.5 * hp_percentage)

        switch_scores[pkmn.name] = {
            'pokemon': pkmn,
            'score': score,
            'defensive_matchup': calculate_defensive_matchup_score(pkmn, battle.opponent.active)
        }

    if not switch_scores:
        return None

    best_switch = max(switch_scores.items(), key=lambda x: x[1]['score'])

    logger.info("Best switch option: {} (score: {:.2f}, defensive: {:.2f})".format(
        best_switch[0],
        best_switch[1]['score'],
        best_switch[1]['defensive_matchup']
    ))

    return {
        'pokemon_name': best_switch[0],
        'pokemon': best_switch[1]['pokemon'],
        'score': best_switch[1]['score'],
        'all_options': switch_scores
    }


def calculate_switch_damage(battle: Battle, switch_pokemon) -> dict:
    if not battle.opponent.active or not switch_pokemon:
        return {'min': 0, 'max': 0, 'safe': True}

    total_min = 0
    total_max = 0

    for move in battle.opponent.active.moves:
        if move.disabled or move.current_pp <= 0:
            continue

        try:
            move_data = all_move_json[move.name]
            if move_data.get('category') not in ['physical', 'special']:
                continue

            move_type = move_data.get('type', 'normal')
            effectiveness = type_effectiveness_modifier(move_type, switch_pokemon.types)

            base_damage = move_data.get('basePower', 0) * effectiveness

            total_min = max(total_min, base_damage * 0.85)
            total_max = max(total_max, base_damage * 1.0)

        except KeyError:
            continue

    switch_hp = switch_pokemon.hp
    is_safe = (total_max < switch_hp * 0.5) if switch_hp > 0 else False

    return {
        'min': int(total_min),
        'max': int(total_max),
        'safe': is_safe,
        'current_hp': switch_hp
    }


def should_switch_defensively(battle: Battle) -> bool:
    if not battle.user.active or not battle.opponent.active:
        return False

    current_hp_pct = battle.user.active.hp / battle.user.active.max_hp if battle.user.active.max_hp > 0 else 0

    if current_hp_pct < 0.2:
        logger.info("Very low HP ({:.1f}%) - defensive switch recommended".format(current_hp_pct * 100))
        return True

    their_best_effectiveness = 0
    for move in battle.opponent.active.moves:
        if not move.disabled and move.current_pp > 0:
            try:
                move_data = all_move_json[move.name]
                if move_data.get('category') in ['physical', 'special']:
                    move_type = move_data.get('type', 'normal')
                    effectiveness = type_effectiveness_modifier(move_type, battle.user.active.types)
                    their_best_effectiveness = max(their_best_effectiveness, effectiveness)
            except KeyError:
                continue

    if their_best_effectiveness >= 4.0:
        logger.info("Opponent has 4x super effective coverage - defensive switch recommended")
        return True

    our_best_effectiveness = 0
    for move in battle.user.active.moves:
        if move.current_pp > 0:
            try:
                move_data = all_move_json[move.name]
                if move_data.get('category') in ['physical', 'special']:
                    move_type = move_data.get('type', 'normal')
                    effectiveness = type_effectiveness_modifier(move_type, battle.opponent.active.types)
                    our_best_effectiveness = max(our_best_effectiveness, effectiveness)
            except KeyError:
                continue

    if our_best_effectiveness <= 0.5 and their_best_effectiveness >= 2.0:
        logger.info("Bad matchup (we: {}, them: {}) - defensive switch recommended".format(
            our_best_effectiveness, their_best_effectiveness
        ))
        return True

    return False


def get_switch_recommendation(battle: Battle) -> dict:
    should_switch = should_switch_defensively(battle)

    if not should_switch:
        return {
            'should_switch': False,
            'reason': 'Current matchup acceptable'
        }

    best_switch = find_best_switch_in(battle)

    if not best_switch:
        return {
            'should_switch': False,
            'reason': 'No better switch options available'
        }

    switch_damage = calculate_switch_damage(battle, best_switch['pokemon'])

    if best_switch['score'] > 0.6 or switch_damage['safe']:
        return {
            'should_switch': True,
            'pokemon_name': best_switch['pokemon_name'],
            'score': best_switch['score'],
            'switch_damage': switch_damage,
            'reason': 'Better defensive matchup available'
        }

    return {
        'should_switch': False,
        'reason': 'Switch options not significantly better'
    }
