import logging
from typing import List

from data import all_move_json
from fp.battle import Battle
from fp.helpers import type_effectiveness_modifier
from fp.strategy.wincon import WinConditionTracker

logger = logging.getLogger(__name__)


def calculate_stealth_rock_damage(pkmn) -> float:
    if not pkmn or not pkmn.types:
        return 0.0

    multiplier = type_effectiveness_modifier('rock', pkmn.types)

    return multiplier / 8.0


def calculate_hp_advantage(battle: Battle) -> dict:
    our_total_hp = 0
    our_effective_hp = 0
    opp_total_hp = 0
    opp_effective_hp = 0

    has_hazards_user = battle.user.side_conditions.get('stealthrock', 0) > 0
    has_hazards_opp = battle.opponent.side_conditions.get('stealthrock', 0) > 0

    our_pokemon = [battle.user.active] + battle.user.reserve if battle.user.active else battle.user.reserve
    for pkmn in our_pokemon:
        if pkmn and pkmn.max_hp > 0:
            our_total_hp += pkmn.max_hp
            our_effective_hp += pkmn.hp

            if has_hazards_user and pkmn.hp > 0:
                hazard_damage = calculate_stealth_rock_damage(pkmn) * pkmn.max_hp
                our_effective_hp -= min(hazard_damage, pkmn.hp)

    opp_pokemon = [battle.opponent.active] + battle.opponent.reserve if battle.opponent.active else battle.opponent.reserve
    for pkmn in opp_pokemon:
        if pkmn and pkmn.max_hp > 0:
            opp_total_hp += pkmn.max_hp
            opp_effective_hp += pkmn.hp

            if has_hazards_opp and pkmn.hp > 0:
                hazard_damage = calculate_stealth_rock_damage(pkmn) * pkmn.max_hp
                opp_effective_hp -= min(hazard_damage, pkmn.hp)

    return {
        'our_hp': our_effective_hp,
        'opp_hp': opp_effective_hp,
        'hp_diff': our_effective_hp - opp_effective_hp,
        'hp_ratio': our_effective_hp / opp_effective_hp if opp_effective_hp > 0 else 2.0
    }


def assess_sweep_potential(side) -> float:
    if not side.active or side.active.hp == 0:
        return 0.0

    sweep_score = 0.0

    boost_value = 0
    for stat, boost in side.active.boosts.items():
        if stat in ['attack', 'special-attack', 'speed']:
            boost_value += boost

    if boost_value >= 2:
        sweep_score += 0.5
    if boost_value >= 4:
        sweep_score += 0.3

    setup_moves = {'swordsdance', 'dragondance', 'nastyplot', 'quiverdance', 'calmmind', 'shellsmash', 'geomancy'}
    has_setup = any(move.name in setup_moves for move in side.active.moves)

    if has_setup and boost_value >= 1:
        sweep_score += 0.2

    hp_percentage = side.active.hp / side.active.max_hp if side.active.max_hp > 0 else 0
    if hp_percentage > 0.7 and boost_value >= 2:
        sweep_score += 0.3

    return min(sweep_score, 1.0)


def detect_win_condition(battle: Battle) -> bool:
    if not battle.user.active or not battle.opponent.active:
        return False

    user_alive = sum(1 for p in [battle.user.active] + battle.user.reserve if p and p.hp > 0)
    opp_alive = sum(1 for p in [battle.opponent.active] + battle.opponent.reserve if p and p.hp > 0)

    if user_alive >= opp_alive + 2:
        return True

    hp_metrics = calculate_hp_advantage(battle)
    if hp_metrics['hp_ratio'] >= 2.5:
        return True

    if assess_sweep_potential(battle.user) >= 0.7 and opp_alive <= 2:
        return True

    return False


def calculate_momentum(battle: Battle) -> float:
    momentum = 0.5

    hp_metrics = calculate_hp_advantage(battle)

    momentum += (hp_metrics['hp_diff'] / 500) * 0.3

    user_alive = sum(1 for p in [battle.user.active] + battle.user.reserve if p and p.hp > 0)
    opp_alive = sum(1 for p in [battle.opponent.active] + battle.opponent.reserve if p and p.hp > 0)
    pokemon_diff = user_alive - opp_alive
    momentum += pokemon_diff * 0.1

    has_user_hazards = battle.opponent.side_conditions.get('stealthrock', 0) > 0
    has_opp_hazards = battle.user.side_conditions.get('stealthrock', 0) > 0

    if has_user_hazards and not has_opp_hazards:
        momentum += 0.1
    elif has_opp_hazards and not has_user_hazards:
        momentum -= 0.1

    user_sweep = assess_sweep_potential(battle.user)
    opp_sweep = assess_sweep_potential(battle.opponent)
    momentum += (user_sweep - opp_sweep) * 0.2

    return max(0.0, min(1.0, momentum))


def has_hazard_control(pokemon) -> bool:
    if not pokemon:
        return False
    return any(move.name in HAZARD_REMOVAL_MOVES for move in pokemon.moves)


def evaluate_hazard_pressure(battle: Battle) -> dict:
    user_hazard_damage = 0.0
    opp_hazard_damage = 0.0

    has_user_hazards = battle.user.side_conditions.get('stealthrock', 0) > 0
    has_opp_hazards = battle.opponent.side_conditions.get('stealthrock', 0) > 0

    if has_user_hazards:
        for pkmn in battler_team(battle.user):
            if pkmn and pkmn.max_hp > 0 and pkmn.hp > 0:
                damage_fraction = calculate_stealth_rock_damage(pkmn)
                user_hazard_damage += damage_fraction

    if has_opp_hazards:
        for pkmn in battler_team(battle.opponent):
            if pkmn and pkmn.max_hp > 0 and pkmn.hp > 0:
                damage_fraction = calculate_stealth_rock_damage(pkmn)
                opp_hazard_damage += damage_fraction

    our_control = sum(1 for p in battler_team(battle.user) if has_hazard_control(p))
    opp_control = sum(1 for p in battler_team(battle.opponent) if has_hazard_control(p))

    return {
        'user_hazard_damage': user_hazard_damage,
        'opp_hazard_damage': opp_hazard_damage,
        'hazard_advantage': opp_hazard_damage - user_hazard_damage,
        'clear_control': {
            'our_tools': our_control,
            'opp_tools': opp_control
        }
    }


def assess_speed_control(battle: Battle) -> dict:
    if not battle.user.active or not battle.opponent.active:
        return {'has_speed_control': False, 'speed_advantage': 0}

    user_speed = battle.user.active.speed
    opp_speed = battle.opponent.active.speed

    has_speed_control = user_speed > opp_speed

    user_speed_boost = battle.user.active.boosts.get('speed', 0)
    opp_speed_boost = battle.opponent.active.boosts.get('speed', 0)

    speed_advantage = user_speed_boost - opp_speed_boost

    user_priority = sum(1 for move in battle.user.active.moves if move.name in PRIORITY_MOVES)
    opp_priority = sum(1 for move in battle.opponent.active.moves if move.name in PRIORITY_MOVES)

    scarf_bonus = 0
    if battle.user.active.item == 'choicescarf':
        scarf_bonus += 1
    if battle.opponent.active.item == 'choicescarf':
        scarf_bonus -= 1

    trick_room = battle.trick_room

    return {
        'has_speed_control': has_speed_control,
        'speed_advantage': speed_advantage,
        'user_speed': user_speed,
        'opp_speed': opp_speed,
        'priority_advantage': user_priority - opp_priority,
        'scarf_advantage': scarf_bonus,
        'trick_room_active': trick_room
    }


def evaluate_tempo_control(battle: Battle, speed_metrics: dict, hazard_metrics: dict) -> dict:
    forcing = 0.0
    reactive = 0.0

    if hazard_metrics['hazard_advantage'] > 0:
        forcing += 0.3
    elif hazard_metrics['hazard_advantage'] < 0:
        reactive += 0.3

    if speed_metrics['has_speed_control']:
        forcing += 0.2
    else:
        reactive += 0.2

    if battle.user.active and battle.user.active.boosts.get('attack', 0) >= 2:
        forcing += 0.3
    if battle.opponent.active and battle.opponent.active.boosts.get('attack', 0) >= 2:
        reactive += 0.3

    tempo_score = max(0.0, min(1.0, 0.5 + forcing - reactive))
    return {
        'tempo_score': tempo_score,
        'forcing': forcing,
        'reactive': reactive
    }


def evaluate_positional_safety(battle: Battle) -> dict:
    if not battle.user.active or not battle.opponent.active:
        return {'safe_to_setup': True, 'opponent_setup_window': False, 'safety_score': 0.5}

    opponent_setup = any(move.name in SETUP_THREATS for move in battle.opponent.active.moves)
    our_priority = any(move.name in PRIORITY_MOVES for move in battle.user.active.moves)
    hp_ratio = battle.user.active.hp / battle.user.active.max_hp if battle.user.active.max_hp else 0

    safety_score = 0.5
    if opponent_setup:
        safety_score -= 0.2
    if hp_ratio < 0.4:
        safety_score -= 0.2
    if our_priority:
        safety_score += 0.1

    return {
        'safe_to_setup': hp_ratio > 0.5 and not opponent_setup,
        'opponent_setup_window': opponent_setup and hp_ratio < 0.5,
        'safety_score': max(0.0, min(1.0, safety_score))
    }


def evaluate_win_conditions(battle: Battle) -> dict:
    if hasattr(battle, 'strategic_context') and battle.strategic_context:
        return battle.strategic_context.wincon_tracker.evaluate(battle)
    tracker = WinConditionTracker()
    return tracker.evaluate(battle)


def get_position_metrics(battle: Battle) -> dict:
    metrics = {}

    metrics['hp_advantage'] = calculate_hp_advantage(battle)
    metrics['user_sweep_potential'] = assess_sweep_potential(battle.user)
    metrics['opp_sweep_potential'] = assess_sweep_potential(battle.opponent)
    metrics['has_win_condition'] = detect_win_condition(battle)
    metrics['hazard_pressure'] = evaluate_hazard_pressure(battle)
    metrics['speed_control'] = assess_speed_control(battle)
    metrics['tempo'] = evaluate_tempo_control(battle, metrics['speed_control'], metrics['hazard_pressure'])
    metrics['positional_safety'] = evaluate_positional_safety(battle)
    metrics['win_conditions'] = evaluate_win_conditions(battle)
    metrics['momentum'] = calculate_momentum(battle)

    logger.info("Position Metrics:")
    logger.info("  HP Advantage: {} (ratio: {:.2f})".format(
        int(metrics['hp_advantage']['hp_diff']),
        metrics['hp_advantage']['hp_ratio']
    ))
    logger.info("  Sweep Potential: User={:.2f}, Opp={:.2f}".format(
        metrics['user_sweep_potential'],
        metrics['opp_sweep_potential']
    ))
    logger.info("  Momentum: {:.2f} {}".format(
        metrics['momentum'],
        "(winning)" if metrics['momentum'] > 0.6 else "(losing)" if metrics['momentum'] < 0.4 else "(even)"
    ))
    logger.info("  Tempo: {:.2f} {}".format(
        metrics['tempo']['tempo_score'],
        "(forcing)" if metrics['tempo']['tempo_score'] > 0.55 else "(reacting)" if metrics['tempo']['tempo_score'] < 0.45 else "(neutral)"
    ))
    logger.info("  Hazard Game: ours={} opp={} clear_tools {}-{}".format(
        round(metrics['hazard_pressure']['user_hazard_damage'], 2),
        round(metrics['hazard_pressure']['opp_hazard_damage'], 2),
        metrics['hazard_pressure']['clear_control']['our_tools'],
        metrics['hazard_pressure']['clear_control']['opp_tools'],
    ))
    logger.info("  Positional Safety: {:.2f} (opp setup threat: {})".format(
        metrics['positional_safety']['safety_score'],
        metrics['positional_safety']['opponent_setup_window']
    ))
    wincon_info = metrics['win_conditions']
    if wincon_info.get('primary'):
        logger.info("  Primary wincons: {}".format(
            ", ".join("{} ({:.0f}%)".format(wc['name'], wc['hp_ratio'] * 100) for wc in wincon_info['primary'])
        ))
    if wincon_info.get('preserve'):
        logger.info("  Do-not-sack list: {}".format(", ".join(wincon_info['preserve'])))

    return metrics
HAZARD_REMOVAL_MOVES = {'defog', 'rapidspin', 'tidyup', 'mortalspin', 'courtchange', 'spinout'}
PRIORITY_MOVES = {'iceshard', 'extremespeed', 'suckerpunch', 'shadowsneak', 'machpunch', 'jetpunch', 'aquajet', 'bulletpunch', 'firstimpression'}
SETUP_THREATS = {'dragondance', 'swordsdance', 'nastyplot', 'shellsmash', 'acidarmor', 'irondefense', 'quiverdance', 'calmmind', 'bulkup'}


def battler_team(battler) -> List:
    return [battler.active] + battler.reserve if battler.active else battler.reserve
