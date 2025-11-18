import logging
from fp.battle import Battle
from fp.helpers import type_effectiveness_modifier
from data import all_move_json

logger = logging.getLogger(__name__)


def calculate_move_effectiveness(move_name: str, defender_types: list) -> float:
    try:
        move_data = all_move_json[move_name]
        move_type = move_data.get('type', 'normal')
        return type_effectiveness_modifier(move_type, defender_types)
    except KeyError:
        return 1.0


def is_status_move(move_name: str) -> bool:
    try:
        return all_move_json[move_name].get('category') == 'status'
    except KeyError:
        return False


def is_setup_move(move_name: str) -> bool:
    setup_moves = {
        'swordsdance', 'dragondance', 'nastyplot', 'quiverdance', 'calmmind',
        'irondefense', 'agility', 'rockpolish', 'shellsmash', 'geomancy',
        'bulkup', 'curse', 'coil', 'honeclaws', 'workup', 'growth',
        'acupressure', 'bellydrum', 'chargebeam', 'cosmicpower'
    }
    return move_name in setup_moves


def can_setup_safely(battle: Battle) -> bool:
    if not battle.user.active or not battle.opponent.active:
        return False

    user_hp_percentage = battle.user.active.hp / battle.user.active.max_hp if battle.user.active.max_hp > 0 else 0

    if user_hp_percentage < 0.5:
        return False

    for move in battle.opponent.active.moves:
        if move.disabled:
            continue
        effectiveness = calculate_move_effectiveness(move.name, battle.user.active.types)
        if effectiveness >= 2.0:
            return False

    return True


def hits_opponent_team_super_effectively(move_name: str, opponent_team: list) -> bool:
    count = 0
    for pkmn in opponent_team:
        if pkmn.hp > 0:
            effectiveness = calculate_move_effectiveness(move_name, pkmn.types)
            if effectiveness >= 2.0:
                count += 1
    return count >= 2


def calculate_move_priority(battle: Battle, move_name: str) -> float:
    if not battle.opponent.active:
        return 1.0

    priority_score = 1.0

    effectiveness = calculate_move_effectiveness(move_name, battle.opponent.active.types)

    if effectiveness == 0:
        priority_score *= 0.01
        logger.debug("Move {} is immune - heavily deprioritized".format(move_name))
    elif effectiveness < 0.5:
        priority_score *= 0.3
        logger.debug("Move {} is not very effective ({}) - deprioritized".format(move_name, effectiveness))
    elif effectiveness == 1.0:
        priority_score *= 1.0
    elif effectiveness >= 2.0:
        priority_score *= 2.0
        logger.debug("Move {} is super effective ({}) - boosted".format(move_name, effectiveness))

    if is_setup_move(move_name) and can_setup_safely(battle):
        priority_score *= 2.5
        logger.debug("Setup move {} is safe - heavily boosted".format(move_name))

    opponent_team = [battle.opponent.active] + battle.opponent.reserve
    if hits_opponent_team_super_effectively(move_name, opponent_team):
        priority_score *= 1.5
        logger.debug("Move {} hits multiple opponents super effectively - boosted".format(move_name))

    return priority_score


def get_move_priorities(battle: Battle) -> dict:
    if not battle.user.active:
        return {}

    priorities = {}
    for move in battle.user.active.moves:
        if move.current_pp <= 0:
            priorities[move.name] = 0.01
        else:
            priorities[move.name] = calculate_move_priority(battle, move.name)

    total = sum(priorities.values())
    if total > 0:
        priorities = {k: v / total for k, v in priorities.items()}

    logger.info("Move priorities: {}".format(
        ", ".join("{}: {:.2f}".format(k, v) for k, v in sorted(priorities.items(), key=lambda x: x[1], reverse=True))
    ))

    return priorities


def should_strongly_consider_switching(battle: Battle) -> bool:
    if not battle.user.active or not battle.opponent.active:
        return False

    our_moves_ineffective = True
    for move in battle.user.active.moves:
        if move.current_pp > 0 and not is_status_move(move.name):
            effectiveness = calculate_move_effectiveness(move.name, battle.opponent.active.types)
            if effectiveness >= 1.0:
                our_moves_ineffective = False
                break

    if our_moves_ineffective:
        logger.info("All damaging moves are not very effective or immune - switching recommended")
        return True

    their_best_effectiveness = 0
    for move in battle.opponent.active.moves:
        if not move.disabled:
            effectiveness = calculate_move_effectiveness(move.name, battle.user.active.types)
            their_best_effectiveness = max(their_best_effectiveness, effectiveness)

    if their_best_effectiveness >= 4.0:
        logger.info("Opponent has 4x super effective move - switching strongly recommended")
        return True

    return False
