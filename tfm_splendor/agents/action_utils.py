from splendor.splendor.gym.envs.utils import ALL_ACTIONS
from splendor.splendor.gym.envs.actions import Action
from splendor.splendor.splendor_model import SplendorState
from splendor.splendor.types import ActionType

def action_to_index(action, state, agent_index) -> int  :
    action_element = Action.to_action_element(action, state, agent_index)
    return ALL_ACTIONS.index(action_element)


COLOR_ORDER = ("black", "red", "yellow", "green", "blue", "white")

def gems_key(gems):
    if gems is None:
        return None
    return tuple(gems.get(c, 0) for c in COLOR_ORDER)

def action_key(action: Action):
    return (
        action.type_enum,
        gems_key(action.collected_gems),
        gems_key(action.returned_gems),
        None if action.position is None else (
            action.position.tier,
            action.position.card_index,
            action.position.reserved_index,
        ),
        action.noble_index,
    )

ACTION_KEY_TO_INDEX = {
    action_key(a): i for i, a in enumerate(ALL_ACTIONS)
}


def action_to_index_2(action, state, agent_index) -> int:
    action_element = Action.to_action_element(action, state, agent_index)
    return ACTION_KEY_TO_INDEX[action_key(action_element)]