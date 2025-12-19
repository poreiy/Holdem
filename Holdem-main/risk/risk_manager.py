# risk/risk_manager.py
from env.poker_env import ActionType, Street

SAFE_BANNED = {
    ActionType.ALL_IN,
    ActionType.RAISE_POT,
}

def filter_actions(legal_actions, state, mode):
    if mode == "SAFE":
        actions = [a for a in legal_actions if a not in SAFE_BANNED]
        return actions if actions else legal_actions
    return legal_actions
