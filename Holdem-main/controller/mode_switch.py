# controller/mode_switch.py
def decide_mode(bankroll, start_bankroll, drawdown, state):
    if drawdown > 0.3 * start_bankroll:
        return "SAFE"
    if bankroll < 0.7 * start_bankroll:
        return "SAFE"
    return "AGGRESSIVE"
