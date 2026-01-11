# controller/mode_switch.py
# controller/mode_switch.py

def decide_mode(bankroll, start_bankroll, drawdown, state, opponent_type="UNKNOWN"):
    # 资金回撤 → 安全模式
    if drawdown > 0.3 * start_bankroll:
        return "SAFE"
    if bankroll < 0.7 * start_bankroll:
        return "SAFE"

    # 对手驱动的切换（新增）
    if opponent_type in ("LOOSE-AGGRESSIVE", "TIGHT-AGGRESSIVE"):
        return "SAFE"

    if opponent_type in ("OVERFOLDER", "LOOSE-PASSIVE"):
        return "AGGRESSIVE"

    return "AGGRESSIVE"

