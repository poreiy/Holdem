# evaluation/evaluate.py
from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from env.poker_env import PokerEnv, ActionType, GameState
from agents.rl_only_agent import RlOnlyAgent, RLOpt


# ----------------------------
# Baseline agents
# ----------------------------

class BaseAgent:
    def act(self, state: GameState, legal_actions: List[ActionType]) -> ActionType:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, state: GameState, legal_actions: List[ActionType]) -> ActionType:
        return self.rng.choice(legal_actions)


class TightRuleAgent(BaseAgent):
    """
    Simple tight-passive baseline:
    - If facing a bet: fold often unless CALL is cheap
    - Rare raises
    """
    def __init__(self, seed: int = 0, fold_prob_when_facing_bet: float = 0.60):
        self.rng = random.Random(seed)
        self.fold_prob = fold_prob_when_facing_bet

    def act(self, state: GameState, legal_actions: List[ActionType]) -> ActionType:
        pid = state.to_act
        ps = state.players[pid]
        to_call = max(0, state.current_bet - ps.street_commit)

        if to_call > 0 and ActionType.FOLD in legal_actions:
            if self.rng.random() < self.fold_prob:
                return ActionType.FOLD
            if ActionType.CHECK_CALL in legal_actions:
                return ActionType.CHECK_CALL

        if ActionType.CHECK_CALL in legal_actions:
            return ActionType.CHECK_CALL

        return legal_actions[0]


def make_agent(kind: str, seed: int, model_path: str) -> BaseAgent:
    from agents.mode_switch_agent import ModeSwitchAgent

    kind = kind.lower()
    if kind == "no_mode":
        return RlOnlyAgent(RLOpt(model_path=model_path, explore=False, seed=seed))
    if kind == "mode":
        return ModeSwitchAgent(model_path=model_path, seed=seed)
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "tight":
        return TightRuleAgent(seed=seed)
    if kind == "rl":
        return RlOnlyAgent(RLOpt(model_path=model_path, explore=False, seed=seed))
    raise ValueError(f"Unknown agent kind: {kind}. Use random/tight/rl")


# ----------------------------
# Metrics
# ----------------------------

@dataclass
class EvalResult:
    hands: int
    p0_wins: int
    p1_wins: int
    ties: int

    total_p0_profit: int          # chips
    avg_p0_profit_per_hand: float # chips/hand
    bb_per_100: float

    max_drawdown: int             # chips (>=0)
    final_bankroll: int           # p0 end-of-session bankroll

    # diagnostics
    avg_steps: float
    showdown_rate: float
    fold_rate: float
    avg_peak_pot: float


def compute_max_drawdown(equity_curve: List[int]) -> int:
    """
    equity_curve: bankroll values over time (per hand)
    max_drawdown = max(peak - current)
    """
    peak = equity_curve[0]
    mdd = 0
    for x in equity_curve:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > mdd:
            mdd = dd
    return mdd


# ----------------------------
# Match runner
# ----------------------------

def play_match(
    hands: int,
    seed: int,
    p0_kind: str,
    p1_kind: str,
    model_path: str,
    starting_stack: int = 100,
    sb: int = 1,
    bb: int = 2,
    max_raises_per_street: int = 4,
) -> EvalResult:
    env = PokerEnv(seed=seed, starting_stack=starting_stack, sb=sb, bb=bb, max_raises_per_street=max_raises_per_street)

    p0 = make_agent(p0_kind, seed=seed + 10, model_path=model_path)
    p1 = make_agent(p1_kind, seed=seed + 20, model_path=model_path)

    p0_wins = p1_wins = ties = 0
    total_p0_profit = 0

    bankroll = starting_stack
    equity_curve = [bankroll]  # per hand

    # diagnostics
    total_steps = 0
    showdown_cnt = 0
    fold_cnt = 0
    total_peak_pot = 0.0

    for _h in range(1, hands + 1):
        s = env.reset()

        # IMPORTANT:
        # env reward is per-step delta stack for p0:
        # reward = p0_after - p0_before
        # so single-hand profit must be sum of rewards over the hand.
        hand_profit = 0
        steps_this_hand = 0

        peak_pot_this_hand = s.pot  # include blinds
        last_info: Dict = {}

        while not s.done:
            legal = env.legal_actions()
            # 决策
            a = p0.act(s, legal) if s.to_act == 0 else p1.act(s, legal)

            # 让 ModeSwitchAgent 观察对手行为
            if hasattr(p0, "observe_opponent"):
                p0.observe_opponent(s, a, s.to_act)
            if hasattr(p1, "observe_opponent"):
                p1.observe_opponent(s, a, s.to_act)

            # 执行动作
            s, r, _done, info = env.step(a)
            hand_profit += int(r)  # r is already int in env signature
            # 通知 agent 本步 reward（如果支持）
            if hasattr(p0, "observe_reward") and s.to_act != 0:
                p0.observe_reward(r)
            if hasattr(p1, "observe_reward") and s.to_act != 1:
                p1.observe_reward(-r)

            steps_this_hand += 1

            last_info = info if isinstance(info, dict) else {}
            pot_now = last_info.get("pot", None)
            if pot_now is not None:
                try:
                    peak_pot_this_hand = max(peak_pot_this_hand, float(pot_now))
                except Exception:
                    pass

        total_steps += steps_this_hand
        total_peak_pot += peak_pot_this_hand

        # terminal diagnostics: showdown vs fold
        # If both players still in_hand at terminal => showdown; else fold.
        if s.players[0].in_hand and s.players[1].in_hand:
            showdown_cnt += 1
        else:
            fold_cnt += 1

        # profit / bankroll update (session-level)
        p0_profit = hand_profit
        total_p0_profit += p0_profit

        bankroll += p0_profit
        equity_curve.append(bankroll)

        # win/loss aligned with profit sign
        if p0_profit > 0:
            p0_wins += 1
        elif p0_profit < 0:
            p1_wins += 1
        else:
            ties += 1

    avg_profit = total_p0_profit / hands
    bb100 = (avg_profit / bb) * 100.0  # chips->BB, per 100 hands
    mdd = compute_max_drawdown(equity_curve)

    avg_steps = total_steps / hands
    showdown_rate = showdown_cnt / hands
    fold_rate = fold_cnt / hands
    avg_peak_pot = total_peak_pot / hands

    return EvalResult(
        hands=hands,
        p0_wins=p0_wins,
        p1_wins=p1_wins,
        ties=ties,
        total_p0_profit=total_p0_profit,
        avg_p0_profit_per_hand=avg_profit,
        bb_per_100=bb100,
        max_drawdown=mdd,
        final_bankroll=equity_curve[-1],
        avg_steps=avg_steps,
        showdown_rate=showdown_rate,
        fold_rate=fold_rate,
        avg_peak_pot=avg_peak_pot,
    )


def print_result(p0_kind: str, p1_kind: str, res: EvalResult) -> None:
    wr = res.p0_wins / res.hands
    print("=" * 60)
    print(f"Match: p0={p0_kind} vs p1={p1_kind}")
    print(f"Hands: {res.hands}")
    print(f"Wins (by profit sign): p0={res.p0_wins}, p1={res.p1_wins}, ties={res.ties} | p0 winrate={wr:.3f}")
    print(f"Total p0 profit (chips): {res.total_p0_profit}")
    print(f"Avg p0 profit/hand (chips): {res.avg_p0_profit_per_hand:.4f}")
    print(f"BB/100: {res.bb_per_100:.3f}")
    print(f"Max Drawdown (chips): {res.max_drawdown}")
    print(f"Final session bankroll (p0): {res.final_bankroll}")
    print(f"Avg steps/hand: {res.avg_steps:.2f}")
    print(f"Showdown rate: {res.showdown_rate:.3f} | Fold rate: {res.fold_rate:.3f}")
    print(f"Avg peak pot (chips): {res.avg_peak_pot:.2f}")
    print("=" * 60)


def maybe_save_csv(path: str, p0_kind: str, p1_kind: str, res: EvalResult) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    header = [
        "hands", "p0_kind", "p1_kind",
        "p0_wins", "p1_wins", "ties",
        "total_p0_profit", "avg_p0_profit_per_hand",
        "bb_per_100", "max_drawdown", "final_bankroll",
        "avg_steps", "showdown_rate", "fold_rate", "avg_peak_pot",
    ]
    row = [
        res.hands, p0_kind, p1_kind,
        res.p0_wins, res.p1_wins, res.ties,
        res.total_p0_profit, f"{res.avg_p0_profit_per_hand:.6f}",
        f"{res.bb_per_100:.6f}", res.max_drawdown, res.final_bankroll,
        f"{res.avg_steps:.6f}", f"{res.showdown_rate:.6f}", f"{res.fold_rate:.6f}", f"{res.avg_peak_pot:.6f}",
    ]

    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)

    print(f"Saved results to {path}")


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--p0", type=str, default="rl", help="random | tight | rl")
    ap.add_argument("--p1", type=str, default="random", help="random | tight | rl")
    ap.add_argument("--model", type=str, default="outputs/models/q_table.pkl")
    ap.add_argument("--csv", type=str, default="", help="optional output csv path, e.g., outputs/reports/eval.csv")

    ap.add_argument("--starting_stack", type=int, default=100)
    ap.add_argument("--sb", type=int, default=1)
    ap.add_argument("--bb", type=int, default=2)
    ap.add_argument("--max_raises_per_street", type=int, default=4)
    args = ap.parse_args()

    res = play_match(
        hands=args.hands,
        seed=args.seed,
        p0_kind=args.p0,
        p1_kind=args.p1,
        model_path=args.model,
        starting_stack=args.starting_stack,
        sb=args.sb,
        bb=args.bb,
        max_raises_per_street=args.max_raises_per_street,
    )
    print_result(args.p0, args.p1, res)
    maybe_save_csv(args.csv, args.p0, args.p1, res)


if __name__ == "__main__":
    main()
