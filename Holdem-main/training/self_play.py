# training/self_play.py
from __future__ import annotations

import argparse
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from env.poker_env import PokerEnv, ActionType, GameState, Street
from env.hand_eval import evaluate_5card

# ----------------------------
# Utilities: discretization / feature buckets
# ----------------------------

def _bucket(x: int, cuts: List[int]) -> int:
    """
    Return bucket index for integer x based on cut points.
    Example cuts [5,10] => buckets: (-inf..5)->0, (6..10)->1, (11..inf)->2
    """
    b = 0
    for c in cuts:
        if x <= c:
            return b
        b += 1
    return b

def hole_strength_bucket(hole) -> int:
    """
    Very simple preflop strength bucket (0..5).
    Uses pair / suited / high cards.
    """
    r1 = hole[0].rank.value
    r2 = hole[1].rank.value
    hi, lo = max(r1, r2), min(r1, r2)
    pair = (r1 == r2)
    suited = (hole[0].suit == hole[1].suit)

    # crude scoring
    score = 0
    if pair:
        score += 6
    if suited:
        score += 1
    if hi >= 13:
        score += 3
    elif hi >= 11:
        score += 2
    elif hi >= 9:
        score += 1

    gap = hi - lo
    if gap == 0:
        score += 2
    elif gap == 1:
        score += 1
    elif gap >= 5:
        score -= 1

    # map score -> bucket
    # (tune later if you want)
    if score >= 10:
        return 5
    if score >= 7:
        return 4
    if score >= 5:
        return 3
    if score >= 3:
        return 2
    if score >= 1:
        return 1
    return 0

def flop_hand_category_bucket(hole, board) -> int:
    """
    Since your simplified showdown is exactly 5 cards (2 hole + 3 flop),
    we can evaluate the 5-card category directly on flop.
    Returns category 0..8 (high card..straight flush).
    """
    if len(board) != 3:
        return -1
    cat, _k = evaluate_5card(list(hole) + list(board))
    return int(cat)

# ----------------------------
# State encoding (symmetric, "player-to-act" perspective)
# ----------------------------

def encode_state(s: GameState, player_id: int) -> Tuple[int, ...]:
    """
    Encodes state from the perspective of player_id ("me").
    This is crucial: it makes one Q-table usable for BOTH players (self-play symmetry).

    Returns a small discrete tuple; feel free to extend later.
    """
    me = player_id
    opp = 1 - player_id
    ps = s.players[me]
    os = s.players[opp]

    street = int(s.street.value)  # 0/1
    is_dealer = 1 if (s.dealer == me) else 0

    pot_b = _bucket(s.pot, [3, 8, 15, 25, 40, 60, 90])
    my_stack_b = _bucket(ps.stack, [0, 5, 10, 20, 35, 60, 90])
    opp_stack_b = _bucket(os.stack, [0, 5, 10, 20, 35, 60, 90])

    to_call = max(0, s.current_bet - ps.street_commit)
    to_call_b = _bucket(to_call, [0, 1, 2, 4, 8, 16, 32])

    curbet_b = _bucket(s.current_bet, [0, 1, 2, 4, 8, 16, 32])

    # cards:
    hole_b = hole_strength_bucket(ps.hole)
    flop_cat = flop_hand_category_bucket(ps.hole, s.board) if s.street == Street.FLOP else -1

    # street dynamics (you added checks_in_row in env)
    checks = getattr(s, "checks_in_row", 0)
    checks_b = min(int(checks), 2)  # cap

    return (
        street,
        is_dealer,
        pot_b,
        my_stack_b,
        opp_stack_b,
        to_call_b,
        curbet_b,
        hole_b,
        flop_cat,
        checks_b,
    )

# ----------------------------
# Tabular Q-Learning Agent
# ----------------------------

@dataclass
class QConfig:
    alpha: float = 0.10
    gamma: float = 0.95
    epsilon: float = 0.15
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.9995  # per hand

class TabularQLearner:
    def __init__(self, cfg: QConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        # Q[state][action_name] -> value
        self.Q: Dict[Tuple[int, ...], Dict[str, float]] = {}

    def _ensure_state(self, state: Tuple[int, ...], legal: List[ActionType]) -> None:
        if state not in self.Q:
            self.Q[state] = {}
        for a in legal:
            self.Q[state].setdefault(a.name, 0.0)

    def choose_action(self, state: Tuple[int, ...], legal: List[ActionType], explore: bool = True) -> ActionType:
        assert len(legal) > 0
        self._ensure_state(state, legal)

        eps = self.cfg.epsilon if explore else 0.0
        if self.rng.random() < eps:
            return self.rng.choice(legal)

        # greedy (tie-break random)
        qd = self.Q[state]
        best_v = None
        best_actions: List[ActionType] = []
        for a in legal:
            v = qd.get(a.name, 0.0)
            if best_v is None or v > best_v:
                best_v = v
                best_actions = [a]
            elif v == best_v:
                best_actions.append(a)
        return self.rng.choice(best_actions)

    def update(
        self,
        state: Tuple[int, ...],
        action: ActionType,
        reward: float,
        next_state: Optional[Tuple[int, ...]],
        next_legal: List[ActionType],
        done: bool,
    ) -> None:
        # ensure entries exist
        self._ensure_state(state, [action] + next_legal)

        q_sa = self.Q[state][action.name]
        if done or next_state is None or len(next_legal) == 0:
            target = reward
        else:
            self._ensure_state(next_state, next_legal)
            max_next = max(self.Q[next_state].get(a.name, 0.0) for a in next_legal)
            target = reward + self.cfg.gamma * max_next

        self.Q[state][action.name] = q_sa + self.cfg.alpha * (target - q_sa)

    def decay_epsilon(self) -> None:
        self.cfg.epsilon = max(self.cfg.epsilon_min, self.cfg.epsilon * self.cfg.epsilon_decay)

# ----------------------------
# Self-play training loop
# ----------------------------

def train_self_play(
    hands: int,
    seed: int,
    out_path: str,
    starting_stack: int = 100,
    sb: int = 1,
    bb: int = 2,
    max_raises_per_street: int = 4,
) -> None:
    env = PokerEnv(seed=seed, starting_stack=starting_stack, sb=sb, bb=bb, max_raises_per_street=max_raises_per_street)
    agent = TabularQLearner(QConfig(), seed=seed + 999)

    rng = random.Random(seed + 2027)

    total_p0_profit = 0
    total_steps = 0

    for h in range(1, hands + 1):
        env.reset()  # dealer alternates internally if you coded it that way
        done = False
        steps = 0

        while not done:
            s = env.get_state()
            pid = s.to_act  # current actor
            state = encode_state(s, pid)

            legal = env.legal_actions()
            a = agent.choose_action(state, legal, explore=True)

            # compute reward from acting player's perspective using stack delta
            stacks_before = (s.players[0].stack, s.players[1].stack)

            _ns, _r_p0, done, _info = env.step(a)

            s2 = env.get_state()
            stacks_after = (s2.players[0].stack, s2.players[1].stack)

            reward_actor = stacks_after[pid] - stacks_before[pid]

            if done:
                next_state = None
                next_legal = []
            else:
                pid2 = s2.to_act
                next_state = encode_state(s2, pid2)
                next_legal = env.legal_actions()

            agent.update(
                state=state,
                action=a,
                reward=float(reward_actor),
                next_state=next_state,
                next_legal=next_legal,
                done=done,
            )

            steps += 1
            total_steps += 1

            # safety against accidental infinite loops
            if steps > 200:
                raise RuntimeError("Too many steps in one hand. Check env street-ending logic.")

        # bookkeeping: p0 profit at end of hand
        end_s = env.get_state()
        p0_profit = end_s.players[0].stack - starting_stack
        total_p0_profit += p0_profit

        agent.decay_epsilon()

        if h % 500 == 0:
            avg_steps = total_steps / h
            print(f"[{h}/{hands}] eps={agent.cfg.epsilon:.4f} avg_steps={avg_steps:.2f} total_p0_profit={total_p0_profit}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "Q": agent.Q,

        # IMPORTANT:
        # Do NOT pickle custom class instances (like QConfig),
        # otherwise loading in a different module will fail.
        # Save config as plain dict instead.
        "cfg": {
            "alpha": agent.cfg.alpha,
            "gamma": agent.cfg.gamma,
            "epsilon": agent.cfg.epsilon,
            "epsilon_min": agent.cfg.epsilon_min,
            "epsilon_decay": agent.cfg.epsilon_decay,
        },

        "meta": {
            "hands": hands,
            "seed": seed,
            "starting_stack": starting_stack,
            "sb": sb,
            "bb": bb,
            "max_raises_per_street": max_raises_per_street,
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)


    print("Saved Q-table to:", out_path)

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", type=str, default="outputs/models/q_table.pkl")
    ap.add_argument("--starting_stack", type=int, default=100)
    ap.add_argument("--sb", type=int, default=1)
    ap.add_argument("--bb", type=int, default=2)
    ap.add_argument("--max_raises_per_street", type=int, default=4)
    args = ap.parse_args()

    train_self_play(
        hands=args.hands,
        seed=args.seed,
        out_path=args.out,
        starting_stack=args.starting_stack,
        sb=args.sb,
        bb=args.bb,
        max_raises_per_street=args.max_raises_per_street,
    )

if __name__ == "__main__":
    main()
