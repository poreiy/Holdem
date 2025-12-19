# agents/rl_only_agent.py
from __future__ import annotations

import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from env.poker_env import ActionType, GameState, Street

# IMPORTANT:
# We reuse the SAME state encoding as training/self_play.py
# so the keys match exactly.
from training.self_play import encode_state, hole_strength_bucket


@dataclass
class RLOpt:
    model_path: str = "outputs/models/q_table.pkl"
    explore: bool = False          # True for ε-greedy during play; False for eval
    epsilon: float = 0.0           # only used when explore=True
    seed: int = 0

    # Preflop fallback behavior (until EV module is plugged in)
    preflop_tightness: int = 3     # 0..5 (bigger = tighter)
    allow_allin_preflop: bool = False


class RlOnlyAgent:
    """
    A minimal agent for your project:

    - Preflop: rule-based fallback (tight fold/call/raise)
    - Flop: greedy action from Q-table (optionally ε-greedy)

    Interface:
      act(state, legal_actions) -> ActionType
    """

    def __init__(self, opt: RLOpt):
        self.opt = opt
        self.rng = random.Random(opt.seed)
        self.Q: Dict[Tuple[int, ...], Dict[str, float]] = {}
        self._load_model(opt.model_path)

    def _load_model(self, path: str) -> None:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.Q = payload.get("Q", {})
        # If you want, you can also read payload["cfg"] / payload["meta"]

    def act(self, state: GameState, legal_actions: List[ActionType]) -> ActionType:
        assert len(legal_actions) > 0

        # Use different policies by street
        if state.street == Street.PREFLOP:
            return self._act_preflop(state, legal_actions)
        else:
            return self._act_flop_rl(state, legal_actions)

    # ----------------------------
    # Preflop fallback (simple & safe)
    # ----------------------------

    def _act_preflop(self, state: GameState, legal: List[ActionType]) -> ActionType:
        """
        Simple preflop rule until you implement EV module.

        Idea:
        - Compute a crude hole strength bucket 0..5
        - If weak -> fold to pressure, otherwise call/check
        - If strong -> raise sometimes (but avoid all-in unless allowed)
        """
        pid = state.to_act
        ps = state.players[pid]

        strength = hole_strength_bucket(ps.hole)  # 0..5

        # Determine if call is needed
        to_call = max(0, state.current_bet - ps.street_commit)

        # If there is a bet to call:
        if to_call > 0:
            # fold weak hands
            if strength < self.opt.preflop_tightness and ActionType.FOLD in legal:
                return ActionType.FOLD

            # with decent+ hands, call
            if ActionType.CHECK_CALL in legal:
                # sometimes raise with very strong hands
                if strength >= 5:
                    for a in [ActionType.RAISE_POT, ActionType.RAISE_2_3_POT, ActionType.RAISE_1_3_POT]:
                        if a in legal:
                            return a
                return ActionType.CHECK_CALL

        # If no bet to call (check option):
        if ActionType.CHECK_CALL in legal:
            # raise occasionally if strong
            if strength >= 5:
                for a in [ActionType.RAISE_2_3_POT, ActionType.RAISE_1_3_POT]:
                    if a in legal:
                        return a
            return ActionType.CHECK_CALL

        # Fallback
        return legal[0]

    # ----------------------------
    # Flop RL policy
    # ----------------------------

    def _act_flop_rl(self, state: GameState, legal: List[ActionType]) -> ActionType:
        pid = state.to_act
        key = encode_state(state, pid)

        # ε-greedy exploration if enabled
        if self.opt.explore and self.opt.epsilon > 0.0:
            if self.rng.random() < self.opt.epsilon:
                return self.rng.choice(legal)

        # If state unseen, pick a safe default:
        qd = self.Q.get(key)
        if not qd:
            # prefer check/call if available
            if ActionType.CHECK_CALL in legal:
                return ActionType.CHECK_CALL
            return self.rng.choice(legal)

        # Greedy over legal actions (tie-break random)
        best_v: Optional[float] = None
        best: List[ActionType] = []

        for a in legal:
            v = float(qd.get(a.name, 0.0))
            if best_v is None or v > best_v:
                best_v = v
                best = [a]
            elif v == best_v:
                best.append(a)

        # Optional: avoid ALL_IN on flop unless it is clearly best
        if ActionType.ALL_IN in best and len(best) > 1:
            best = [a for a in best if a != ActionType.ALL_IN] or best

        return self.rng.choice(best)
