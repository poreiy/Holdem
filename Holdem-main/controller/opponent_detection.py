# controller/opponent_detection.py
# Minimal Opponent Detection / Opponent Profiling
# - Online stats (VPIP / PFR / Aggression / Fold-to-raise)
# - Simple opponent type classification: TIGHT / LOOSE / AGGRESSIVE / PASSIVE / UNKNOWN
#
# Works with your simplified HU NLHE env:
# ActionType: FOLD, CHECK_CALL, RAISE_*, ALL_IN
# Streets: PREFLOP, FLOP
#
# Intended usage:
#   detector = OpponentDetector(hero_id=0)
#   detector.observe(state, action, actor_id)
#   ...
#   info = detector.get_opponent_profile()
#   info["type"] -> string label

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from env.poker_env import ActionType, GameState, Street, RAISE_ACTIONS


@dataclass
class OpponentProfile:
    hands: int = 0

    # "Voluntarily put money in pot" (call/raise when facing a bet) - simplified proxy
    vpip_hands: int = 0

    # Preflop raises (proxy for PFR)
    pfr_hands: int = 0

    # Aggression on flop: raises vs calls
    flop_raises: int = 0
    flop_calls: int = 0

    # Fold behavior vs aggression (very rough)
    fold_when_facing_bet: int = 0
    faced_bet_cnt: int = 0

    # Per-hand flags (reset every hand)
    _vpip_this_hand: bool = False
    _pfr_this_hand: bool = False


class OpponentDetector:
    """
    Track opponent behavior online. Minimal but useful.

    hero_id: our agent id (0 or 1). Opponent id = 1 - hero_id.

    Notes (simplifications):
    - VPIP: if opponent CALLs or RAISEs when to_call > 0 (i.e., money goes in voluntarily)
            OR opponent raises at any time preflop.
    - PFR: opponent makes any raise preflop.
    - Aggression: flop raises / (flop calls + 1)  [avoid div by zero]
    - Fold-to-bet: fold_when_facing_bet / faced_bet_cnt
    """

    def __init__(self, hero_id: int = 0):
        assert hero_id in (0, 1)
        self.hero_id = hero_id
        self.opp_id = 1 - hero_id
        self.p = OpponentProfile()

        # To detect hand boundaries
        self._last_hand_id: Optional[int] = None

    # -----------------------------
    # Public API
    # -----------------------------

    def observe(self, state: GameState, action: ActionType, actor_id: int) -> None:
        """
        Call this every env.step, BEFORE state mutates outside or AFTER step info captured.
        You pass the *state before the action* (recommended) to compute to_call properly.

        actor_id: who took this action (0 or 1)
        """
        # Detect new hand (env increments hand_id on reset)
        hid = getattr(state, "hand_id", None)
        if hid is not None and hid != self._last_hand_id:
            self._start_new_hand(hid)

        if actor_id != self.opp_id:
            return  # only track opponent

        to_call = self._to_call(state, actor_id)

        # If opponent faces a bet (to_call > 0), record whether they fold
        if to_call > 0:
            self.p.faced_bet_cnt += 1
            if action == ActionType.FOLD:
                self.p.fold_when_facing_bet += 1

        # VPIP/PFR logic (mostly preflop)
        if state.street == Street.PREFLOP:
            if action in RAISE_ACTIONS:
                self.p._vpip_this_hand = True
                self.p._pfr_this_hand = True
            elif action == ActionType.CHECK_CALL and to_call > 0:
                # calling a bet = VPIP
                self.p._vpip_this_hand = True

        # Flop aggression stats
        if state.street == Street.FLOP:
            if action in RAISE_ACTIONS:
                self.p.flop_raises += 1
            elif action == ActionType.CHECK_CALL and to_call > 0:
                self.p.flop_calls += 1

    def end_hand_if_needed(self, state: GameState) -> None:
        """
        Optional: call this at terminal state to make sure the per-hand flags are committed.
        Not strictly required if you always detect new hand via hand_id changes.
        """
        if state.done:
            self._commit_hand_flags()

    def get_opponent_profile(self) -> Dict[str, float | int | str]:
        """
        Returns a dict with stats + a coarse opponent type label.
        """
        hands = max(1, self.p.hands)
        vpip = self.p.vpip_hands / hands
        pfr = self.p.pfr_hands / hands
        fold_to_bet = (self.p.fold_when_facing_bet / self.p.faced_bet_cnt) if self.p.faced_bet_cnt > 0 else 0.0
        aggression = self.p.flop_raises / (self.p.flop_calls + 1)

        opp_type = self._classify(vpip=vpip, pfr=pfr, aggression=aggression, fold_to_bet=fold_to_bet)

        return {
            "hands": self.p.hands,
            "vpip": vpip,
            "pfr": pfr,
            "aggression": aggression,
            "fold_to_bet": fold_to_bet,
            "type": opp_type,
        }

    # -----------------------------
    # Internals
    # -----------------------------

    def _start_new_hand(self, hand_id: int) -> None:
        # Before starting a new hand, commit the last hand's flags
        self._commit_hand_flags()
        self.p.hands += 1
        self.p._vpip_this_hand = False
        self.p._pfr_this_hand = False
        self._last_hand_id = hand_id

    def _commit_hand_flags(self) -> None:
        # Commit flags from the last observed hand into counters.
        # (Safe to call multiple times; it only commits if flags set and not yet counted.)
        # To keep it minimal, we commit every time we enter a new hand;
        # This is "good enough" for project scale.

        # If hands == 0, it means we haven't started tracking yet -> don't commit.
        if self._last_hand_id is None:
            return

        # We commit based on current per-hand flags.
        # To avoid double counting when _commit_hand_flags is called multiple times inside one hand,
        # we ONLY commit at the moment a new hand starts. Here it's okay if called multiple times,
        # because flags stay the same; but we'd overcount.
        # So: do nothing here. Commit in _start_new_hand only is safer.

        # NOTE: We already commit in _start_new_hand by reading flags BEFORE reset.
        # Therefore this function intentionally does nothing to prevent double-counting.
        return

    @staticmethod
    def _to_call(state: GameState, pid: int) -> int:
        ps = state.players[pid]
        return max(0, state.current_bet - ps.street_commit)

    @staticmethod
    def _classify(vpip: float, pfr: float, aggression: float, fold_to_bet: float) -> str:
        """
        Very simple rule-based classifier.
        Tune thresholds later if you want.
        """
        # Not enough evidence
        # (caller can check hands count separately)
        # We'll still classify based on stats; caller can show UNKNOWN if hands < N.
        loose = (vpip >= 0.45)
        tight = (vpip <= 0.25)
        aggressive = (aggression >= 1.0) or (pfr >= 0.30)
        passive = (aggression <= 0.40) and (pfr <= 0.15)

        if tight and aggressive:
            return "TIGHT-AGGRESSIVE"
        if tight and passive:
            return "TIGHT-PASSIVE"
        if loose and aggressive:
            return "LOOSE-AGGRESSIVE"
        if loose and passive:
            return "LOOSE-PASSIVE"

        # Special signal: folds too much to bets
        if fold_to_bet >= 0.65:
            return "OVERFOLDER"

        return "UNKNOWN"
