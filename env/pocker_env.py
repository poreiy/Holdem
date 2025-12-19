# env/poker_env.py
# Two-street (Preflop + Flop) Heads-Up No-Limit Texas Hold'em (simplified)
# Action space (discrete):
#   - FOLD
#   - CHECK_CALL
#   - RAISE_1_3_POT
#   - RAISE_2_3_POT
#   - RAISE_POT
#   - ALL_IN
#
# Notes:
# - This env is intentionally "CS188-friendly": clean, deterministic, easy to plug into EV/RL/risk modules.
# - It supports two betting streets (Preflop, Flop). After flop betting ends -> showdown.
# - Uses a simple but correct chip accounting model with "to_call" and per-player committed bets per street.
#
# You can extend:
# - richer info in `info`
# - more detailed hand equity in features/ layer
# - opponent modeling outside env

from env.hand_eval import compare_showdown_hands
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import random

# ----------------------------
# Cards / Deck
# ----------------------------

class Suit(Enum):
    HEARTS = "H"
    DIAMONDS = "D"
    CLUBS = "C"
    SPADES = "S"

class Rank(Enum):
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

@dataclass(frozen=True)
class Card:
    suit: Suit
    rank: Rank

    def __repr__(self) -> str:
        r = self.rank.value
        rs = str(r) if r <= 10 else {11: "J", 12: "Q", 13: "K", 14: "A"}[r]
        return f"{rs}{self.suit.value}"

class Deck:
    def __init__(self, rng: random.Random):
        self._rng = rng
        self.cards: List[Card] = [Card(s, r) for s in Suit for r in Rank]
        self._rng.shuffle(self.cards)

    def deal(self) -> Card:
        return self.cards.pop()

# ----------------------------
# Actions / Streets
# ----------------------------

class Street(Enum):
    PREFLOP = 0
    FLOP = 1

class ActionType(Enum):
    FOLD = auto()
    CHECK_CALL = auto()
    RAISE_1_3_POT = auto()
    RAISE_2_3_POT = auto()
    RAISE_POT = auto()
    ALL_IN = auto()

RAISE_ACTIONS = {
    ActionType.RAISE_1_3_POT,
    ActionType.RAISE_2_3_POT,
    ActionType.RAISE_POT,
    ActionType.ALL_IN,
}

# ----------------------------
# State
# ----------------------------

@dataclass
class PlayerState:
    stack: int
    hole: List[Card] = field(default_factory=list)
    in_hand: bool = True
    # committed in current street (used to compute to_call etc.)
    street_commit: int = 0

@dataclass
class GameState:
    p0_stack_prev: int = 0
    # public
    street: Street
    board: List[Card]
    pot: int
    # action
    to_act: int                 # 0 or 1
    dealer: int                 # button index
    # betting
    current_bet: int            # highest street_commit among active players
    last_aggressor: Optional[int]  # who last raised in this street (None if none)
    # players
    players: List[PlayerState]
    # terminal flags
    done: bool = False
    winner: Optional[int] = None   # 0/1 if terminal by fold or showdown
    # bookkeeping (optional)
    hand_id: int = 0

# ----------------------------
# Hand evaluation (placeholder)
# ----------------------------
# IMPORTANT: Keep env independent. You can swap this with a real evaluator later.
# For now we include a tiny showdown function that returns a random winner if tie.
# Replace with env/hand_eval.py when ready.

def _determine_showdown_winner_rng(rng: random.Random, p0_hole: List[Card], p1_hole: List[Card], board3: List[Card]) -> int:
    # Placeholder: RANDOM winner (replace with real evaluation).
    # This keeps training loop functional before you plug in a real evaluator.
    return 0 if rng.random() < 0.5 else 1

# ----------------------------
# PokerEnv
# ----------------------------

class PokerEnv:
    """
    Two-player (Heads-Up), two-street (Preflop+Flop) simplified NLHE.

    Blinds:
      SB = 1, BB = 2 by default.
    Stacks:
      each starts with 100 by default.
    """

    def __init__(
        self,
        seed: int = 0,
        starting_stack: int = 100,
        sb: int = 1,
        bb: int = 2,
        rake: int = 0,          # keep 0 for class project
        max_raises_per_street: int = 4,
    ):
        self.rng = random.Random(seed)
        self.starting_stack = starting_stack
        self.sb = sb
        self.bb = bb
        self.rake = rake
        self.max_raises_per_street = max_raises_per_street

        self._deck: Optional[Deck] = None
        self._state: Optional[GameState] = None
        self._hand_counter = 0
        self._raises_in_street = 0

    # ---------- Public API ----------

    def reset(self, dealer: Optional[int] = None) -> GameState:
        """
        Start a new hand. dealer=0/1 sets who is Button (acts first preflop).
        If None, alternates each hand.
        """
        self._hand_counter += 1
        if dealer is None:
            dealer = (self._hand_counter + 1) % 2  # alternate, deterministic
        assert dealer in (0, 1)

        self._deck = Deck(self.rng)

        players = [
            PlayerState(stack=self.starting_stack),
            PlayerState(stack=self.starting_stack),
        ]

        # Deal private cards (2 each)
        for _ in range(2):
            players[0].hole.append(self._deck.deal())
            players[1].hole.append(self._deck.deal())

        # Post blinds: dealer is SB in HU, other is BB
        sb_i = dealer
        bb_i = 1 - dealer

        pot = 0
        pot += self._post_blind(players[sb_i], self.sb)
        pot += self._post_blind(players[bb_i], self.bb)

        # Preflop: SB acts first in HU (dealer acts first preflop)
        to_act = sb_i

        current_bet = max(players[0].street_commit, players[1].street_commit)

        self._raises_in_street = 0

        self._state = GameState(
            street=Street.PREFLOP,
            board=[],
            pot=pot,
            to_act=to_act,
            dealer=dealer,
            current_bet=current_bet,
            last_aggressor=None,
            players=players,
            done=False,
            winner=None,
            hand_id=self._hand_counter,
            p0_stack_prev=players[0].stack,
        )
        return self._copy_state()

    def step(self, action: ActionType) -> Tuple[GameState, int, bool, Dict]:
        """
        Apply action for current player.
        Returns: (next_state, reward_for_player0, done, info)
        Reward is always from player0 perspective (RL-friendly).
        """
        assert self._state is not None, "Call reset() first."
        s = self._state
        p0_before = s.players[0].stack
        assert not s.done, "Hand is already done."

        p = s.to_act
        opp = 1 - p
        ps = s.players[p]
        os = s.players[opp]

        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"Illegal action: {action}. Legal: {legal}")

        info: Dict = {"street": s.street.name, "to_act": p, "action": action.name}

        # --- FOLD ---
        if action == ActionType.FOLD:
            ps.in_hand = False
            s.done = True
            s.winner = opp
            # winner takes pot
            s.players[opp].stack += s.pot
            s.pot = 0
            return self._finalize_and_return(info, p0_before)

        # --- CHECK/CALL ---
        if action == ActionType.CHECK_CALL:
            to_call = s.current_bet - ps.street_commit
            if to_call > 0:
                paid = self._pay(ps, to_call)
                s.pot += paid
                ps.street_commit += paid
            # after call, betting may end -> advance street or showdown
            s.to_act = opp
            self._maybe_end_street_after_non_raise()
            return self._transition_and_return(info, p0_before)

        # --- RAISES (including all-in) ---
        target_commit = self._compute_raise_target_commit(action, p)
        # ensure target_commit > current_bet (a raise), unless all-in smaller (should not happen with legality checks)
        add = max(0, target_commit - ps.street_commit)
        paid = self._pay(ps, add)
        ps.street_commit += paid
        s.pot += paid

        # update bet
        if ps.street_commit > s.current_bet:
            s.current_bet = ps.street_commit
            s.last_aggressor = p
            self._raises_in_street += 1

        # after raise, opponent acts
        s.to_act = opp
        return self._transition_and_return(info, p0_before)

    def legal_actions(self) -> List[ActionType]:
        """
        Legal actions for current to_act.
        """
        assert self._state is not None
        s = self._state
        if s.done:
            return []

        p = s.to_act
        ps = s.players[p]
        opp = 1 - p
        os = s.players[opp]

        if not ps.in_hand:
            return []  # should not happen

        actions: List[ActionType] = []

        to_call = s.current_bet - ps.street_commit
        if to_call > 0:
            actions.append(ActionType.FOLD)
            actions.append(ActionType.CHECK_CALL)  # call
        else:
            actions.append(ActionType.CHECK_CALL)  # check

        # raising allowed if player has chips and raise cap not exceeded
        if ps.stack > 0 and self._raises_in_street < self.max_raises_per_street:
            # If opponent is all-in already (cannot raise beyond), disable raises:
            if os.stack == 0 and os.street_commit == s.current_bet:
                return actions

            # check each raise size feasibility
            for a in [ActionType.RAISE_1_3_POT, ActionType.RAISE_2_3_POT, ActionType.RAISE_POT, ActionType.ALL_IN]:
                if self._is_raise_legal(a, p):
                    actions.append(a)

        # de-duplicate while preserving order
        seen = set()
        uniq = []
        for a in actions:
            if a not in seen:
                uniq.append(a)
                seen.add(a)
        return uniq

    def get_state(self) -> GameState:
        assert self._state is not None
        return self._copy_state()

    # ---------- Internal helpers ----------

    def _post_blind(self, ps: PlayerState, amount: int) -> int:
        paid = self._pay(ps, amount)
        ps.street_commit += paid
        return paid

    @staticmethod
    def _pay(ps: PlayerState, amount: int) -> int:
        paid = min(amount, ps.stack)
        ps.stack -= paid
        return paid

    def _is_raise_legal(self, action: ActionType, p: int) -> bool:
        s = self._state
        assert s is not None
        ps = s.players[p]

        target = self._compute_raise_target_commit(action, p)
        if target <= s.current_bet:
            return False

        # must be able to put at least (target - street_commit)
        need = target - ps.street_commit
        return need <= (ps.stack + 0)  # stack remaining

    def _compute_raise_target_commit(self, action: ActionType, p: int) -> int:
        """
        Returns desired street_commit after raise for player p.
        This is a simplified NL bet-sizing model:
        - Compute pot "as-is" (includes all previous contributions).
        - For pot-fraction raises, set raise amount based on pot, then target = current_bet + raise_amount.
        - ALL_IN sets target to street_commit + remaining stack.
        """
        s = self._state
        assert s is not None
        ps = s.players[p]

        if action == ActionType.ALL_IN:
            return ps.street_commit + ps.stack

        # pot fraction sizes (very simplified, but stable)
        pot_now = s.pot  # includes both players contributions
        if action == ActionType.RAISE_1_3_POT:
            raise_amt = max(1, pot_now // 3)
        elif action == ActionType.RAISE_2_3_POT:
            raise_amt = max(1, (2 * pot_now) // 3)
        elif action == ActionType.RAISE_POT:
            raise_amt = max(1, pot_now)
        else:
            raise ValueError(f"Not a raise action: {action}")

        # target commit cannot exceed all-in
        target = s.current_bet + raise_amt
        all_in_target = ps.street_commit + ps.stack
        return min(target, all_in_target)

    def _maybe_end_street_after_non_raise(self) -> None:
        """
        Called after CHECK/CALL. Determine if betting round ends.
        Simple HU logic:
        - If both players have equal commits (no one owes to_call) and the last action was a check/call,
          and either:
            a) no aggressor in street and both checked (we detect via equality and whose turn), or
            b) aggressor existed and has been called (equality achieved),
          then street ends.
        """
        s = self._state
        assert s is not None
        if s.players[0].street_commit != s.players[1].street_commit:
            return

        # If equalized, street betting is closed.
        self._advance_street_or_showdown()

    def _advance_street_or_showdown(self) -> None:
        s = self._state
        assert s is not None
        if s.street == Street.PREFLOP:
            # Deal flop (3 cards)
            assert self._deck is not None
            s.board = [self._deck.deal(), self._deck.deal(), self._deck.deal()]
            s.street = Street.FLOP

            # Reset street betting bookkeeping
            for ps in s.players:
                ps.street_commit = 0
            s.current_bet = 0
            s.last_aggressor = None
            self._raises_in_street = 0

            # On flop, first to act is BB (non-dealer) in HU
            s.to_act = 1 - s.dealer
        else:
            # Flop betting ended -> showdown
            self._resolve_showdown()

    def _resolve_showdown(self) -> None:
        s = self._state
        assert s is not None
        assert s.street == Street.FLOP
        assert len(s.board) == 3
    
        # If any player folded earlier, should already be terminal
        if not s.players[0].in_hand:
            s.done = True
            s.winner = 1
        elif not s.players[1].in_hand:
            s.done = True
            s.winner = 0
        else:
            # Use real hand evaluator
            from env.hand_eval import compare_showdown_hands
    
            w = compare_showdown_hands(
                s.players[0].hole,
                s.players[1].hole,
                s.board,
                allow_tie=True
            )
            s.done = True
            s.winner = w  # 0/1/None
    
        # Award pot (handle tie)
        if s.winner is None:
            # Split pot as evenly as possible
            half = s.pot // 2
            s.players[0].stack += half
            s.players[1].stack += (s.pot - half)  # give remainder to player1 deterministically
        else:
            s.players[s.winner].stack += s.pot
    
        s.pot = 0


    def _transition_and_return(self, info: Dict, p0_before: int) -> Tuple[GameState, int, bool, Dict]:
        """
        Per-step incremental reward from player0 perspective:
          reward = p0_after - p0_before
        Over a whole hand, the sum of step rewards equals final profit/loss.
        """
        s = self._state
        assert s is not None
    
        p0_after = s.players[0].stack
        reward = p0_after - p0_before
    
        info["p0_stack"] = p0_after
        info["p1_stack"] = s.players[1].stack
        info["pot"] = s.pot
        info["board"] = list(s.board)
    
        return self._copy_state(), reward, s.done, info

    def _finalize_and_return(self, info: Dict, p0_before: int) -> Tuple[GameState, int, bool, Dict]:
        # fold terminals go here
        return self._transition_and_return(info, p0_before)

    def _copy_state(self) -> GameState:
        # A lightweight deep copy (avoid accidental mutation outside env)
        assert self._state is not None
        s = self._state
        players_copy = [
            PlayerState(
                stack=ps.stack,
                hole=list(ps.hole),
                in_hand=ps.in_hand,
                street_commit=ps.street_commit,
            )
            for ps in s.players
        ]
        return GameState(
            street=s.street,
            board=list(s.board),
            pot=s.pot,
            to_act=s.to_act,
            dealer=s.dealer,
            current_bet=s.current_bet,
            last_aggressor=s.last_aggressor,
            players=players_copy,
            done=s.done,
            winner=s.winner,
            hand_id=s.hand_id,
        )

# ----------------------------
# Quick sanity check (optional)
# ----------------------------

if __name__ == "__main__":
    env = PokerEnv(seed=42)
    s = env.reset(dealer=0)
    print("RESET:", s.street, "dealer=", s.dealer, "to_act=", s.to_act)
    print("P0 hole:", s.players[0].hole, "P1 hole:", s.players[1].hole)
    print("Legal:", env.legal_actions())

    # play random until done
    while True:
        a = env.rng.choice(env.legal_actions())
        s, r, done, info = env.step(a)
        print("Act:", a, "| street:", s.street, "| pot:", info["pot"], "| stacks:", info["p0_stack"], info["p1_stack"])
        if done:
            print("DONE winner:", s.winner, "reward(p0):", r, "board:", info["board"])
            break
