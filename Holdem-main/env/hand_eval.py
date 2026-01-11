# env/hand_eval.py
# Hand evaluation utilities for simplified HU NLHE (2 hole + 3 flop = exactly 5 cards at showdown).
#
# Design goals:
# - Pure Python, no external deps
# - Deterministic, comparable hand score tuples
# - Easy to extend later (e.g., 7-card best-of-5 if you add turn/river)

from __future__ import annotations

from typing import List, Tuple, Optional
from collections import Counter
from itertools import combinations

# IMPORTANT:
# This file expects you to use the same Card / Suit / Rank definitions as env/poker_env.py.
# So we import Card type only for type hints (optional).
try:
    from env.poker_env import Card  # if running as a package
except Exception:
    try:
        from poker_env import Card   # if running locally
    except Exception:
        Card = object  # type: ignore


# ----------------------------
# Core evaluation (5 cards)
# ----------------------------

# Category ranks (bigger is better)
# 8 Straight Flush
# 7 Four of a Kind
# 6 Full House
# 5 Flush
# 4 Straight
# 3 Three of a Kind
# 2 Two Pair
# 1 One Pair
# 0 High Card

HandScore = Tuple[int, Tuple[int, ...]]  # (category, kickers tuple)

def _ranks(cards: List[Card]) -> List[int]:
    return sorted((c.rank.value for c in cards), reverse=True)

def _is_flush(cards: List[Card]) -> bool:
    suits = [c.suit for c in cards]
    return len(set(suits)) == 1

def _straight_high(ranks_desc: List[int]) -> Optional[int]:
    """
    Returns high card of straight if straight, else None.
    Handles wheel straight A-5 (A,5,4,3,2).
    Input ranks_desc must be descending (may include duplicates - we remove them).
    """
    uniq = sorted(set(ranks_desc), reverse=True)
    if len(uniq) < 5:
        return None

    # Normal straight: e.g., 9-8-7-6-5
    if uniq[0] - uniq[4] == 4 and len(uniq) == 5:
        return uniq[0]

    # Wheel straight: A-5-4-3-2 -> treat as 5-high
    # uniq would be [14,5,4,3,2]
    if uniq == [14, 5, 4, 3, 2]:
        return 5

    return None

def evaluate_5card(cards: List[Card]) -> HandScore:
    """
    Evaluate exactly 5 cards and return a comparable score.
    Higher score tuple means better hand.
    """
    if len(cards) != 5:
        raise ValueError("evaluate_5card expects exactly 5 cards")

    ranks_desc = _ranks(cards)
    flush = _is_flush(cards)
    straight_high = _straight_high(ranks_desc)

    counts = Counter(ranks_desc)  # rank -> frequency
    # Sort by (frequency desc, rank desc)
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    freqs = sorted(counts.values(), reverse=True)

    # Straight flush
    if flush and straight_high is not None:
        return (8, (straight_high,))

    # Four of a kind: [4,1]
    if freqs == [4, 1]:
        quad_rank = groups[0][0]
        kicker = groups[1][0]
        return (7, (quad_rank, kicker))

    # Full house: [3,2]
    if freqs == [3, 2]:
        trips_rank = groups[0][0]
        pair_rank = groups[1][0]
        return (6, (trips_rank, pair_rank))

    # Flush
    if flush:
        # Kickers are just sorted ranks
        return (5, tuple(ranks_desc))

    # Straight
    if straight_high is not None:
        return (4, (straight_high,))

    # Three of a kind: [3,1,1]
    if freqs == [3, 1, 1]:
        trips_rank = groups[0][0]
        kickers = sorted([r for r, c in groups[1:]], reverse=True)
        return (3, (trips_rank, *kickers))

    # Two pair: [2,2,1]
    if freqs == [2, 2, 1]:
        pair1 = groups[0][0]
        pair2 = groups[1][0]
        kicker = groups[2][0]
        hi, lo = max(pair1, pair2), min(pair1, pair2)
        return (2, (hi, lo, kicker))

    # One pair: [2,1,1,1]
    if freqs == [2, 1, 1, 1]:
        pair = groups[0][0]
        kickers = sorted([r for r, c in groups[1:]], reverse=True)
        return (1, (pair, *kickers))

    # High card: [1,1,1,1,1]
    return (0, tuple(ranks_desc))


# ----------------------------
# Best-of evaluation (optional extension)
# ----------------------------

def evaluate_best_of(cards: List[Card]) -> HandScore:
    """
    Evaluate best 5-card hand from N cards (N >= 5).
    Useful if later you add turn/river and have 7 cards total.
    """
    if len(cards) < 5:
        raise ValueError("Need at least 5 cards")
    best: Optional[HandScore] = None
    for combo in combinations(cards, 5):
        score = evaluate_5card(list(combo))
        if best is None or score > best:
            best = score
    assert best is not None
    return best


# ----------------------------
# Showdown compare helpers
# ----------------------------

def compare_showdown_hands(
    hole0: List[Card],
    hole1: List[Card],
    board: List[Card],
    allow_tie: bool = True,
) -> Optional[int]:
    """
    Compare two players at showdown.

    In your simplified game (2 hole + 3 board), each player has exactly 5 cards total:
      score0 = evaluate_5card(hole0 + board)
      score1 = evaluate_5card(hole1 + board)

    Returns:
      - 0 if player0 wins
      - 1 if player1 wins
      - None if tie (only if allow_tie=True)
      If allow_tie=False, ties are broken deterministically in favor of player0.
    """
    if len(hole0) != 2 or len(hole1) != 2:
        raise ValueError("Expect exactly 2 hole cards per player")
    if len(board) != 3:
        raise ValueError("Expect exactly 3 board cards (flop-only showdown)")

    score0 = evaluate_5card(hole0 + board)
    score1 = evaluate_5card(hole1 + board)

    if score0 > score1:
        return 0
    if score1 > score0:
        return 1
    return None if allow_tie else 0


# ----------------------------
# Simple self-test (optional)
# ----------------------------

if __name__ == "__main__":
    # This self-test only runs if Card/Suit/Rank are importable.
    try:
        from env.poker_env import Suit, Rank  # type: ignore
        # Example: straight flush vs trips
        sf = [
            Card(Suit.HEARTS, Rank.NINE),
            Card(Suit.HEARTS, Rank.EIGHT),
            Card(Suit.HEARTS, Rank.SEVEN),
            Card(Suit.HEARTS, Rank.SIX),
            Card(Suit.HEARTS, Rank.FIVE),
        ]
        trips = [
            Card(Suit.CLUBS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.ACE),
            Card(Suit.SPADES, Rank.ACE),
            Card(Suit.HEARTS, Rank.TWO),
            Card(Suit.CLUBS, Rank.THREE),
        ]
        print("SF score:", evaluate_5card(sf))
        print("Trips score:", evaluate_5card(trips))
        assert evaluate_5card(sf) > evaluate_5card(trips)
        print("Self-test OK.")
    except Exception as e:
        print("Self-test skipped (Card/Suit/Rank not importable):", e)
