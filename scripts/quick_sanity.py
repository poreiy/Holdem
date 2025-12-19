# scripts/quick_sanity.py
from __future__ import annotations

import random
from typing import Tuple

# Adjust import depending on your project layout
from env.poker_env import PokerEnv, Street, ActionType


def assert_invariants(env: PokerEnv) -> None:
    s = env.get_state()

    # Basic numeric sanity
    assert s.pot >= 0, f"Pot negative: {s.pot}"
    for i, p in enumerate(s.players):
        assert p.stack >= 0, f"Player {i} stack negative: {p.stack}"
        assert p.street_commit >= 0, f"Player {i} street_commit negative: {p.street_commit}"

    # Conservation (2 players only)
    total = s.pot + s.players[0].stack + s.players[1].stack
    expect = 2 * env.starting_stack
    assert total == expect, f"Chip conservation broken: total={total}, expect={expect}"

    # Terminal invariants
    if s.done:
        assert env.legal_actions() == [], "Done state should have no legal actions"
        assert s.winner in (0, 1, None), f"Winner invalid: {s.winner}"


def run_one_hand(env: PokerEnv, rng: random.Random) -> Tuple[int, int]:
    """
    Returns (steps, reward_sum) for player0
    """
    s = env.reset()
    assert_invariants(env)

    steps = 0
    reward_sum = 0

    flop_started = False
    flop_first_action_checked = False

    while True:
        s = env.get_state()
        if s.done:
            break

        # Detect flop start
        if s.street == Street.FLOP and not flop_started:
            flop_started = True
            flop_first_action_checked = False

        legal = env.legal_actions()
        assert len(legal) > 0, "Non-terminal state must have legal actions"

        # Choose random legal action
        a = rng.choice(legal)

        # Record if flop first action is CHECK (not CALL)
        if s.street == Street.FLOP and not flop_first_action_checked:
            # It's a CHECK if to_call==0
            to_call = s.current_bet - s.players[s.to_act].street_commit
            if a == ActionType.CHECK_CALL and to_call == 0:
                flop_first_action_checked = True

        # Step
        _, r, done, _info = env.step(a)
        steps += 1
        reward_sum += r

        assert_invariants(env)

        # Critical check:
        # If flop first action was CHECK, the hand must NOT end immediately on that step.
        if flop_first_action_checked:
            s2 = env.get_state()
            assert not s2.done, (
                "BUG: Hand ended immediately after the first CHECK on the FLOP. "
                "This means your street-ending logic is wrong (should require two checks)."
            )
            # only check this once: after verifying, reset flag
            flop_first_action_checked = False

        if done:
            break

        # prevent infinite loops
        assert steps < 200, "Too many steps in one hand -> possible loop bug"

    # At hand end, total reward sum should equal final p0 profit (since incremental rewards)
    s_end = env.get_state()
    p0_profit = s_end.players[0].stack - env.starting_stack
    assert reward_sum == p0_profit, f"Reward sum mismatch: sum={reward_sum}, profit={p0_profit}"

    return steps, reward_sum


def main():
    env = PokerEnv(seed=123, starting_stack=100, sb=1, bb=2, max_raises_per_street=4)
    rng = random.Random(999)

    hands = 1000
    total_steps = 0
    total_profit = 0

    for i in range(hands):
        steps, profit = run_one_hand(env, rng)
        total_steps += steps
        total_profit += profit

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{hands}] avg_steps={total_steps/(i+1):.2f}, total_profit_p0={total_profit}")

    print("Sanity check PASSED.")
    print(f"Hands={hands}, avg_steps={total_steps/hands:.2f}, total_profit_p0={total_profit}")


if __name__ == "__main__":
    main()
