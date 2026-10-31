"""Module 6, exercise 6 — n-step returns, between Monte Carlo and TD."""

import pytest


@pytest.mark.skip(reason="Course Module 6, exercise 6: implement "
                         "n_step_targets(results, values, n) in "
                         "selfplay/tictactoe_lab.py, wire an n_step parameter "
                         "through selfplay_game and run, then delete this skip line")
def test_n_step_targets_span_td_to_monte_carlo():
    """Contract: selfplay/tictactoe_lab.py gains

        def n_step_targets(results, values, n) -> list of floats

    for one game of T positions, where for each position t (in play order)
        results[t] = the final game result from the point of view of the
                     side to move at t (+1, 0 or -1)
        values[t]  = the table's current value of position t, from the side
                     to move at t
    and returns the value target for each position:
        if t + n >= T:  results[t]                  (the game ends within n moves)
        else:           (-1)**n * values[t + n]     (bootstrap from n moves ahead;
                                                     the side to move alternates)
    Rewards are 0 until the end, so nothing else is added. n = 1 is the TD
    target that --td already uses; n >= T is the Monte Carlo target.

    Then give selfplay_game(..., n_step=None) and run(..., n_step=None) a
    parameter that uses it (None keeps Monte Carlo), and add --n-step.
    """
    from selfplay.tictactoe_lab import n_step_targets, run

    results = [1.0, -1.0, 1.0, -1.0]          # X won; X moved at t = 0 and 2
    values = [0.1, 0.2, 0.3, 0.4]

    assert n_step_targets(results, values, 1) == pytest.approx([-0.2, -0.3, -0.4, -1.0])
    assert n_step_targets(results, values, 2) == pytest.approx([0.3, 0.4, 1.0, -1.0])
    assert n_step_targets(results, values, 10) == pytest.approx(results)

    # It still learns with a bootstrapped target in the loop
    rows = run(iterations=5, games_per_iter=60, n_step=3, seed=0, verbose=False)
    assert rows[-1]['value_mse'] < rows[0]['value_mse']
    assert rows[-1]['vs_perfect'] > rows[0]['vs_perfect']
