"""Sanity checks for the Chicken ranking result.

Each check is independent. Run with:
    /Users/yoav.levy/chicken/.venv-tomsup/bin/python chicken_tomsup/sanity.py
"""
from __future__ import annotations

import os

os.chdir("/tmp")

import numpy as np
import pandas as pd
import tomsup as ts


def chicken_matrix(crash: float = -10.0) -> ts.PayoffMatrix:
    """Correct Chicken (NOT tomsup's broken built-in!).

    0 = Swerve, 1 = Straight. Crash sits on (Straight, Straight) = (1, 1).
    """
    matrix = np.array(
        [
            [(0.0, -1.0), (1.0, crash)],
            [(0.0, 1.0), (-1.0, crash)],
        ],
        dtype=float,
    )
    return ts.PayoffMatrix(name="custom", predefined=matrix)


def per_agent_ranking(results: pd.DataFrame, agent_names: list[str]) -> pd.Series:
    """Aggregate mean payoff per round for each agent across all its matchups."""
    out = {}
    for name in agent_names:
        as_a0 = results[results["agent0"] == name]["payoff_agent0"]
        as_a1 = results[results["agent1"] == name]["payoff_agent1"]
        all_payoffs = pd.concat([as_a0, as_a1])
        out[name] = all_payoffs.mean()
    return pd.Series(out).sort_values(ascending=False)


def run(name: str, p_matrix: ts.PayoffMatrix, agents: list[str],
        start_params: list[dict], n_rounds: int, n_sim: int) -> pd.Series:
    print(f"\n=== {name} ===")
    group = ts.create_agents(agents, start_params)
    group.set_env("round_robin")
    res = group.compete(
        p_matrix=p_matrix, n_rounds=n_rounds, n_sim=n_sim, verbose=False,
    )
    ranking = per_agent_ranking(res, agents)
    print("Ranking by mean payoff/round (higher = better):")
    for rank, (a, score) in enumerate(ranking.items(), 1):
        print(f"  {rank}. {a:<8s} {score:+.3f}")
    return ranking, res


# ----------------------------------------------------------------------- (a)
def check_a_zero_sum_validation():
    """If our setup is right, in a zero-sum game like Matching Pennies the
    k-ToM ranking should be 2-TOM > 1-TOM > 0-TOM (paper's classic result).
    """
    pennies = ts.PayoffMatrix(name="penny_competitive")
    agents = ["RB", "0-TOM", "1-TOM", "2-TOM"]
    start = [{"bias": 0.5}, {}, {}, {}]
    ranking, _ = run(
        "(a) Matching Pennies (zero-sum control)",
        pennies, agents, start, n_rounds=50, n_sim=100,
    )
    tom_order = [a for a in ranking.index if "TOM" in a]
    expected = ["2-TOM", "1-TOM", "0-TOM"]
    print(f"  k-ToM order observed: {tom_order}")
    print(f"  k-ToM order expected: {expected}")
    print(f"  PASS" if tom_order == expected else "  FAIL — setup is wrong!")


# ----------------------------------------------------------------------- (b)
def check_b_robustness():
    """Re-run Chicken with 4× more sims and report bootstrap-style std."""
    matrix = chicken_matrix(crash=-10.0)
    agents = ["RB", "WSLS", "0-TOM", "1-TOM", "2-TOM"]
    start = [{"bias": 0.5}, {}, {}, {}, {}]

    rankings = []
    for seed_offset in range(4):
        np.random.seed(1000 + seed_offset)
        group = ts.create_agents(agents, start)
        group.set_env("round_robin")
        res = group.compete(
            p_matrix=matrix, n_rounds=50, n_sim=100, verbose=False,
        )
        rankings.append(per_agent_ranking(res, agents))

    df = pd.DataFrame(rankings)
    print("\n=== (b) Chicken (crash=-10), 4 independent runs of 100 sims each ===")
    print("Agent   | mean    std     min     max     run-by-run order stable?")
    print("-" * 70)
    for a in agents:
        col = df[a]
        print(f"  {a:<6s}| {col.mean():+.3f}  {col.std():.3f}  "
              f"{col.min():+.3f}  {col.max():+.3f}")
    # Check ordering across runs
    orderings = [tuple(r.sort_values(ascending=False).index) for r in rankings]
    unique = set(orderings)
    print(f"\nDistinct rank orderings across 4 runs: {len(unique)}")
    for o in unique:
        print(f"  {o}")


# ----------------------------------------------------------------------- (c)
def check_c_action_lockin():
    """Did 0-TOM-vs-2-TOM actually converge to (Straight, Swerve)?

    Action 0 = Swerve, 1 = Straight.
    """
    matrix = chicken_matrix(crash=-10.0)
    agents = ["0-TOM", "2-TOM"]
    group = ts.create_agents(agents, [{}, {}])
    group.set_env("round_robin")
    res = group.compete(p_matrix=matrix, n_rounds=50, n_sim=200, verbose=False)
    p_straight_a0 = res["choice_agent0"].mean()
    p_straight_a1 = res["choice_agent1"].mean()
    print("\n=== (c) Action lock-in: 0-TOM vs 2-TOM (200 sims) ===")
    print(f"  0-TOM P(Straight): {p_straight_a0:.3f}")
    print(f"  2-TOM P(Straight): {p_straight_a1:.3f}")
    late = res[res["round"] >= 30]
    print(f"  late rounds (30+):  0-TOM={late['choice_agent0'].mean():.3f}, "
          f"2-TOM={late['choice_agent1'].mean():.3f}")
    crash_rate = ((res["choice_agent0"] == 1) & (res["choice_agent1"] == 1)).mean()
    print(f"  crash rate (both Straight): {crash_rate:.4f}")


# ----------------------------------------------------------------------- (d)
def check_d_unbluffable():
    """RB with bias=1 always plays Straight (action 1). Every rational k-ToM
    should learn to Swerve and accept payoff -1 each round (instead of -10).
    """
    matrix = chicken_matrix(crash=-10.0)
    agents = ["RB", "0-TOM", "1-TOM", "2-TOM"]
    start = [{"bias": 1.0}, {}, {}, {}]  # bias=1 -> always plays 1 -> always Straight
    ranking, res = run(
        "(d) Unbluffable opponent: RB(bias=1) always plays Straight",
        matrix, agents, start, n_rounds=50, n_sim=50,
    )
    print("  Expected: RB wins big, all k-TOM lose around -1 (the swerve loser payoff)")

    # And the inverse: RB(bias=0) always Swerves; k-ToM should all go Straight.
    start = [{"bias": 0.0}, {}, {}, {}]
    ranking2, _ = run(
        "(d') Pushover opponent: RB(bias=0) always Swerves",
        matrix, agents, start, n_rounds=50, n_sim=50,
    )
    print("  Expected: RB loses, all k-TOM win around +1 (the straight winner payoff)")


if __name__ == "__main__":
    check_a_zero_sum_validation()
    check_b_robustness()
    check_c_action_lockin()
    check_d_unbluffable()
