"""Crash-penalty sweep in Chicken.

The intel experiment showed that behavioral intel doesn't help because the
crash penalty (-10) is so dominant that the optimal action threshold sits at
~10% P(opp Straight). For intel (or any modeling) to matter, the threshold
needs to be near commonly-observed frequencies.

This script varies the crash penalty and re-runs the 5-agent tournament at
each level. The threshold P(opp Straight) below which Straight is preferred
is `1/|crash|`, so:
    crash = -10  →  threshold = 0.10
    crash =  -5  →  threshold = 0.20
    crash =  -3  →  threshold = 0.33
    crash =  -2  →  threshold = 0.50
    crash = -1.5 →  threshold = 0.67
    crash = -1.1 →  threshold = 0.91  (close to never crash)

Question: does the k-TOM ranking (`0 > 1 > 2`) flip toward `2 > 1 > 0` at
lower crash penalties, where deeper opponent modeling matters more?

Run:
    /Users/yoav.levy/chicken/.venv-tomsup/bin/python chicken_tomsup/crash_sweep.py
"""
from __future__ import annotations

import os
import warnings

os.chdir("/tmp")
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomsup as ts

OUT_DIR = "/Users/yoav.levy/chicken"
CRASH_LEVELS = [-10.0, -5.0, -3.0, -2.0, -1.5, -1.1]
AGENT_NAMES = ["RB", "WSLS", "0-TOM", "1-TOM", "2-TOM"]
START_PARAMS = [{"bias": 0.5}, {}, {}, {}, {}]
N_ROUNDS = 50
N_SIM = 50


def chicken_matrix(crash: float) -> ts.PayoffMatrix:
    """0=Swerve, 1=Straight, crash sits on (Straight, Straight)."""
    m = np.array(
        [
            [(0.0, -1.0), (1.0, crash)],
            [(0.0, 1.0), (-1.0, crash)],
        ],
        dtype=float,
    )
    return ts.PayoffMatrix(name="custom", predefined=m)


def per_agent_ranking(results: pd.DataFrame, names: list[str]) -> pd.Series:
    out = {}
    for name in names:
        as_a0 = results[results["agent0"] == name]["payoff_agent0"]
        as_a1 = results[results["agent1"] == name]["payoff_agent1"]
        out[name] = pd.concat([as_a0, as_a1]).mean()
    return pd.Series(out)


def run_one_crash(crash: float) -> tuple[pd.Series, pd.DataFrame]:
    matrix = chicken_matrix(crash)
    group = ts.create_agents(AGENT_NAMES, START_PARAMS)
    group.set_env("round_robin")
    res = group.compete(p_matrix=matrix, n_rounds=N_ROUNDS, n_sim=N_SIM, verbose=False)
    ranking = per_agent_ranking(res, AGENT_NAMES)
    # Per-matchup empirical P(Straight) for diagnostic.
    pair_freqs = (
        res.groupby(["agent0", "agent1"])[["choice_agent0", "choice_agent1"]]
        .mean()
        .round(3)
    )
    return ranking, pair_freqs


def main() -> None:
    rankings = {}
    print(f"{'crash':>7s} | " + " ".join(f"{n:>8s}" for n in AGENT_NAMES) + " | k-TOM order")
    print("-" * 80)
    diagnostic_freqs = {}
    for crash in CRASH_LEVELS:
        ranking, pair_freqs = run_one_crash(crash)
        rankings[crash] = ranking
        diagnostic_freqs[crash] = pair_freqs
        ktom_order = " > ".join(
            ranking.loc[["0-TOM", "1-TOM", "2-TOM"]]
            .sort_values(ascending=False)
            .index.tolist()
        )
        print(
            f"{crash:>+7.1f} | "
            + " ".join(f"{ranking[n]:>+8.3f}" for n in AGENT_NAMES)
            + f" | {ktom_order}"
        )

    # Plot: payoff vs crash level, one line per agent.
    fig, ax = plt.subplots(figsize=(8, 5))
    crash_arr = np.array(CRASH_LEVELS)
    for name in AGENT_NAMES:
        ys = np.array([rankings[c][name] for c in CRASH_LEVELS])
        marker = "o" if "TOM" in name else "x"
        ax.plot(crash_arr, ys, marker=marker, label=name)
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.set_xlabel("Crash penalty (more negative = more dangerous)")
    ax.set_ylabel("Mean payoff per round")
    ax.set_title("Chicken: per-agent payoff vs crash penalty\n"
                 "(at low crash, the strategic threshold sits near typical frequencies)")
    ax.invert_xaxis()  # most punitive on the left
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "tomsup_crash_sweep.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {out}")

    # Diagnostic: P(Straight) per matchup at the extreme crash levels.
    for crash in [CRASH_LEVELS[0], CRASH_LEVELS[-1]]:
        print(f"\nEmpirical P(Straight) per matchup at crash={crash}:")
        print(diagnostic_freqs[crash].to_string())


if __name__ == "__main__":
    main()
