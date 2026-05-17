"""Asymmetric behavioral intel for higher-k ToM in Chicken.

For each non-self-play match, the higher-ranked agent (by the order
RB ≤ WSLS < 0-TOM < 1-TOM < 2-TOM) gets a warm-started prior on the
opponent's stationary P(Straight), measured from a baseline tournament.

Phases:
  A. Baseline tournament (no intel) — measures per-matchup empirical P(Straight).
  B. Intel tournament — higher-k agent's `p_op_mean` (log-odds) and the nested
     0-ToM's `p_op_mean0` are set to logit(opponent's empirical P(Straight)),
     with tight variance.
  C. Comparison: table + grouped bar chart.

Run:
    /Users/yoav.levy/chicken/.venv-tomsup/bin/python chicken_tomsup/intel.py
"""
from __future__ import annotations

import os
import warnings

os.chdir("/tmp")  # avoid the cwd-shadows-installed-tomsup issue
warnings.filterwarnings("ignore")  # tomsup k-ToM emits divide-by-zero in some priors

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomsup as ts

OUT_DIR = "/Users/yoav.levy/chicken"

# (name, "order rank" for who-gets-intel, constructor)
AGENT_DEFS = [
    ("RB",    0, lambda: ts.RB(bias=0.5)),
    ("WSLS",  0, lambda: ts.WSLS()),
    ("0-TOM", 1, lambda: ts.TOM(level=0)),
    ("1-TOM", 2, lambda: ts.TOM(level=1)),
    ("2-TOM", 3, lambda: ts.TOM(level=2)),
]
AGENT_RANK = {name: rank for name, rank, _ in AGENT_DEFS}
AGENT_CTOR = {name: ctor for name, _, ctor in AGENT_DEFS}
AGENT_NAMES = [d[0] for d in AGENT_DEFS]

N_ROUNDS = 50
N_SIM = 100


def chicken_matrix(crash: float = -10.0) -> ts.PayoffMatrix:
    """Correct Chicken matrix. 0 = Swerve, 1 = Straight."""
    matrix = np.array(
        [
            [(0.0, -1.0), (1.0, crash)],
            [(0.0, 1.0), (-1.0, crash)],
        ],
        dtype=float,
    )
    return ts.PayoffMatrix(name="custom", predefined=matrix)


def per_agent_ranking(results: pd.DataFrame, agent_names: list[str]) -> pd.Series:
    out = {}
    for name in agent_names:
        as_a0 = results[results["agent0"] == name]["payoff_agent0"]
        as_a1 = results[results["agent1"] == name]["payoff_agent1"]
        out[name] = pd.concat([as_a0, as_a1]).mean()
    return pd.Series(out)


def logit(p: float) -> float:
    p = max(min(p, 1 - 1e-3), 1e-3)
    return float(np.log(p / (1 - p)))


def warm_start_tom(level: int, opp_p_straight: float, tight_var: float = -3.0) -> ts.TOM:
    """Build a fresh k-TOM whose prior beliefs about the opponent's P(action=1)
    are set to logit(opp_p_straight), with tightened variance.

    For 0-TOM: scalar `p_op_mean0` and `p_op_var0` on own_states.
    For k≥1:
      - `own_states.p_op_mean[:]`  — set every level's mean to logit(...)
      - the nested 0-ToM's `own_states.p_op_mean0` and `p_op_var0`
    """
    agent = ts.TOM(level=level)
    init = agent.get_internal_states()
    own = init["own_states"]
    val = logit(opp_p_straight)
    if level == 0:
        own["p_op_mean0"] = float(val)
        own["p_op_var0"] = float(tight_var)
    else:
        # Cast to float so we can store non-integer values (default dtype is int64).
        own["p_op_mean"] = np.full(own["p_op_mean"].shape, val, dtype=float)
        # Also seed the nested 0-ToM's prior on opp's choice.
        nested0_own = init["opponent_states"][0]["own_states"]
        nested0_own["p_op_mean0"] = float(val)
        nested0_own["p_op_var0"] = float(tight_var)
    agent.set_internal_states(init)
    return agent


def play_pair(
    matrix: ts.PayoffMatrix,
    name_a: str,
    name_b: str,
    intel_freq_a: float | None,
    intel_freq_b: float | None,
    n_rounds: int,
    n_sim: int,
) -> pd.DataFrame:
    """Run n_sim independent simulations of (A vs B), building fresh agents per sim."""
    rows = []
    for sim in range(n_sim):
        a_obj = warm_start_tom(_tom_level(name_a), intel_freq_a) if intel_freq_a is not None else AGENT_CTOR[name_a]()
        b_obj = warm_start_tom(_tom_level(name_b), intel_freq_b) if intel_freq_b is not None else AGENT_CTOR[name_b]()
        r = ts.compete(
            a_obj, b_obj, p_matrix=matrix, n_rounds=n_rounds,
            reset_agent=False, verbose=False, return_val="df",
        )
        r["sim"] = sim
        r["agent0"] = name_a
        r["agent1"] = name_b
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def _tom_level(name: str) -> int:
    return int(name.split("-")[0])


def all_pairs() -> list[tuple[str, str]]:
    """All ordered pairs (a, b) with no self-play."""
    pairs = []
    for i, a in enumerate(AGENT_NAMES):
        for b in AGENT_NAMES[i + 1 :]:
            pairs.append((a, b))
    return pairs


def phase_a_baseline(matrix: ts.PayoffMatrix) -> tuple[pd.DataFrame, dict]:
    """Run all pairs without intel; collect per-matchup empirical P(Straight)."""
    dfs = []
    freqs: dict[tuple[str, str], float] = {}
    for a, b in all_pairs():
        df = play_pair(matrix, a, b, None, None, N_ROUNDS, N_SIM)
        dfs.append(df)
        freqs[(a, b)] = float(df["choice_agent0"].mean())  # P(a plays Straight)
        freqs[(b, a)] = float(df["choice_agent1"].mean())  # P(b plays Straight)
    return pd.concat(dfs, ignore_index=True), freqs


def phase_b_intel(matrix: ts.PayoffMatrix, freqs: dict) -> pd.DataFrame:
    """Higher-ranked agent in each pair gets intel about opp's baseline freq."""
    dfs = []
    for a, b in all_pairs():
        ra, rb = AGENT_RANK[a], AGENT_RANK[b]
        intel_a = intel_b = None
        if ra > rb and "TOM" in a:
            intel_a = freqs[(b, a)]  # opp b's freq of Straight against a
        elif rb > ra and "TOM" in b:
            intel_b = freqs[(a, b)]
        # Ties (RB vs WSLS): no intel.
        df = play_pair(matrix, a, b, intel_a, intel_b, N_ROUNDS, N_SIM)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def plot_comparison(baseline: pd.Series, intel: pd.Series, path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(AGENT_NAMES))
    w = 0.35
    ax.bar(x - w / 2, [baseline[n] for n in AGENT_NAMES], w, label="Baseline (no intel)")
    ax.bar(x + w / 2, [intel[n] for n in AGENT_NAMES], w, label="With intel (asym, higher-k)")
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_NAMES)
    ax.set_ylabel("Mean payoff per round")
    ax.set_title("Chicken: behavioral intel for higher-order k-ToM\n"
                 "(higher-k agent in each match gets opponent's true P(Straight))")
    ax.axhline(0, color="black", lw=0.5, alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def main() -> None:
    matrix = chicken_matrix(crash=-10.0)

    print("=" * 72)
    print("Phase A: baseline tournament (no intel)")
    print("=" * 72)
    baseline_df, freqs = phase_a_baseline(matrix)
    baseline_rank = per_agent_ranking(baseline_df, AGENT_NAMES)
    for n, s in baseline_rank.sort_values(ascending=False).items():
        print(f"  {n:<6s} {s:+.3f}")

    print("\n  Empirical P(Straight) per (focal vs opponent):")
    for a, b in all_pairs():
        print(f"  {a:<6s} vs {b:<6s}: a={freqs[(a, b)]:.3f}  b={freqs[(b, a)]:.3f}")

    print()
    print("=" * 72)
    print("Phase B: asymmetric intel — higher-ranked agent gets opp's freq")
    print("=" * 72)
    intel_df = phase_b_intel(matrix, freqs)
    intel_rank = per_agent_ranking(intel_df, AGENT_NAMES)
    for n, s in intel_rank.sort_values(ascending=False).items():
        print(f"  {n:<6s} {s:+.3f}")

    print()
    print("=" * 72)
    print("Comparison: baseline vs intel")
    print("=" * 72)
    print(f"  {'agent':<6s} {'baseline':>10s} {'intel':>10s} {'delta':>10s}")
    for n in AGENT_NAMES:
        b, i = baseline_rank[n], intel_rank[n]
        marker = " <-- helped" if i > b + 0.05 else (" <-- hurt" if i < b - 0.05 else "")
        print(f"  {n:<6s} {b:>+10.3f} {i:>+10.3f} {(i - b):>+10.3f}{marker}")

    print("\n  k-TOM order baseline: "
          + " > ".join(baseline_rank.loc[["0-TOM", "1-TOM", "2-TOM"]]
                       .sort_values(ascending=False).index.tolist()))
    print("  k-TOM order intel:    "
          + " > ".join(intel_rank.loc[["0-TOM", "1-TOM", "2-TOM"]]
                       .sort_values(ascending=False).index.tolist()))

    plot_comparison(baseline_rank, intel_rank,
                    os.path.join(OUT_DIR, "tomsup_intel_compare.png"))

    # Sanity check: verify warm-start actually shifted the prior.
    print("\n--- Sanity: 2-TOM prior on opp's P(action=1) ---")
    print("  default 2-TOM, own_states.p_op_mean:",
          ts.TOM(level=2).get_internal_states()["own_states"]["p_op_mean"])
    print("  warm-started (opp P(Straight)=0.8), own_states.p_op_mean:",
          warm_start_tom(2, 0.8).get_internal_states()["own_states"]["p_op_mean"])


if __name__ == "__main__":
    main()
