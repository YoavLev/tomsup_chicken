"""Toughness-bias experiments for k-ToM agents in Chicken.

Tests whether setting tomsup's `bias` parameter on TOM agents — which adds a
constant to the expected payoff of action 1 (= Straight in our encoding) —
can flip the k-ToM ranking from "lower order wins" to the paper's
"higher order wins" pattern.

Three experiments:
  (1) sweep uniform bias for all k-TOM, plot per-agent payoff vs bias
  (2) bias-vs-bias heatmap for 2-TOM head-to-head with 1-TOM
  (3) re-rank the full tournament at the best bias from (1)

Run from anywhere:
    /Users/yoav.levy/chicken/.venv-tomsup/bin/python chicken_tomsup/toughness.py
"""
from __future__ import annotations

import os

# tomsup's package directory shares a name with the cloned-repo root, so cwd
# can hide the installed package. chdir away before importing.
os.chdir("/tmp")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomsup as ts

OUT_DIR = "/Users/yoav.levy/chicken"
BIAS_LEVELS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


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
    """Mean payoff per round for each agent across all matchups it appears in."""
    out = {}
    for name in agent_names:
        as_a0 = results[results["agent0"] == name]["payoff_agent0"]
        as_a1 = results[results["agent1"] == name]["payoff_agent1"]
        out[name] = pd.concat([as_a0, as_a1]).mean()
    return pd.Series(out)


# ----------------------------------------------------------------- experiment 1
def experiment_1_bias_sweep(matrix: ts.PayoffMatrix, n_rounds: int, n_sim: int) -> pd.DataFrame:
    """For each bias level, run the 5-agent tournament and record rankings."""
    rows = []
    for bias in BIAS_LEVELS:
        agents = ["RB", "WSLS", "0-TOM", "1-TOM", "2-TOM"]
        start_params = [
            {"bias": 0.5},  # RB random 50/50
            {},
            {"bias": bias},
            {"bias": bias},
            {"bias": bias},
        ]
        group = ts.create_agents(agents, start_params)
        group.set_env("round_robin")
        res = group.compete(
            p_matrix=matrix, n_rounds=n_rounds, n_sim=n_sim, verbose=False,
        )
        ranking = per_agent_ranking(res, agents)
        for agent, score in ranking.items():
            rows.append({"bias": bias, "agent": agent, "payoff": score})
        print(
            f"  bias={bias:>4}: "
            + "  ".join(f"{a}={ranking[a]:+.2f}" for a in agents)
        )
    return pd.DataFrame(rows)


def plot_bias_sweep(df: pd.DataFrame, path: str) -> None:
    plt.figure(figsize=(7, 5))
    for agent in ["RB", "WSLS", "0-TOM", "1-TOM", "2-TOM"]:
        sub = df[df["agent"] == agent].sort_values("bias")
        marker = "o" if "TOM" in agent else "x"
        plt.plot(sub["bias"], sub["payoff"], marker=marker, label=agent)
    plt.axhline(0, color="black", lw=0.5, alpha=0.5)
    plt.xlabel("Toughness bias on all k-TOM agents")
    plt.ylabel("Mean payoff per round")
    plt.title("Chicken: per-agent payoff vs k-TOM bias")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# ----------------------------------------------------------------- experiment 2
def experiment_2_bias_arms_race(
    matrix: ts.PayoffMatrix, n_rounds: int, n_sim: int
) -> np.ndarray:
    """6x6 grid of (1-TOM bias, 2-TOM bias). Each cell is a head-to-head match.

    Returns the grid of mean 2-TOM payoff per round. NaN cells indicate
    numerical instability in tomsup's variational Bayes update at extreme bias.
    """
    grid = np.full((len(BIAS_LEVELS), len(BIAS_LEVELS)), np.nan, dtype=float)
    for i, b1 in enumerate(BIAS_LEVELS):
        for j, b2 in enumerate(BIAS_LEVELS):
            agents = ["1-TOM", "2-TOM"]
            start = [{"bias": b1}, {"bias": b2}]
            group = ts.create_agents(agents, start)
            group.set_env("round_robin")
            try:
                res = group.compete(
                    p_matrix=matrix, n_rounds=n_rounds, n_sim=n_sim, verbose=False,
                )
                grid[i, j] = res["payoff_agent1"].mean()
            except (ValueError, FloatingPointError) as e:
                # tomsup k-TOM internals can NaN out at extreme biases;
                # leave the cell as NaN.
                print(f"  [warn] cell (b1={b1}, b2={b2}) failed: {e}")
        cells = "  ".join(
            f"2T_b={b2}->{grid[i, j]:+.2f}" if not np.isnan(grid[i, j])
            else f"2T_b={b2}->NaN"
            for j, b2 in enumerate(BIAS_LEVELS)
        )
        print(f"  1-TOM bias={b1:>4}: {cells}")
    return grid


def plot_bias_arms_race(grid: np.ndarray, path: str) -> None:
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        print(f"  [warn] no finite values; skipping plot {path}")
        return
    vmax = max(abs(finite.min()), abs(finite.max()))
    plt.figure(figsize=(6, 5))
    masked = np.ma.masked_invalid(grid)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="lightgray")
    im = plt.imshow(
        masked,
        origin="lower",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )
    plt.xticks(range(len(BIAS_LEVELS)), [str(b) for b in BIAS_LEVELS])
    plt.yticks(range(len(BIAS_LEVELS)), [str(b) for b in BIAS_LEVELS])
    plt.xlabel("2-TOM bias")
    plt.ylabel("1-TOM bias")
    plt.title("2-TOM mean payoff vs 1-TOM (head-to-head)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            label = f"{grid[i, j]:+.2f}" if np.isfinite(grid[i, j]) else "NaN"
            plt.text(j, i, label, ha="center", va="center", color="black", fontsize=9)
    plt.colorbar(im, label="2-TOM mean payoff/round")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# ----------------------------------------------------------------- experiment 3
def experiment_3_rerank(
    matrix: ts.PayoffMatrix, df_sweep: pd.DataFrame, n_rounds: int, n_sim: int
) -> None:
    """Show the ranking at every bias level — does the k-TOM order ever flip?"""
    pivot = (
        df_sweep.pivot(index="agent", columns="bias", values="payoff")
        .reindex(["RB", "WSLS", "0-TOM", "1-TOM", "2-TOM"])
    )
    print("\n  Mean payoff/round by bias level (rows=agent, cols=bias on k-TOM):")
    print(pivot.round(3).to_string())

    # k-TOM ordering at each bias level.
    print("\n  k-TOM ordering at each bias (best -> worst):")
    for bias in df_sweep["bias"].unique():
        sub = df_sweep[df_sweep["bias"] == bias].set_index("agent")["payoff"]
        order = sub.loc[["0-TOM", "1-TOM", "2-TOM"]].sort_values(ascending=False)
        order_str = " > ".join(f"{a}({s:+.2f})" for a, s in order.items())
        print(f"    bias={bias:>4}:  {order_str}")

    # Theoretical-needed bias to break the inversion: with crash=-10,
    # win=+1, swerve-loss=-1, an agent facing a "certainly Straight" opponent
    # has E[Swerve]=-1, E[Straight]=-10. Bias would need to exceed 9 to make
    # Straight preferred even when opponent is known-tough.
    print("\n  Note: with crash=-10 and swerve-loss=-1, a bias > 9 is needed")
    print("  for Straight to beat Swerve when opponent is known to be tough.")
    print("  tomsup's k-TOM goes numerically unstable above ~5, so bias-as-")
    print("  toughness can't push k-TOM agents past the crash deterrent.")


# ---------------------------------------------------------------------- main
def main() -> None:
    matrix = chicken_matrix(crash=-10.0)

    print("=" * 72)
    print("Experiment 1: bias sweep across all k-TOM agents")
    print("=" * 72)
    df_sweep = experiment_1_bias_sweep(matrix, n_rounds=50, n_sim=50)
    plot_bias_sweep(df_sweep, os.path.join(OUT_DIR, "tomsup_bias_sweep.png"))

    print()
    print("=" * 72)
    print("Experiment 2: 2-TOM vs 1-TOM, bias arms race")
    print("=" * 72)
    grid = experiment_2_bias_arms_race(matrix, n_rounds=50, n_sim=100)
    plot_bias_arms_race(grid, os.path.join(OUT_DIR, "tomsup_bias_2vs1.png"))

    print()
    print("=" * 72)
    print("Experiment 3: re-rank at best bias")
    print("=" * 72)
    experiment_3_rerank(matrix, df_sweep, n_rounds=50, n_sim=50)


if __name__ == "__main__":
    main()
