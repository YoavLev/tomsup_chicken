"""Chicken simulation using the tomsup package.

Mirrors the structure of the from-scratch implementation in `chicken/`, but
uses `tomsup`'s k-ToM agents (which follow Devaine et al. 2017's variational
Bayesian formulation rather than de Weerd 2013's belief-and-confidence model).

Run from a directory other than /Users/yoav.levy/chicken (the cloned repo
shadows the package import otherwise). The script chdir's to /tmp at the top
to handle this automatically.
"""
from __future__ import annotations

import os

# tomsup's package directory shares a name with the cloned-repo root, so cwd
# can hide the installed package. chdir away before importing.
os.chdir("/tmp")

import matplotlib.pyplot as plt
import numpy as np
import tomsup as ts

OUT_DIR = "/Users/yoav.levy/chicken"


def chicken_matrix(crash: float = -10.0, win: float = 1.0, tie: float = 0.0) -> ts.PayoffMatrix:
    """Custom Chicken with configurable crash penalty.

    CORRECT Chicken encoding (NOT tomsup's broken built-in!):
      0 = Swerve (the safe move)
      1 = Straight (the dangerous move; crash if both pick 1)

    matrix[agent, my_choice, opp_choice]:
      (Swerve,   Swerve)   = ( tie,  tie)         both back down
      (Swerve,   Straight) = (-win, +win)         I yield, you take
      (Straight, Swerve)   = (+win, -win)         I take, you yield
      (Straight, Straight) = (crash, crash)       BOTH STRAIGHT = CRASH

    tomsup's built-in `chicken` puts the crash on (0,0), which inverts the game
    (action 1 becomes strictly dominant — no longer a Chicken game).
    """
    matrix = np.array(
        [
            # Agent 0's payoffs, indexed [my_choice, opp_choice]
            [(tie, -win), (win, crash)],
            # Agent 1's payoffs, indexed [agent_0_choice, agent_1_choice]
            # so we have to think from the other side: agent_1 is the column
            [(tie, win), (-win, crash)],
        ],
        dtype=float,
    )
    return ts.PayoffMatrix(name="custom", predefined=matrix)


def run_tournament(crash: float, label: str, n_rounds: int = 50, n_sim: int = 50) -> None:
    matrix = chicken_matrix(crash=crash)
    print(f"\n=== Chicken {label} (crash={crash}) ===")
    print(matrix)

    # Five agents: three k-ToM levels plus two baselines.
    agent_names = ["RB", "WSLS", "0-TOM", "1-TOM", "2-TOM"]
    start_params = [
        {"bias": 0.5},  # RB: 50/50 random
        {},  # WSLS: defaults
        {},  # 0-TOM: defaults
        {},  # 1-TOM: defaults
        {"save_history": True},  # 2-TOM: keep internal states for inspection
    ]

    group = ts.create_agents(agent_names, start_params)
    group.set_env(env="round_robin")

    results = group.compete(
        p_matrix=matrix,
        n_rounds=n_rounds,
        n_sim=n_sim,
        save_history=True,
        verbose=False,
    )

    # Per-pair mean payoff for each agent.
    print("\nMean payoff per round, per matchup (focal agent's perspective):")
    summary = (
        results.groupby(["agent0", "agent1"])[["payoff_agent0", "payoff_agent1"]]
        .mean()
        .round(3)
    )
    print(summary.to_string())

    # Heatmap of total winnings (built-in tomsup plot).
    plt.figure(figsize=(6, 5))
    group.plot_heatmap(cmap="RdBu_r", show=False)
    plt.title(f"Chicken tournament — {label}")
    out = os.path.join(OUT_DIR, f"tomsup_chicken_heatmap_{label}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")

    # Action-frequency over time for the k-ToM matchups, so we can see who
    # is locked into Swerve and who's running roughshod.
    # Action 0 = Swerve, 1 = Straight in the corrected matrix.
    # plt.figure(figsize=(8, 4))
    # for pair in [("0-TOM", "1-TOM"), ("0-TOM", "2-TOM"), ("1-TOM", "2-TOM")]:
    #     sub = results[(results["agent0"] == pair[0]) & (results["agent1"] == pair[1])]
    #     per_round = sub.groupby("round")[["choice_agent0", "choice_agent1"]].mean()
    #     plt.plot(per_round.index, per_round["choice_agent0"], label=f"{pair[0]} (vs {pair[1]})")
    #     plt.plot(per_round.index, per_round["choice_agent1"], "--", label=f"{pair[1]} (vs {pair[0]})")
    # plt.xlabel("round")
    # plt.ylabel("P(Straight)")
    # plt.title(f"Action frequency by round — {label}")
    # plt.legend(fontsize=8, loc="best")
    # plt.ylim(-0.05, 1.05)
    # out = os.path.join(OUT_DIR, f"tomsup_action_freq_{label}.png")
    # plt.savefig(out, dpi=120, bbox_inches="tight")
    # plt.close()
    # print(f"  saved {out}")


def main() -> None:
    for crash, label in [(-10.0, "crash=-10"), (-4.0, "crash=-4")]:
        run_tournament(crash=crash, label=label)


if __name__ == "__main__":
    main()
