#!/usr/bin/env python3
"""Plotting script for paper validation and end-to-end BKZ benchmarks.

Generates:
1. data/figures/fig_paper_validation.pdf (.png) - 3-panel plot replicating Figure 5 of Dachman-Soled et al. (CRYPTO 2020)
2. data/figures/fig_e2e_bkz_recovery.pdf (.png) - 2-panel plot showing wall-clock BKZ reduction time and tours vs hint count.
"""

import csv
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 8,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
})

def plot_paper_validation(decay_csv, out_pdf, out_png):
    data = {"Perfect": ([], [], []), "Modular": ([], [], []), "Approximate": ([], [], [])}
    
    with open(decay_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ht = row["hint_type"]
            if ht == "Baseline":
                for k in data:
                    data[k][0].append(0)
                    data[k][1].append(float(row["beta_gsa"]))
                    data[k][2].append(float(row["beta_prob"]))
            elif ht in data:
                data[ht][0].append(int(row["count"]))
                data[ht][1].append(float(row["beta_gsa"]))
                data[ht][2].append(float(row["beta_prob"]))

    # Sized for the text width of the report (about 6.5 in), so that 8 pt text prints at 8 pt.
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)

    configs = [
        ("Perfect", axes[0], "Number of perfect hints", "Perfect hints"),
        ("Modular", axes[1], "Number of modular hints", "Modular hints (mod 11)"),
        ("Approximate", axes[2], "Number of approximate hints", r"Approximate hints ($\sigma_\epsilon^2 = 3$)"),
    ]

    for ht, ax, xlabel, subtitle in configs:
        counts, gsa, prob = data[ht]
        # Sort by count
        pts = sorted(zip(counts, gsa, prob), key=lambda x: x[0])
        x = [p[0] for p in pts]
        y_gsa = [p[1] for p in pts]
        y_prob = [p[2] for p in pts]

        ax.grid(True, linestyle="-", alpha=0.7)
        ax.plot(x, y_gsa, marker="o", color="#2a78d6", linewidth=1.5, markersize=3.5, label="GSA-Intersect")
        ax.plot(x, y_prob, marker="^", color="#eb6834", linewidth=1.5, markersize=3.5, linestyle="--",
                label="Probabilistic simulation")

        ax.set_title(subtitle, fontsize=8.5, pad=4, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=8, color=INK)
        ax.set_ylim(0, 50)

    axes[0].set_ylabel(r"Required block size $\beta$", fontsize=8, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {out_pdf} and {out_png}")

def plot_e2e_bkz_recovery(e2e_csv, out_pdf, out_png):
    hints = []
    times_ms = []
    tours = []
    succ_rates = []

    with open(e2e_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hints.append(int(row["k"]))
            times_ms.append(float(row["mean_time_ms"]))
            tours.append(float(row["mean_tours"]))
            succ_rates.append(float(row["success_rate"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

    # Panel 1: Wall-Clock Runtime
    # Panel 1: wall-clock time (log scale). Direct labels on the two end points only, placed beside the marker.
    ax1.grid(True, linestyle="-", alpha=0.7)
    ax1.plot(hints, times_ms, marker="o", color="#2a78d6", linewidth=1.5, markersize=3.5)
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of perfect coordinate hints $k$", fontsize=8, color=INK)
    ax1.set_ylabel("Mean BKZ-4 time (ms, log scale)", fontsize=8, color=INK)
    ax1.set_title("Reduction time (dimension 25 to 13)", fontsize=8.5, pad=4, color=INK, loc="left")
    ax1.set_xticks(hints)
    if len(times_ms) > 0:
        ax1.annotate(f"{times_ms[0]:.0f} ms", (hints[0], times_ms[0]), textcoords="offset points",
                     xytext=(7, 3), fontsize=7.5, color=INK_SECONDARY, va="bottom")
        ax1.annotate(f"{times_ms[-1]:.1f} ms", (hints[-1], times_ms[-1]), textcoords="offset points",
                     xytext=(-6, 0), fontsize=7.5, color=INK_SECONDARY, ha="right", va="center")
    ax1.set_ylim(times_ms[-1] / 2.0, times_ms[0] * 2.0)

    # Panel 2: BKZ tours
    ax2.grid(True, linestyle="-", alpha=0.7)
    ax2.plot(hints, tours, marker="s", color="#eb6834", linewidth=1.5, markersize=3.5)
    ax2.set_xlabel("Number of perfect coordinate hints $k$", fontsize=8, color=INK)
    ax2.set_ylabel("Mean BKZ-4 tours", fontsize=8, color=INK)
    ax2.set_title("Tours to convergence", fontsize=8.5, pad=4, color=INK, loc="left")
    ax2.set_xticks(hints)
    ax2.set_ylim(0, 18)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote {out_pdf} and {out_png}")

if __name__ == "__main__":
    os.makedirs("data/figures", exist_ok=True)
    os.makedirs("report/figures", exist_ok=True)

    decay_csv = "data/paper_validation_decay.csv"
    e2e_csv = "data/e2e_bkz_recovery.csv"

    fig_val_pdf = "data/figures/fig_paper_validation.pdf"
    fig_val_png = "data/figures/fig_paper_validation.png"
    fig_e2e_pdf = "data/figures/fig_e2e_bkz_recovery.pdf"
    fig_e2e_png = "data/figures/fig_e2e_bkz_recovery.png"

    plot_paper_validation(decay_csv, fig_val_pdf, fig_val_png)
    plot_e2e_bkz_recovery(e2e_csv, fig_e2e_pdf, fig_e2e_png)

    # Copy to report/figures
    shutil.copy(fig_val_pdf, "report/figures/fig_paper_validation.pdf")
    shutil.copy(fig_val_png, "report/figures/fig_paper_validation.png")
    shutil.copy(fig_e2e_pdf, "report/figures/fig_e2e_bkz_recovery.pdf")
    shutil.copy(fig_e2e_png, "report/figures/fig_e2e_bkz_recovery.png")
    print("[OK] Copied figures to report/figures/")

