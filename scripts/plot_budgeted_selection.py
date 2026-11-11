#!/usr/bin/env python3
"""Figures for the budgeted hint selection benchmark (examples/06_budgeted_selection.cpp).

Reads the CSV written by example_06_budgeted_selection and produces four figures (PNG and PDF) plus two
summary tables (CSV), so that every plotted value is also available as a number.

    fig1_quality_vs_budget   required block size and log-volume gain versus the budget k, per pool and strategy
    fig2_certified_gap       greedy block size with the certified range of the optimum
    fig3_selection_cost      gain evaluations and selection time, greedy versus lazy greedy
    fig4_optimality_ratio    f(greedy) / f(exact) on tiny pools against the 1 - 1/e guarantee

Usage:
    python3 -m venv .venv && .venv/bin/pip install -r scripts/requirements.txt
    .venv/bin/python scripts/plot_budgeted_selection.py [results.csv] [--outdir data/figures]

Only the standard library and matplotlib are required.
"""

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- Visual parameters ------------------------------------------------------------------------------------
# Categorical colours are assigned to strategies in a fixed order and never reused for another strategy.
# The set was checked for colour-vision-deficiency separation; marker shape and line style repeat the identity
# so the figures also read in greyscale print.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

STRATEGY_STYLE = {
    "greedy": dict(color="#2a78d6", marker="o", linestyle="-", label="Greedy"),
    "static": dict(color="#eb6834", marker="s", linestyle="-", label="Static top-k"),
    "random": dict(color="#1baf7a", marker="^", linestyle="-", label="Random"),
    "lookahead2": dict(color="#eda100", marker="D", linestyle=(0, (4, 2)), label="Lookahead-2"),
    "lazy": dict(color="#e87ba4", marker="s", linestyle="-", label="Lazy greedy"),
}
POOL_TITLES = {
    "A": "Pool A: one hint per coordinate",
    "B": "Pool B: repeated measurements",
    "C": "Pool C: overlapping sums",
    "D": "Pool D: dense combinations",
}
# Short titles for the four-panel figures, which are sized for the report's text width.
POOL_SHORT = {
    "A": "A: per coordinate",
    "B": "B: repeated",
    "C": "C: overlapping sums",
    "D": "D: dense",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.titlecolor": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 4.5,
})


def line_kwargs(strategy):
    style = STRATEGY_STYLE[strategy]
    return dict(color=style["color"], marker=style["marker"], linestyle=style["linestyle"], label=style["label"],
                markeredgecolor=SURFACE, markeredgewidth=1.2, solid_capstyle="round")


# --- Data -------------------------------------------------------------------------------------------------
def load(path):
    rows = []
    with open(path, newline="") as handle:
        for raw in csv.DictReader(handle):
            rows.append({
                "experiment": raw["experiment"],
                "pool": raw["pool"],
                "N": int(raw["N"]),
                "k": int(raw["k"]),
                "strategy": raw["strategy"],
                "trial": int(raw["trial"]),
                "f": float(raw["f"]),
                "U_best": float(raw["U_best"]),
                "beta": float(raw["beta_cont"]),
                "beta_before": float(raw["beta_before"]),
                "beta_opt_lower": float(raw["beta_opt_lower_cont"]),
                "evaluations": float(raw["evaluations"]),
                "ms": float(raw["ms"]),
            })
    if not rows:
        raise SystemExit(f"no data rows in {path}")
    return rows


def group_mean(rows, keys, field):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row[field])
    return {key: statistics.fmean(values) for key, values in buckets.items()}


def group_sd(rows, keys, field):
    buckets = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in keys)].append(row[field])
    return {key: (statistics.pstdev(values) if len(values) > 1 else 0.0) for key, values in buckets.items()}


def save(fig, outdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# --- Figure 1: quality versus budget ----------------------------------------------------------------------
def fig_quality(rows, outdir):
    e1 = [r for r in rows if r["experiment"] == "E1"]
    if not e1:
        return
    pools = sorted({r["pool"] for r in e1})
    budgets = sorted({r["k"] for r in e1})
    beta = group_mean(e1, ("pool", "k", "strategy"), "beta")
    gain = group_mean(e1, ("pool", "k", "strategy"), "f")
    order = ["random", "static", "greedy", "lookahead2"]

    fig, axes = plt.subplots(2, len(pools), figsize=(7.0, 4.6), sharex=True, sharey="row", squeeze=False)
    for col, pool in enumerate(pools):
        for row, (table, ylabel) in enumerate(((beta, "Required block size β"),
                                               (gain, "Gain f(S) (nats)"))):
            ax = axes[row][col]
            for strategy in order:
                xs = [k for k in budgets if (pool, k, strategy) in table]
                if xs:
                    ax.plot(xs, [table[(pool, k, strategy)] for k in xs], **line_kwargs(strategy),
                            zorder=3 if strategy == "lookahead2" else 2)
            if row == 0:
                ax.set_title(POOL_SHORT.get(pool, f"Pool {pool}"), loc="left")
            else:
                ax.set_xlabel("Budget k")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xticks([k for k in budgets if k != 5])
    # The y-axis is shared along each row, so set its range once from all panels of that row.
    for row, table in enumerate((beta, gain)):
        axes[row][0].set_ylim(0, 1.06 * max(table.values()))
    handles, labels = axes[0][0].get_legend_handles_labels()
    # Legend order follows the reading order of the story: adaptive rules first.
    wanted = ["Greedy", "Lookahead-2", "Static top-k", "Random"]
    picked = [(handles[labels.index(name)], name) for name in wanted if name in labels]
    fig.legend([h for h, _ in picked], [n for _, n in picked], loc="upper center", ncol=len(picked),
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, outdir, "fig1_quality_vs_budget")


# --- Figure 2: certified gap ------------------------------------------------------------------------------
def fig_certificate(rows, outdir):
    greedy = [r for r in rows if r["experiment"] == "E1" and r["strategy"] == "greedy"]
    if not greedy:
        return
    pools = sorted({r["pool"] for r in greedy})
    budgets = sorted({r["k"] for r in greedy})
    beta = group_mean(greedy, ("pool", "k"), "beta")
    lower = group_mean(greedy, ("pool", "k"), "beta_opt_lower")
    color = STRATEGY_STYLE["greedy"]["color"]

    fig, axes = plt.subplots(1, len(pools), figsize=(7.0, 2.6), sharey=True, squeeze=False)
    for col, pool in enumerate(pools):
        ax = axes[0][col]
        xs = [k for k in budgets if (pool, k) in beta]
        top = [beta[(pool, k)] for k in xs]
        bottom = [lower[(pool, k)] for k in xs]
        ax.fill_between(xs, bottom, top, color=color, alpha=0.22, linewidth=0,
                        label="Certified range of the optimum")
        ax.plot(xs, bottom, color=color, linewidth=1.0, alpha=0.7)
        kwargs = line_kwargs("greedy")
        kwargs["label"] = "Greedy selection"
        ax.plot(xs, top, **kwargs)
        widest = max(range(len(xs)), key=lambda i: top[i] - bottom[i])
        gap = top[widest] - bottom[widest]
        note = "gap 0 at every k" if gap < 0.005 else f"max gap {gap:.2f}\nat k = {xs[widest]}"
        ax.text(0.97, 0.95, note,
                transform=ax.transAxes, ha="right", va="top", color=INK_SECONDARY, fontsize=7.5)
        ax.set_title(POOL_SHORT.get(pool, f"Pool {pool}"), loc="left")
        ax.set_xlabel("Budget k")
        ax.set_xticks([k for k in xs if k != 5])
        ax.set_ylim(bottom=0)
        if col == 0:
            ax.set_ylabel("Required block size β")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, outdir, "fig2_certified_gap")


# --- Figure 3: selection cost -----------------------------------------------------------------------------
def fig_cost(rows, outdir):
    e4 = [r for r in rows if r["experiment"] == "E4"]
    if not e4:
        return
    pools = sorted({r["pool"] for r in e4})
    sizes = sorted({r["N"] for r in e4})
    evals = group_mean(e4, ("pool", "N", "strategy"), "evaluations")
    times = group_mean(e4, ("pool", "N", "strategy"), "ms")
    budget = e4[0]["k"]

    fig, axes = plt.subplots(2, len(pools), figsize=(7.0, 5.0), sharex=True, squeeze=False)
    for col, pool in enumerate(pools):
        for row, (table, ylabel, fmt) in enumerate(((evals, "Marginal-gain evaluations", "{:,.0f}"),
                                                    (times, "Selection time (ms)", "{:.2f}"))):
            ax = axes[row][col]
            for strategy in ("greedy", "lazy"):
                xs = [n for n in sizes if (pool, n, strategy) in table]
                ys = [table[(pool, n, strategy)] for n in xs]
                ax.plot(xs, ys, **line_kwargs(strategy))
                # Direct label on the last point only.
                ax.annotate(fmt.format(ys[-1]), (xs[-1], ys[-1]), textcoords="offset points", xytext=(-6, 7),
                            ha="right", color=INK_SECONDARY, fontsize=8.5)
            ax.set_xscale("log")
            ax.set_xticks(sizes)
            ax.set_xticklabels([f"{n:,}" for n in sizes])
            ax.minorticks_off()
            top = max(table[(pool, n, s)] for n in sizes for s in ("greedy", "lazy") if (pool, n, s) in table)
            if row == 0:
                ax.set_yscale("log")
                ax.set_ylim(top=top * 2.6)
                ax.set_title(f"{POOL_TITLES.get(pool, 'Pool ' + pool)} (k = {budget})", loc="left")
            else:
                ax.set_ylim(0, top * 1.18)
                ax.set_xlabel("Pool size N (candidate hints)")
            if col == 0:
                ax.set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, outdir, "fig3_selection_cost")


# --- Figure 4: optimality ratio ---------------------------------------------------------------------------
def fig_ratio(rows, outdir):
    e2 = [r for r in rows if r["experiment"] == "E2"]
    if not e2:
        return
    by_trial = defaultdict(dict)
    for r in e2:
        by_trial[r["trial"]][r["strategy"]] = r["f"]
    ratios = [v["greedy"] / v["exact"] for v in by_trial.values()
              if "greedy" in v and "exact" in v and v["exact"] > 0.0]
    if not ratios:
        return
    floor = 1.0 - math.exp(-1.0)
    optimal = sum(1 for x in ratios if x > 1.0 - 1e-9)
    pool_size, budget = e2[0]["N"], e2[0]["k"]

    # Bins of width 0.01 for the sub-optimal pools; pools where greedy is exactly optimal get their own bar at 1,
    # so that they are not merged with the [0.99, 1.00) bin.
    width = 0.01
    lo = 0.60
    bins = int(round((1.0 - lo) / width))
    counts = [0] * (bins + 1)
    for x in ratios:
        if x > 1.0 - 1e-9:
            counts[bins] += 1
        else:
            counts[min(bins - 1, max(0, int((x - lo) / width)))] += 1
    centers = [lo + (i + 0.5) * width for i in range(bins)] + [1.0 + width]

    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.bar(centers, counts, width=width * 0.8, color=STRATEGY_STYLE["greedy"]["color"], zorder=2)
    for center, count in zip(centers, counts):
        if count:
            ax.annotate(str(count), (center, count), textcoords="offset points", xytext=(0, 3), ha="center",
                        color=INK_SECONDARY, fontsize=8.5)
    ax.axvline(floor, color=INK_SECONDARY, linewidth=1.2, zorder=1)
    ax.annotate("worst-case guarantee\n1 − 1/e ≈ 0.632", (floor, max(counts) * 0.72), textcoords="offset points",
                xytext=(6, 0), ha="left", va="center", color=INK_SECONDARY, fontsize=8.5)
    ax.text(0.5, 0.52,
            f"greedy optimal in {optimal} of {len(ratios)} pools\nworst ratio {min(ratios):.4f}",
            transform=ax.transAxes, ha="center", va="center", color=INK, fontsize=9)
    ax.set_xlim(lo, 1.0 + 2.5 * width)
    ax.set_ylim(0, max(counts) * 1.12)
    ax.set_xlabel("f(greedy) / f(optimal); the right-most bar counts the pools where greedy is exactly optimal")
    ax.set_ylabel("Number of pools")
    ax.set_title(f"Greedy against brute force (N = {pool_size}, k = {budget}, {len(ratios)} pools)", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, outdir, "fig4_optimality_ratio")


# --- Summary tables ---------------------------------------------------------------------------------------
def write_tables(rows, outdir):
    e1 = [r for r in rows if r["experiment"] == "E1"]
    if e1:
        keys = ("pool", "k", "strategy")
        mean_f, sd_f = group_mean(e1, keys, "f"), group_sd(e1, keys, "f")
        mean_b, sd_b = group_mean(e1, keys, "beta"), group_sd(e1, keys, "beta")
        mean_lo = group_mean(e1, keys, "beta_opt_lower")
        with open(os.path.join(outdir, "summary_quality.csv"), "w", newline="") as handle:
            out = csv.writer(handle)
            out.writerow(["pool", "k", "strategy", "mean_gain_nats", "sd_gain_nats", "mean_beta", "sd_beta",
                          "mean_certified_beta_lower_bound"])
            for key in sorted(mean_f):
                bound = f"{mean_lo[key]:.4f}" if key[2] == "greedy" else ""
                out.writerow([*key, f"{mean_f[key]:.4f}", f"{sd_f[key]:.4f}", f"{mean_b[key]:.4f}",
                              f"{sd_b[key]:.4f}", bound])
        print("  wrote summary_quality.csv")
    e4 = [r for r in rows if r["experiment"] == "E4"]
    if e4:
        keys = ("pool", "N", "strategy")
        evals, times = group_mean(e4, keys, "evaluations"), group_mean(e4, keys, "ms")
        with open(os.path.join(outdir, "summary_cost.csv"), "w", newline="") as handle:
            out = csv.writer(handle)
            out.writerow(["pool", "N", "strategy", "mean_gain_evaluations", "mean_selection_ms"])
            for key in sorted(evals):
                out.writerow([*key, f"{evals[key]:.1f}", f"{times[key]:.4f}"])
        print("  wrote summary_cost.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default="data/budgeted_selection_results.csv")
    parser.add_argument("--outdir", default="data/figures")
    args = parser.parse_args()

    rows = load(args.csv)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"{len(rows)} rows from {args.csv}; writing to {args.outdir}")
    fig_quality(rows, args.outdir)
    fig_certificate(rows, args.outdir)
    fig_cost(rows, args.outdir)
    fig_ratio(rows, args.outdir)
    write_tables(rows, args.outdir)


if __name__ == "__main__":
    main()
