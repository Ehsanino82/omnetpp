#!/usr/bin/env python3
"""
plot_results.py — Read OMNeT++ scalar results for each offloading policy and
produce comparison charts (spec Section 10) suitable for a report/presentation.

Usage:
    python3 plot_results.py            # reads results/<Policy>-#0.sca
Output:
    results/plots/*.png  and a combined  results/plots/summary_dashboard.png
"""

import os
import re
import shlex
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Policies to compare, in display order (skip the duplicate [General] run)
POLICIES = ["Local", "Random", "RoundRobin", "Greedy", "DQN"]
COLORS = ["#8c8c8c", "#4c9be8", "#f2a541", "#5cb85c", "#d9534f"]

SCALAR_RE = re.compile(r'^scalar\s+(.*)$')


def parse_sca(path):
    """Return dict: {(module, name): value} from an OMNeT++ .sca file."""
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.startswith("scalar"):
                continue
            # scalar <module> <name> <value>   (module/name may be quoted)
            parts = shlex.split(line.strip())
            if len(parts) < 4:
                continue
            _, module, name, value = parts[0], parts[1], parts[2], parts[3]
            try:
                out[(module, name)] = float(value)
            except ValueError:
                pass
    return out


def agg(scalars, name, how="mean", node="iot"):
    """Aggregate a scalar across all matching modules (iot[*] or fog[*])."""
    vals = [v for (mod, n), v in scalars.items()
            if n == name and f".{node}[" in mod]
    if not vals:
        return np.nan
    if how == "mean":
        return float(np.mean(vals))
    if how == "sum":
        return float(np.sum(vals))
    if how == "max":
        return float(np.max(vals))
    return np.nan


def collect():
    rows = {}
    for pol in POLICIES:
        sc = parse_sca(os.path.join(RESULTS_DIR, f"{pol}-#0.sca"))
        if not sc:
            print(f"  ! no results for {pol} (run: ../src/Tutorial -u Cmdenv -c {pol})")
            continue
        local = agg(sc, "tasksLocal", "sum")
        off = agg(sc, "tasksOffloaded", "sum")
        total = (local or 0) + (off or 0)
        rows[pol] = {
            "hit": agg(sc, "hitRatio", "mean") * 100.0,
            "delay": agg(sc, "e2eDelay:mean", "mean"),
            "delay_max": agg(sc, "e2eDelay:max", "max"),
            "energy": agg(sc, "energy:mean", "mean"),
            "util": agg(sc, "utilization", "mean", node="fog") * 100.0,
            "backlog": agg(sc, "queueBacklog:mean", "mean", node="fog"),
            "local_pct": (local / total * 100.0) if total else 0.0,
            "offload_pct": (off / total * 100.0) if total else 0.0,
        }
    return rows


def bar(ax, labels, values, colors, title, ylabel, fmt="{:.2f}"):
    bars = ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.tick_params(axis="x", rotation=15)
    for b, v in zip(bars, values):
        if not np.isnan(v):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=9)
    return bars


def save_single(labels, values, colors, title, ylabel, fname, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar(ax, labels, values, colors, title, ylabel, fmt)
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    rows = collect()
    if not rows:
        print("No results found. Run the simulation configs first (see RUNBOOK.md Step 5).")
        return

    labels = [p for p in POLICIES if p in rows]
    colors = [COLORS[POLICIES.index(p)] for p in labels]
    hit = [rows[p]["hit"] for p in labels]
    delay = [rows[p]["delay"] * 1000 for p in labels]        # -> ms
    energy = [rows[p]["energy"] for p in labels]
    util = [rows[p]["util"] for p in labels]
    local_pct = [rows[p]["local_pct"] for p in labels]
    offload_pct = [rows[p]["offload_pct"] for p in labels]

    # ---- individual charts ----
    save_single(labels, hit, colors,
                "Deadline Hit Ratio by Policy", "Hit ratio (%)",
                "hit_ratio.png", fmt="{:.1f}")
    save_single(labels, delay, colors,
                "Average End-to-End Delay by Policy", "Avg delay (ms)",
                "avg_delay.png", fmt="{:.1f}")
    save_single(labels, energy, colors,
                "Average Energy per Task by Policy", "Energy (units)",
                "energy.png", fmt="{:.2f}")
    save_single(labels, util, colors,
                "Mean Fog-Server Utilization by Policy", "Utilization (%)",
                "fog_utilization.png", fmt="{:.1f}")

    # ---- local vs offload (stacked) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, local_pct, label="Local", color="#8c8c8c", edgecolor="black")
    ax.bar(labels, offload_pct, bottom=local_pct, label="Offloaded",
           color="#4c9be8", edgecolor="black")
    ax.set_title("Local vs Offloaded Task Share", fontsize=12, fontweight="bold")
    ax.set_ylabel("Share of tasks (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    fig.tight_layout()
    p = os.path.join(PLOTS_DIR, "local_vs_offload.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")

    # ---- combined dashboard ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("IoT–Fog Task Offloading: Policy Comparison (2000 tasks, 15 IoT / 5 Fog)",
                 fontsize=15, fontweight="bold")
    bar(axes[0, 0], labels, hit, colors, "Deadline Hit Ratio", "%", "{:.1f}")
    bar(axes[0, 1], labels, delay, colors, "Avg End-to-End Delay", "ms", "{:.1f}")
    bar(axes[0, 2], labels, energy, colors, "Avg Energy per Task", "units", "{:.2f}")
    bar(axes[1, 0], labels, util, colors, "Mean Fog Utilization", "%", "{:.1f}")
    bar(axes[1, 1], labels, [rows[p]["backlog"] * 1000 for p in labels], colors,
        "Mean Fog Queue Backlog", "ms", "{:.1f}")
    ax = axes[1, 2]
    ax.bar(labels, local_pct, label="Local", color="#8c8c8c", edgecolor="black")
    ax.bar(labels, offload_pct, bottom=local_pct, label="Offloaded",
           color="#4c9be8", edgecolor="black")
    ax.set_title("Local vs Offloaded", fontsize=12, fontweight="bold")
    ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.legend()
    for a in axes.flat:
        a.tick_params(axis="x", rotation=15)
        a.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(PLOTS_DIR, "summary_dashboard.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")

    # ---- console table ----
    print("\n" + "=" * 78)
    print(f"{'Policy':<12}{'Hit%':>8}{'Delay(ms)':>12}{'Energy':>10}"
          f"{'FogUtil%':>10}{'Local%':>9}")
    print("-" * 78)
    for p_ in labels:
        r = rows[p_]
        print(f"{p_:<12}{r['hit']:>8.1f}{r['delay']*1000:>12.2f}{r['energy']:>10.2f}"
              f"{r['util']:>10.1f}{r['local_pct']:>9.1f}")
    print("=" * 78)
    print(f"\nAll charts written to: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
