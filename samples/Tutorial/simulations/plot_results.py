#!/usr/bin/env python3
"""
plot_results.py — Read OMNeT++ scalar/vector results for each offloading policy
and produce comparison charts (spec Section 10, teacher feedback):

  * Deadline hit ratio, average end-to-end delay, average energy per task
  * Mean fog-server utilization + per-fog utilization over time (vector)
  * Local vs offloaded share
  * Multi-task-count comparison (100/200/500/1000 tasks)

Usage:
    python3 plot_results.py                 # results/ dir, current run
    python3 plot_results.py --results results
    python3 plot_results.py --multi         # scan results/n*/ subdirs
"""

import os
import re
import shlex
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POLICIES = ["Local", "Random", "RoundRobin", "Greedy", "DQN"]
COLORS = ["#8c8c8c", "#4c9be8", "#f2a541", "#5cb85c", "#d9534f"]
FOG_NAMES = ["IBM(QPU)", "IonQ(QPU)", "RTX3090(GPU)", "RTX3090b(GPU)", "CPU-11core"]
FOG_COLORS = ["#9b59b6", "#c39bd3", "#e67e22", "#f39c12", "#3498db"]


def parse_sca(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.startswith("scalar"):
                continue
            parts = shlex.split(line.strip())
            if len(parts) < 4:
                continue
            _, module, name, value = parts[0], parts[1], parts[2], parts[3]
            try:
                out[(module, name)] = float(value)
            except ValueError:
                pass
    return out


def parse_vec(path):
    """Return {vector_id: (module, name, [(t, v), ...])} from an OMNeT++ .vec file.

    OMNeT++ 6 .vec format: a `vector <module> <name> <vectorId> ...` header line
    per vector, then data lines `<vectorId> <eventNum> <simtime> <value>`.
    """
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            if line.startswith("vector"):
                parts = shlex.split(line.strip())
                # vector <vectorId> <module> <name> <columns>
                if len(parts) >= 4:
                    vid, module, name = parts[1], parts[2], parts[3]
                    out[vid] = (module, name, [])
            else:
                parts = line.split()
                if len(parts) >= 4 and parts[0] in out:
                    try:
                        t = float(parts[-2])
                        v = float(parts[-1])
                    except ValueError:
                        continue
                    out[parts[0]][2].append((t, v))
    return out


def agg(scalars, name, how="mean", node="iot"):
    vals = [v for (mod, n), v in scalars.items() if n == name and f".{node}[" in mod]
    if not vals:
        return np.nan
    if how == "mean":
        return float(np.mean(vals))
    if how == "sum":
        return float(np.sum(vals))
    if how == "max":
        return float(np.max(vals))
    return np.nan


def collect(results_dir):
    rows = {}
    for pol in POLICIES:
        sc = parse_sca(os.path.join(results_dir, f"{pol}-#0.sca"))
        if not sc:
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


def save_single(labels, values, colors, title, ylabel, fname, plots_dir, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar(ax, labels, values, colors, title, ylabel, fmt)
    fig.tight_layout()
    p = os.path.join(plots_dir, fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")


def plot_fog_utilization_over_time(results_dir, plots_dir, policy="DQN"):
    """Per-fog utilization over time from the .vec file (teacher: 'utilization
    rate of each fog at every moment')."""
    vec = parse_vec(os.path.join(results_dir, f"{policy}-#0.vec"))
    series = {}
    for vid, (module, name, pts) in vec.items():
        if "fogUtilization" not in name:
            continue
        m = re.search(r"\.fog\[(\d+)\]", module)
        if not m:
            continue
        idx = int(m.group(1))
        if pts:
            ts = np.array([p[0] for p in pts])
            vs = np.array([p[1] for p in pts])
            series[idx] = (ts, vs)

    if not series:
        print(f"  ! no fogUtilization vectors in {policy}-#0.vec")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx in sorted(series):
        ts, vs = series[idx]
        label = FOG_NAMES[idx] if idx < len(FOG_NAMES) else f"fog[{idx}]"
        ax.plot(ts, vs * 100.0, label=label, color=FOG_COLORS[idx % len(FOG_COLORS)],
                linewidth=1.4, alpha=0.9)
    ax.set_title(f"Per-Fog Utilization Over Time ({policy})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Simulation time (s)")
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, 105)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    p = os.path.join(plots_dir, f"fog_utilization_over_time_{policy}.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")

    # Also produce a per-fog mean-utilization bar chart across all policies
    return series


def plot_per_fog_util_bar(results_dir, plots_dir, labels):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(FOG_NAMES))
    width = 0.15
    for i, pol in enumerate(labels):
        sc = parse_sca(os.path.join(results_dir, f"{pol}-#0.sca"))
        utils = []
        for idx in range(len(FOG_NAMES)):
            vals = [v for (mod, n), v in sc.items()
                    if n == "utilization" and f".fog[{idx}]" in mod]
            utils.append(np.mean(vals) * 100 if vals else 0)
        ax.bar(x + (i - len(labels) / 2) * width, utils, width, label=pol,
               color=COLORS[POLICIES.index(pol)], edgecolor="black", linewidth=0.4)
    ax.set_title("Per-Fog Utilization by Policy", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean utilization (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(FOG_NAMES, rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(plots_dir, "per_fog_utilization.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")


def plot_main(results_dir):
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    rows = collect(results_dir)
    if not rows:
        print("No results found. Run the simulation configs first (see RUNBOOK.md).")
        return

    labels = [p for p in POLICIES if p in rows]
    colors = [COLORS[POLICIES.index(p)] for p in labels]
    hit = [rows[p]["hit"] for p in labels]
    delay = [rows[p]["delay"] * 1000 for p in labels]
    energy = [rows[p]["energy"] for p in labels]
    util = [rows[p]["util"] for p in labels]
    local_pct = [rows[p]["local_pct"] for p in labels]
    offload_pct = [rows[p]["offload_pct"] for p in labels]

    save_single(labels, hit, colors, "Deadline Hit Ratio by Policy", "Hit ratio (%)",
                "hit_ratio.png", plots_dir, "{:.1f}")
    save_single(labels, delay, colors, "Average End-to-End Delay by Policy", "Avg delay (ms)",
                "avg_delay.png", plots_dir, "{:.1f}")
    save_single(labels, energy, colors, "Average Energy per Task by Policy", "Energy (J)",
                "energy.png", plots_dir, "{:.3f}")
    save_single(labels, util, colors, "Mean Fog-Server Utilization by Policy",
                "Utilization (%)", "fog_utilization.png", plots_dir, "{:.1f}")

    # local vs offload (stacked)
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
    p = os.path.join(plots_dir, "local_vs_offload.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")

    # per-fog utilization over time (DQN run)
    plot_fog_utilization_over_time(results_dir, plots_dir, policy="DQN")
    plot_fog_utilization_over_time(results_dir, plots_dir, policy="Greedy")
    plot_per_fog_util_bar(results_dir, plots_dir, labels)

    # combined dashboard
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("IoT-Fog Task Offloading: Policy Comparison (heterogeneous CPU/GPU/QPU fog)",
                 fontsize=14, fontweight="bold")
    bar(axes[0, 0], labels, hit, colors, "Deadline Hit Ratio", "%", "{:.1f}")
    bar(axes[0, 1], labels, delay, colors, "Avg End-to-End Delay", "ms", "{:.1f}")
    bar(axes[0, 2], labels, energy, colors, "Avg Energy per Task", "J", "{:.3f}")
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
    p = os.path.join(plots_dir, "summary_dashboard.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")

    # console table
    print("\n" + "=" * 80)
    print(f"{'Policy':<12}{'Hit%':>8}{'Delay(ms)':>12}{'Energy(J)':>12}"
          f"{'FogUtil%':>10}{'Local%':>9}")
    print("-" * 80)
    for p_ in labels:
        r = rows[p_]
        print(f"{p_:<12}{r['hit']:>8.1f}{r['delay']*1000:>12.2f}{r['energy']:>12.3f}"
              f"{r['util']:>10.1f}{r['local_pct']:>9.1f}")
    print("=" * 80)
    print(f"\nAll charts written to: {plots_dir}/")


def plot_multi(results_root):
    """Compare metrics across task counts (100/200/500/1000) for each policy."""
    subdirs = sorted(glob.glob(os.path.join(results_root, "n*")),
                     key=lambda d: int(re.search(r"n(\d+)", d).group(1)))
    if not subdirs:
        print(f"No n*/ subdirs found in {results_root}. Run run_experiments.sh first.")
        return

    plots_dir = os.path.join(results_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ns, by_pol = [], {p: {"hit": [], "delay": [], "energy": []} for p in POLICIES}
    for d in subdirs:
        n = int(re.search(r"n(\d+)", d).group(1))
        ns.append(n)
        rows = collect(d)
        for p in POLICIES:
            if p in rows:
                by_pol[p]["hit"].append(rows[p]["hit"])
                by_pol[p]["delay"].append(rows[p]["delay"] * 1000)
                by_pol[p]["energy"].append(rows[p]["energy"])
            else:
                by_pol[p]["hit"].append(np.nan)
                by_pol[p]["delay"].append(np.nan)
                by_pol[p]["energy"].append(np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Deadline Hit Ratio vs Task Count", "Avg Delay vs Task Count",
              "Avg Energy per Task vs Task Count"]
    ylabels = ["Hit ratio (%)", "Avg delay (ms)", "Energy (J)"]
    keys = ["hit", "delay", "energy"]
    for ax, title, yl, key in zip(axes, titles, ylabels, keys):
        for p in POLICIES:
            ax.plot(ns, by_pol[p][key], marker="o", label=p,
                    color=COLORS[POLICIES.index(p)], linewidth=1.8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of tasks")
        ax.set_ylabel(yl)
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
    fig.suptitle("Policy Comparison Across Task Counts (heterogeneous CPU/GPU/QPU fog)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(plots_dir, "multi_task_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  saved {p}")

    print("\n" + "=" * 70)
    print(f"{'Tasks':<8}" + "".join(f"{p:>12}" for p in POLICIES) + "  (hit%)")
    print("-" * 70)
    for i, n in enumerate(ns):
        print(f"{n:<8}" + "".join(f"{by_pol[p]['hit'][i]:>12.1f}" for p in POLICIES))
    print("=" * 70)
    print(f"\nMulti-task chart written to: {plots_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="results directory")
    ap.add_argument("--multi", action="store_true",
                    help="scan results/n*/ subdirs for multi-task-count comparison")
    args = ap.parse_args()

    if args.multi:
        plot_multi(args.results)
    else:
        plot_main(args.results)


if __name__ == "__main__":
    main()
