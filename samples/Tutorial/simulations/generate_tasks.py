#!/usr/bin/env python3
"""
generate_tasks.py — Synthetic IoT task-set generator (spec Section 6, teacher feedback).

Each task now carries a **quantum-suitability score** Q_s in [0,1] and a workload
measured in **MFLOP** (million FLOP). Some task types are quantum-optimizable
(high Q_s → benefit from a QPU fog), others are GPU-oriented (low Q_s → best on a
GPU), and a third group is general control/sensor work. This matches the
"some tasks are quantum-optimizable and some are not" framing in teacher_text.md
and Quantom.md Phase 4.

CSV schema (consumed by src/IoTDevice.cc):

    taskId, ownerDeviceId, arrivalTime, taskType, dataSizeKb,
    workload, quantumSuitability, deadline, priority, memoryMb, energyLocalEst

The local-execution energy uses a dynamic-power model (energy = power x time),
consistent with the heterogeneous fog energy model in ProcessorModel.h:

    E_local = P_local * T_local        with   T_local = workload / P_LOCAL_THROUGHPUT

See PROCESSOR_MODEL.md for the full derivation and the per-processor formulas.
"""

import argparse
import csv
import collections
import numpy as np

SEED = 42

NUM_DEVICES = 15
SIM_WINDOW_S = 95.0          # spread arrivals over ~95 s (sim-time-limit = 100 s)

# --- IoT local-execution model (must match omnetpp.ini) -----------------------
# E_local = P_local * T_local,  T_local = workload / P_LOCAL_THROUGHPUT.
# (Dynamic-power model: energy = power x execution time, consistent with the
# heterogeneous fog energy model in ProcessorModel.h / FogServer.cc.)
P_LOCAL_THROUGHPUT = 300.0   # MFLOP/s (slow IoT CPU)
P_LOCAL_POWER = 2.0          # Watts (active)
P_LOCAL_TX  = 1.5            # 5G uplink transmit power (W)
R_UPLINK    = 100e6          # 5G uplink data rate (bps) -> used for tx energy

# --- Task-type profiles -------------------------------------------------------
#   Qs       : quantum suitability score in [0,1]
#   W        : workload range in MFLOP
#   slack_ms : deadline slack range (ms) added to arrival time
# Quantum-optimizable tasks (high Q_s) -> QPU candidate (Quantom.md Table 1)
# GPU-oriented tasks (low Q_s)         -> GPU candidate  (Quantom.md Table 2)
TASK_TYPES = {
    # --- quantum-optimizable (high Q_s) ---
    "tsp":              {"Qs": 0.97, "W": (10, 60),  "slack_ms": (10, 40)},
    "knapsack":         {"Qs": 0.93, "W": (8, 50),   "slack_ms": (10, 40)},
    "maxcut":           {"Qs": 0.95, "W": (10, 60),  "slack_ms": (10, 40)},
    "portfolio":        {"Qs": 0.90, "W": (8, 45),   "slack_ms": (10, 40)},
    "molecular_sim":    {"Qs": 0.99, "W": (5, 30),   "slack_ms": (15, 50)},
    # --- GPU-oriented (low Q_s) ---
    "image_processing": {"Qs": 0.05, "W": (20, 80),  "slack_ms": (8, 30)},
    "cnn_inference":    {"Qs": 0.08, "W": (30, 80),  "slack_ms": (8, 30)},
    "video_processing": {"Qs": 0.02, "W": (25, 80),  "slack_ms": (10, 35)},
    # --- general control / sensor (low-medium Q_s) ---
    "sensor_data":      {"Qs": 0.15, "W": (2, 15),   "slack_ms": (5, 25)},
    "control_signal":   {"Qs": 0.10, "W": (1, 8),    "slack_ms": (4, 15)},
    "aggregation":      {"Qs": 0.35, "W": (5, 25),   "slack_ms": (8, 30)},
    "alert":            {"Qs": 0.10, "W": (1, 6),    "slack_ms": (4, 12)},
}
TYPE_NAMES = list(TASK_TYPES.keys())
# Mix: ~40% quantum-optimizable, ~30% GPU-oriented, ~30% general
TYPE_WEIGHTS = [0.10, 0.08, 0.08, 0.07, 0.07,
                0.12, 0.10, 0.08,
                0.10, 0.08, 0.07, 0.05]

PRIORITY_LEVELS = [1, 2, 3, 4, 5]
PRIORITY_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]


def local_energy_joule(workload_mflop: float) -> float:
    """Local execution energy: E_local = P_local * T_local, T_local = W / P_local."""
    t_local = workload_mflop / P_LOCAL_THROUGHPUT
    return P_LOCAL_POWER * t_local


def generate(num_tasks: int) -> list:
    rng = np.random.default_rng(SEED)

    # Poisson arrival process: exponential inter-arrival times, then rescaled
    # so the last task lands near the end of the simulation window.
    inter = rng.exponential(scale=SIM_WINDOW_S / num_tasks, size=num_tasks)
    arrivals = np.cumsum(inter)
    arrivals *= SIM_WINDOW_S / arrivals[-1]

    rows = []
    for i in range(num_tasks):
        ttype = rng.choice(TYPE_NAMES, p=TYPE_WEIGHTS)
        prof = TASK_TYPES[ttype]

        arrival = float(arrivals[i])
        workload = float(rng.uniform(*prof["W"]))                 # MFLOP
        qs = float(prof["Qs"])
        # small per-task jitter on Qs so identical types aren't identical
        qs = float(np.clip(qs + rng.normal(0, 0.02), 0.0, 1.0))

        slack = float(rng.uniform(*prof["slack_ms"])) / 1000.0     # -> seconds
        deadline = arrival + slack

        data_size_kb = float(rng.uniform(10.0, 500.0))
        priority = int(rng.choice(PRIORITY_LEVELS, p=PRIORITY_WEIGHTS))
        memory_mb = float(rng.uniform(50.0, 512.0))
        energy_local = local_energy_joule(workload)

        rows.append({
            "taskId": i + 1,
            "ownerDeviceId": int(rng.integers(0, NUM_DEVICES)),
            "arrivalTime": round(arrival, 4),
            "taskType": ttype,
            "dataSizeKb": round(data_size_kb, 2),
            "workload": round(workload, 4),
            "quantumSuitability": round(qs, 4),
            "deadline": round(deadline, 4),
            "priority": priority,
            "memoryMb": round(memory_mb, 1),
            "energyLocalEst": round(energy_local, 6),
        })
    return rows


def write_csv(rows: list, path: str) -> None:
    fields = ["taskId", "ownerDeviceId", "arrivalTime", "taskType", "dataSizeKb",
              "workload", "quantumSuitability", "deadline", "priority",
              "memoryMb", "energyLocalEst"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list) -> None:
    print(f"\nGenerated {len(rows)} tasks -> summary statistics")
    print("-" * 68)
    for col in ["arrivalTime", "workload", "quantumSuitability", "deadline",
                "dataSizeKb", "memoryMb", "energyLocalEst"]:
        vals = np.array([r[col] for r in rows], dtype=float)
        print(f"{col:<22} mean={vals.mean():9.4f}  std={vals.std():9.4f}"
              f"  min={vals.min():9.4f}  max={vals.max():9.4f}")

    types = collections.Counter(r["taskType"] for r in rows)
    print("\nTask type distribution (Q_s = quantum suitability):")
    for t in TYPE_NAMES:
        print(f"  {t:<18} Qs={TASK_TYPES[t]['Qs']:.2f}  {types[t]:>5}"
              f"  ({types[t]/len(rows)*100:4.1f}%)")

    quantum = sum(types[t] for t in TYPE_NAMES if TASK_TYPES[t]["Qs"] >= 0.7)
    print(f"\nQuantum-optimizable tasks (Q_s>=0.7): {quantum} "
          f"({quantum/len(rows)*100:.1f}%)")

    devices = collections.Counter(r["ownerDeviceId"] for r in rows)
    print("\nTasks per device (min/max):"
          f" {min(devices.values())} / {max(devices.values())}")


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic IoT tasks.")
    ap.add_argument("-n", "--num", type=int, default=2000,
                    help="number of tasks (default 2000)")
    ap.add_argument("-o", "--out", default="tasks.csv",
                    help="output CSV path (default tasks.csv)")
    args = ap.parse_args()

    rows = generate(args.num)
    write_csv(rows, args.out)
    print_summary(rows)
    print(f"\nWrote {len(rows)} tasks to {args.out}")


if __name__ == "__main__":
    main()
