#!/usr/bin/env python3
"""
generate_tasks.py — Synthetic IoT task-set generator (spec Section 6).

Generates 2000 IoT tasks for the OMNeT++ Fog-offloading simulation and writes
them to ``tasks.csv`` using a fixed seed (SEED = 42) for reproducibility.

CSV schema (consumed by src/IoTDevice.cc):

    taskId, ownerDeviceId, arrivalTime, taskType, dataSizeKb,
    cpuDemand, deadline, priority, memoryMb, energyLocalEst

Units are chosen to be directly usable by the discrete-event simulation:
  * arrivalTime : seconds (Poisson arrival process across the sim window)
  * cpuDemand   : seconds of required processing at fog speed (rate = 1.0)
  * deadline    : absolute simulation time (seconds) by which the task must finish
"""

import argparse
import csv
import numpy as np

SEED = 42

NUM_TASKS = 2000
NUM_DEVICES = 15
SIM_WINDOW_S = 95.0          # spread arrivals over ~95 s (sim-time-limit = 100 s)

LOCAL_CPU_RATE = 0.5         # must match omnetpp.ini *.iot[*].localCpuRate
LOCAL_POWER = 2.0            # relative power draw while processing locally

# Per-task-type profiles: (cpu_low, cpu_high, slack_low, slack_high)
#   cpu_*   -> cpuDemand range in seconds  (kept within [0.005, 0.2])
#   slack_* -> deadline slack range in seconds (added to arrival time)
TASK_TYPES = {
    "control_signal":   (0.005, 0.05, 3.0, 7.0),    # tight deadline, light compute
    "alert":            (0.005, 0.04, 3.0, 6.0),    # tight deadline
    "sensor_data":      (0.010, 0.08, 5.0, 12.0),   # light compute
    "aggregation":      (0.030, 0.15, 10.0, 18.0),  # loose deadline
    "image_processing": (0.080, 0.20, 6.0, 14.0),   # heavy compute
}
TYPE_NAMES = list(TASK_TYPES.keys())
TYPE_WEIGHTS = [0.30, 0.10, 0.30, 0.15, 0.15]

PRIORITY_LEVELS = [1, 2, 3, 4, 5]
PRIORITY_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]


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
        cpu_lo, cpu_hi, slack_lo, slack_hi = TASK_TYPES[ttype]

        arrival = float(arrivals[i])
        cpu_demand = float(rng.uniform(cpu_lo, cpu_hi))
        slack = float(rng.uniform(slack_lo, slack_hi))
        deadline = arrival + slack

        data_size_kb = float(rng.uniform(10.0, 500.0))
        priority = int(rng.choice(PRIORITY_LEVELS, p=PRIORITY_WEIGHTS))
        memory_mb = float(rng.uniform(50.0, 512.0))
        # Energy if executed locally: (proc time at local speed) * local power
        energy_local = (cpu_demand / LOCAL_CPU_RATE) * LOCAL_POWER

        rows.append({
            "taskId": i + 1,
            "ownerDeviceId": int(rng.integers(0, NUM_DEVICES)),
            "arrivalTime": round(arrival, 4),
            "taskType": ttype,
            "dataSizeKb": round(data_size_kb, 2),
            "cpuDemand": round(cpu_demand, 4),
            "deadline": round(deadline, 4),
            "priority": priority,
            "memoryMb": round(memory_mb, 1),
            "energyLocalEst": round(energy_local, 4),
        })
    return rows


def write_csv(rows: list, path: str) -> None:
    fields = ["taskId", "ownerDeviceId", "arrivalTime", "taskType", "dataSizeKb",
              "cpuDemand", "deadline", "priority", "memoryMb", "energyLocalEst"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows: list) -> None:
    import collections
    print(f"\nGenerated {len(rows)} tasks -> summary statistics")
    print("-" * 60)
    for col in ["arrivalTime", "cpuDemand", "deadline", "dataSizeKb",
                "memoryMb", "energyLocalEst"]:
        vals = np.array([r[col] for r in rows], dtype=float)
        print(f"{col:<16} mean={vals.mean():9.4f}  std={vals.std():9.4f}"
              f"  min={vals.min():9.4f}  max={vals.max():9.4f}")

    types = collections.Counter(r["taskType"] for r in rows)
    print("\nTask type distribution:")
    for t in TYPE_NAMES:
        print(f"  {t:<18} {types[t]:>5}  ({types[t]/len(rows)*100:4.1f}%)")

    devices = collections.Counter(r["ownerDeviceId"] for r in rows)
    print("\nTasks per device (min/max):"
          f" {min(devices.values())} / {max(devices.values())}")


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic IoT tasks.")
    ap.add_argument("-n", "--num", type=int, default=NUM_TASKS,
                    help=f"number of tasks (default {NUM_TASKS})")
    ap.add_argument("-o", "--out", default="tasks.csv",
                    help="output CSV path (default tasks.csv)")
    args = ap.parse_args()

    rows = generate(args.num)
    write_csv(rows, args.out)
    print_summary(rows)
    print(f"\nWrote {len(rows)} tasks to {args.out}")


if __name__ == "__main__":
    main()
