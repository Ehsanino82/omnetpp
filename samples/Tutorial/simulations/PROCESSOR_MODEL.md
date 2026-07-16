# Heterogeneous Processor & Energy Model

This document specifies the **execution-time** and **energy** formulas used by the
simulation. They are implemented once in [`src/ProcessorModel.h`](../src/ProcessorModel.h)
(shared by `FogServer.cc` for actual execution and `IoTDevice.cc` for the
completion-time estimate) and mirrored exactly in
[`simulations/train_dqn.py`](train_dqn.py) so the DQN trains on the same model the
simulator runs.

This addresses the teacher's feedback: *fogs are no longer identical*, *fog
selection is intelligent (formula-based, not round-robin)*, *energy is dynamic
(task-size and processor dependent)*.

## 1. Fog servers (heterogeneous)

| Index | Name        | Type | Throughput (MFLOP/s) | Power (W) | Notes |
|-------|-------------|------|----------------------|-----------|-------|
| 0     | IBM         | QPU  | 200000 (P_QPU_max)   | 15000     | superconducting, overhead 1.5 ms |
| 1     | IonQ        | QPU  | 200000               | 12000     | trapped ion, overhead 2.0 ms (slower gates) |
| 2     | RTX3090     | GPU  | 35000                | 350       | eta = 0.55, BW = 936 MB/s, launch 0.1 ms |
| 3     | RTX3090b    | GPU  | 35000                | 350       | (second GPU) |
| 4     | CPU-11core  | CPU  | 1000                 | 65        | 11-core CPU |

IoT local device: throughput 300 MFLOP/s, power 2 W (slow, low-power).

## 2. Task model

Each task carries:
- `workload` **F** in MFLOP (million FLOP) — the required computation.
- `quantumSuitability` **Q_s** in [0,1] — whether the task is quantum-optimizable
  (Quantom.md Phase 4). Quantum-optimizable types (TSP, Knapsack, Max-Cut,
  Portfolio, Molecular-sim) have Q_s ≥ 0.9; GPU-oriented types (image/CNN/video)
  have Q_s ≈ 0.05; control/sensor types Q_s ≈ 0.1–0.35.
- `dataSizeKb` **D** — payload size (drives GPU copy time + transmission energy).
- `deadline` — absolute sim time by which the task must finish.

## 3. Execution-time prediction (Quantom.md Phases 5–6, 8)

Let `theta = 0.7` be the quantum-suitability threshold.

**CPU** (Phase 5, classical):
```
E_CPU = F / P_CPU
```

**GPU** (Phase 5, with copy overhead):
```
E_GPU = T_launch + 2 * D / BW + F / (P_GPU * eta)
        \____________ copy __________/   \__ kernel __/
```

**QPU** (Phase 6, variational / quantum speedup):
```
              { T_overhead + F / (P_QPU * Q_s^2)        if Q_s >= theta   (quantum speedup)
E_QPU =       { T_overhead + F / P_QPU_classical        if Q_s <  theta   (falls back to slow
              {                                                     QPU host CPU = 50 MFLOP/s)
```
Tasks that are **not** quantum-optimizable (Q_s < theta) cannot use the quantum
algorithm; the QPU runs them on its classical host controller, which is slow — a
finite penalty (not a division-by-zero), modeling that sending unsuitable tasks
to a QPU wastes scarce quantum resources (Quantom.md Section 1.2).

## 4. Completion-time prediction (Phase 8) — the intelligent selection

For each candidate placement `a` (local or fog `j`), the device estimates:
```
C_local = local_backlog + E_local
C_fog_j = network_delay_j + queue_backlog_j + E_fog_j
```
- `network_delay_j` — 5G one-way delay, random **1–10 ms** per link (teacher:
  "network delay random 1-10 ms, since our infrastructure is 5G").
- `queue_backlog_j` — current waiting time at fog `j` (broadcast periodically via
  `ServerStatus`, Quantom.md Phase 7: `W = sum of queued E_k`).

The **Greedy** policy picks `argmin C` (Quantom.md Phase 9: *select the processor
with the smallest completion time*). This is the formula-based intelligent
selection the teacher asked for, replacing the old round-robin assignment.

## 5. Energy model (dynamic, task- and processor-dependent)

Energy = power × execution time (dynamic-power model), plus transmission energy
for offloaded tasks:
```
E_local   = P_local   * E_local_exec
E_fog_j   = P_fog_j   * E_fog_j_exec
E_tx      = P_tx      * D_bits / R_up         (5G uplink transmission)
```
- `P_tx = 1.5 W`, `R_up = 100 Mbps` (5G).
- Total energy recorded per task:
  - local:  `E_local`
  - offload:`E_tx + E_fog_j`

Because each processor has a different power and a different execution time for
the same task, the energy is **dynamic**: a GPU is fast and moderately powered
(low energy), a QPU is ultra-fast for suitable tasks but very power-hungry (high
energy), the CPU fog is slow and wastes energy, and the IoT device is low-power
but too slow to meet deadlines. This creates the trade-off the DQN learns.

## 6. DQN state and reward

- **State (18-dim):** `[workload, Q_s, deadline_slack, netDelay_0..4,
  fogBacklog_0..4, execEst_0..4]` — gives the agent the same completion-time
  ingredients the greedy policy uses, per fog.
- **Actions (6):** 0 = local, 1–5 = fog 0–4.
- **Reward:** `-100 * latency - 0.02 * energy - 50 * deadline_miss`.
- **Training:** 3000 episodes × 1000 random tasks/episode (teacher feedback),
  replay buffer, target network, epsilon-greedy decay.
