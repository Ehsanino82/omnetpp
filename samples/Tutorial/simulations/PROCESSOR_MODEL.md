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

IoT local device: **Raspberry Pi 4B** (BCM2711, quad-core Cortex-A72 @ 1.5 GHz),
the kind of low-power edge node common in hospital/health-IoT deployments. Peak
FP32 (NEON, 4 cores) ≈ 12 GFLOP/s, derated to an **effective sustained-available
600 MFLOP/s** under the device's OS + sensing stack and battery constraint;
active CPU power ≈ 5 W. (Battery-throttled, so far weaker than every fog.)

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

### 4.1 Task's time on the server (turnaround time)

The time a task spends *on the fog server* — excluding the 5G network delay — is
the **server turnaround (sojourn) time**, recorded per task by `FogServer`:
```
queue_wait      = start - arrival_at_server         (>= 0; 0 if server idle)
server_turnaround = finish - arrival_at_server = queue_wait + E_fog_j
```
where `arrival_at_server = releaseTime + network_delay` (the instant the task
reaches the fog), `start = max(arrival_at_server, nextFreeTime)` (FCFS), and
`finish = start + E_fog_j`. The IoT device's end-to-end delay is then
`e2e = network_delay + server_turnaround` (the ack return trip is not added).
Signals: `queueWait`, `serverTurnaround`, `execTime`, `respTime` (= net+queue+exec).

### 4.2 Local-feasibility rule (which tasks may run locally)

Not every task may run on the battery-powered Pi. The decision layer masks the
**local** action based on task class (teacher feedback: *"GPU and quantum tasks
should always be offloaded; CPU tasks may be local"*):
```
quantum task   (Qs >= theta)        -> local NOT allowed  (must offload)
GPU-oriented   (image/cnn/video)    -> local NOT allowed  (must offload)
CPU/control    (sensor/control/...) -> local allowed
```
This is applied to every policy (`local`, `random`, `greedy`, `dqn`) by removing
action 0 from the feasible set when the task is GPU- or quantum-oriented. The
`Local` baseline therefore runs only CPU/control tasks locally and force-offloads
GPU/quantum tasks to the greedy-best fog. This prevents heavy tasks from draining
the battery / blowing deadlines on the weak IoT device.

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

## 7. Formula reference sheet

| Quantity | Formula | Source / reference |
|---|---|---|
| CPU exec time | `E_CPU = F / P_CPU` | Quantom.md §8 (Phase 5); classical compute throughput model |
| GPU exec time | `E_GPU = T_launch + 2·D/BW + F/(P_GPU·η)` | Quantom.md §8 (Phase 5): `E_GPU = T_copy + T_kernel + T_copyback` and `F/(P_GPU·η)` |
| QPU exec time (suitable) | `E_QPU = T_overhead + F/(P_QPU·Q_s²)` | Quantom.md §9 (Phase 6): variational `Iterations×CircuitTime`, speedup scales with suitability |
| QPU exec time (unsuitable) | `E_QPU = T_overhead + F/P_QPU_host` (50 MFLOP/s) | Quantom.md §1.2: QPU wasteful for non-quantum workloads (finite classical fallback) |
| Queue waiting time | `W = Σ E_k` over queued tasks | Quantom.md §10 (Phase 7) |
| Completion time | `C = network_delay + W + E` | Quantom.md §11 (Phase 8): `C = W + E`, plus 5G network delay |
| Decision rule | `argmin C` over feasible processors | Quantom.md §12 (Phase 9) |
| Quantum suitability | `Q_s ∈ [0,1]`, threshold `θ=0.7` | Quantom.md §7 (Phase 4) |
| Dynamic energy | `E = P · T` (power × exec time) | Standard dynamic-power model; Quantom.md §14 lists energy as a metric |
| 5G transmission energy | `E_tx = P_tx · D_bits / R_up` | 5G uplink energy model (P_tx=1.5 W, R_up=100 Mbps) |
| Server turnaround | `T_server = queue_wait + E_fog` | Queueing theory sojourn time (M/G/1-style FCFS) |
| Fog utilization | `U = busy_time / sim_duration ∈ [0,1]` | Standard server utilization; queue wait excluded |
| Local feasibility | local only if task is CPU/control class | Teacher feedback (battery IoT device); spec §5.2 local-feasibility check |

**Device specs (cited):** Raspberry Pi 4B — Broadcom BCM2711, quad-core Cortex-A72
@ 1.5 GHz (peak FP32 ≈ 12 GFLOP/s via NEON), ~5 W active. Fog GPUs: NVIDIA RTX
3090 (~350 W TDP, 936 GB/s memory BW). Fog QPUs: IBM superconducting / IonQ
trapped-ion (cryostat-class ~12–15 kW). Fog CPU: 11-core server. Throughputs in
the simulation are *effective sustained* values in MFLOP/s, derated from peak for
realism and kept in a consistent ratio (Pi ≪ CPU-fog < GPU-fog < QPU) so the
offloading trade-off is visible at millisecond scale.
