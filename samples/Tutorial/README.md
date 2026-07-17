# Intelligent Task Offloading in Heterogeneous IoT–Fog (CPU/GPU/QPU) Systems

An OMNeT++ discrete-event simulation of **15 IoT devices** offloading computation
tasks to **5 heterogeneous fog servers** — **2 QPUs (IBM, IonQ), 2 GPUs (RTX 3090),
1 CPU (11-core)** — and comparing offloading policies including a **Deep Q-Network
(DQN)** that learns an intelligent, completion-time-aware placement.

The execution-time and energy models follow the pipeline in `Quantom.md`:
**feature extraction → quantum-suitability detection → execution-time prediction
(per processor) → queue wait → completion-time-aware decision**. See
[`simulations/PROCESSOR_MODEL.md`](simulations/PROCESSOR_MODEL.md) for the formulas.

---

## Table of Contents
- [Motivation](#motivation)
- [System Model](#system-model)
- [RL Formulation](#rl-formulation)
- [Policies](#policies)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Results & Metrics](#results--metrics)
- [References](#references)

---

## Motivation

IoT devices are resource-constrained and often cannot meet task deadlines locally.
Fog servers can absorb such tasks — but the fogs are **heterogeneous**: QPUs only
help *quantum-optimizable* tasks (TSP, Knapsack, Max-Cut, …), GPUs are best for
parallel/GPU-oriented workloads (image/CNN/video), and CPUs are general-purpose.
Sending a non-quantum task to a QPU wastes scarce quantum resources and is slow;
sending a quantum-optimizable task to a GPU forgoes a speedup. A good scheduler
must therefore (a) detect whether a task is quantum-suitable, (b) predict
execution time and energy on each candidate processor, (c) account for queue
backlog and 5G network delay, and (d) place the task where it finishes earliest.

This project implements that pipeline in OMNeT++ and compares four baselines
(Local, Random, Round-Robin, Greedy) against a trained **DQN**.

---

## System Model

### Network Topology
```
   IoT[0] … IoT[14]   ──5G (1–10 ms)──►   Fog[0] IBM   (QPU)
                                            Fog[1] IonQ  (QPU)
                                            Fog[2] RTX3090 (GPU)
                                            Fog[3] RTX3090b (GPU)
                                            Fog[4] CPU-11core (CPU)
```
- 15 IoT devices, fully connected to all 5 fogs.
- 5G links with **random 1–10 ms** one-way delay per link.

### Heterogeneous Fog Servers

| Fog | Name | Type | Throughput | Power | Notes |
|-----|------|------|------------|-------|-------|
| 0 | IBM | QPU | 200000 MFLOP/s | 15000 W | overhead 1.5 ms |
| 1 | IonQ | QPU | 200000 | 12000 W | overhead 2.0 ms |
| 2 | RTX3090 | GPU | 35000 | 350 W | η=0.55, BW=936 MB/s |
| 3 | RTX3090b | GPU | 35000 | 350 W | (second GPU) |
| 4 | CPU-11core | CPU | 1000 | 65 W | |

IoT local device: 300 MFLOP/s, 2 W (slow, low-power).

### Task Model
Each task carries `workload` (MFLOP), `quantumSuitability` Q_s ∈ [0,1],
`dataSizeKb`, `deadline`, `priority`. Task types are split into quantum-optimizable
(Q_s ≥ 0.9), GPU-oriented (Q_s ≈ 0.05), and general control/sensor (Q_s ≈ 0.1–0.35).

### Execution-time & Energy (see PROCESSOR_MODEL.md)
```
CPU:  E = F / P_CPU
GPU:  E = T_launch + 2·D/BW + F / (P_GPU·η)
QPU:  E = T_overhead + F / (P_QPU·Q_s²)            if Q_s ≥ θ  (quantum speedup)
      E = T_overhead + F / P_QPU_classical          if Q_s < θ  (slow host fallback)
Energy = power × exec_time ;  transmission energy = P_tx · D_bits / R_up
```

---

## RL Formulation

### State Space (18 dimensions)
`[workload, Q_s, deadline_slack, netDelay_0..4, fogBacklog_0..4, execEst_0..4]`
— the agent sees the same completion-time ingredients the greedy policy uses,
for each fog.

### Action Space (6 actions, masked)
`0` = local, `1..5` = fog 0..4. A **quantum-suitability mask** (Quantom.md
phase 4) removes QPU actions for tasks with `Q_s < θ = 0.7`, so non-quantum
tasks are never sent to a QPU.

### Reward
```
reward = -200·latency - 0.005·energy - 80·(deadline_miss)
```

### DQN Architecture
`Input(18) → Dense(64) → ReLU → Dense(64) → ReLU → Dense(6)`, trained in pure
NumPy (no PyTorch). Experience replay (50k), target network (update every 10
episodes), ε-greedy decay 1.0 → 0.05, gradient clipping at 1.0.

**Training:** 3000 episodes × 1000 random tasks/episode (per teacher feedback).

---

## Policies

| Config | Policy | Intelligence |
|--------|--------|--------------|
| `Local` | always local | none |
| `Random` | uniform random over all 6 actions | none |
| `RoundRobin` | cycle through fogs | none |
| `Greedy` | `argmin` completion time (network + queue + exec) | formula-based |
| `DQN` | learned policy + suitability mask | RL |

Greedy and DQN estimate, per fog, `C = network_delay + queue_backlog +
execution_time` (Quantom.md phase 8–9) and pick the minimum. The DQN is
expected to beat the dumb baselines and be competitive with Greedy.

---

## Project Structure
```
Tutorial/
├── README.md, RUNBOOK.md, Quantom.md, spec.md, teacher_text.md
├── src/
│   ├── IoTDevice.cc        # policy logic, completion-time estimate, metrics
│   ├── FogServer.cc        # heterogeneous exec (CPU/GPU/QPU), queue, utilization
│   ├── ProcessorModel.h    # shared exec-time formula (single source of truth)
│   ├── DqnPolicy.h         # pure C++ MLP inference + action masking
│   ├── Task.msg, ServerStatus.msg
│   └── Makefile
└── simulations/
    ├── PROCESSOR_MODEL.md  # execution-time + energy formulas (the math)
    ├── FogNetwork.ned      # 15 IoT + 5 Fog, 5G links uniform(1ms,10ms)
    ├── omnetpp.ini         # per-fog specs + policy configs
    ├── generate_tasks.py   # task generator (workload + Q_s, seed 42)
    ├── train_dqn.py        # NumPy DQN trainer (mirrors ProcessorModel.h)
    ├── run_experiments.sh  # 100/200/500/1000 tasks × all policies + plots
    ├── plot_results.py     # bars + per-fog utilization-over-time + multi-N
    ├── dqn_weights/        # exported w1..w3, b1..b3, scaler.csv
    └── results/            # .sca/.vec + plots/
```

---

## Quick Start
```bash
cd /path/to/omnetpp-6.4.0 && source setenv
cd samples/Tutorial/src && make
cd ../simulations
./run_experiments.sh          # train DQN + run 100/200/500/1000 × 5 policies + plots
# or step by step:
python3 generate_tasks.py -n 1000
python3 train_dqn.py
../src/Tutorial -u Cmdenv -c DQN -n .:../src --seed-set=0
python3 plot_results.py --multi
```
See [RUNBOOK.md](RUNBOOK.md) for full instructions.

---

## Results & Metrics

Metrics recorded (per teacher feedback): **energy, latency, per-fog utilization
over time (vector), hit ratio**, plus local/offload share and queue backlog.

| Metric | Where | Signal |
|--------|-------|--------|
| Deadline hit ratio | IoTDevice | `hitRatio`, `deadlineHit:mean` |
| End-to-end delay | IoTDevice | `e2eDelay:mean/max` |
| Energy per task | IoTDevice | `energy:mean/sum` (tx + fog exec) |
| Fog utilization (over time) | FogServer | `fogUtilization` (vector), `utilization` (scalar) |
| Fog exec time / energy | FogServer | `execTime`, `energy` |
| Queue backlog | FogServer | `queueBacklog:mean/max` |

Plots in `results/plots/`: `summary_dashboard.png`, `hit_ratio.png`,
`avg_delay.png`, `energy.png`, `fog_utilization.png`, `per_fog_utilization.png`,
`fog_utilization_over_time_DQN.png`, `local_vs_offload.png`,
`multi_task_comparison.png`, `training_curves.png` (DQN convergence — reward,
loss, hit%, delay over episodes; raw data in `training_history.csv`).

The DQN dramatically outperforms the dumb baselines (Local/Random/RoundRobin)
and is competitive with the formula-based Greedy policy, while factoring energy
into its decisions.

---

## References
- **DQN**: Mnih et al., "Human-level control through deep RL", *Nature* 518 (2015).
- **Offloading energy model**: Mach & Besar (2011); Wang et al. (2018) — `E = power·time`, `E_tx = P_tx·D/R`.
- **Heterogeneous GPU–QPU scheduling**: `Quantom.md` (project conceptual framework).
- **OMNeT++**: Varga, A., *Modeling and Tools for Network Simulation*, Springer 2010.

## License
LGPL v3.0 — see source file headers.
