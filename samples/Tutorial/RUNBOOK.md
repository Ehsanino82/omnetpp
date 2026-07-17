# RUNBOOK: IoT–Fog Task Offloading Simulation (OMNeT++)

## Overview

This simulation models **15 IoT devices** offloading computational tasks to **5
heterogeneous fog servers** and compares offloading policies. Per the teacher's
feedback, the fog servers are **no longer identical**:

| Fog | Name        | Type | Notes |
|-----|-------------|------|-------|
| 0   | IBM         | QPU  | superconducting quantum, 1.5 ms overhead |
| 1   | IonQ        | QPU  | trapped-ion quantum, 2.0 ms overhead |
| 2   | RTX3090     | GPU  | eta = 0.55 |
| 3   | RTX3090b    | GPU  | (second GPU) |
| 4   | CPU-11core  | CPU  | 11-core CPU |

Each task carries a **workload** (MFLOP) and a **quantum-suitability score**
`Q_s` in [0,1] — some tasks are quantum-optimizable (TSP, Knapsack, Max-Cut, …),
others are GPU-oriented (image/CNN/video). Execution time and energy are computed
**per processor** from the formulas in [`PROCESSOR_MODEL.md`](simulations/PROCESSOR_MODEL.md)
(Quantom.md phases 5–9). Network delay is random **1–10 ms** (5G).

Five policies are implemented inside OMNeT++:

| Config | Policy | Logic |
|--------|--------|-------|
| `Local` | Local-only | Always process on the IoT device |
| `Random` | Random | Uniform random over {local, fog0..fog4} |
| `RoundRobin` | Round-robin | Always offload, cycling fogs (no intelligence) |
| `Greedy` | Intelligent completion-time | `argmin` of `network + queue + exec_time` |
| `DQN` | RL (trained DQN) | 18-dim state, learned policy |

The **Greedy** and **DQN** policies are the *intelligent* ones: before sending a
task they estimate, for each fog, the execution time (CPU/GPU/QPU formula), add
the current queue backlog and the 5G network delay, and pick the minimum
completion time. The DQN is expected to beat the dumb baselines (Local/Random/
RoundRobin) and be competitive with Greedy.

---

## Architecture

```
╭────────────────────────────────────────────────────────────╮
│  Python (offline)                                          │
│  ├─ generate_tasks.py  → tasks.csv (workload, Q_s, …)      │
│  └─ train_dqn.py                                            │
│       ├─ Mirrors ProcessorModel.h (CPU/GPU/QPU exec+energy) │
│       ├─ 3000 episodes × 1000 tasks, 18-dim state           │
│       └─ Exports weights → dqn_weights/*.csv                │
╰───────────────────────────┬────────────────────────────────╯
                            │ (tasks.csv + CSV weight files)
                            ▼
╭────────────────────────────────────────────────────────────╮
│  OMNeT++ (online)                                          │
│  ├─ IoTDevice.cc: reads tasks.csv, estimates completion     │
│  │   time per fog (ProcessorModel.h), picks action,         │
│  │   tracks e2e delay / deadline / energy, loads DQN        │
│  └─ FogServer.cc: heterogeneous exec (CPU/GPU/QPU), FIFO    │
│      queue, broadcasts backlog + utilization vector, ack    │
╰────────────────────────────────────────────────────────────╯
```

---

## Key Files

| File | Purpose |
|------|---------|
| `simulations/PROCESSOR_MODEL.md` | Execution-time + energy formulas (the math) |
| `simulations/generate_tasks.py` | Generates `tasks.csv` with workload + Q_s (seed 42) |
| `simulations/train_dqn.py` | NumPy DQN trainer (heterogeneous env), exports weights |
| `simulations/run_experiments.sh` | Full loop: 100/200/500/1000 tasks × all policies + plots |
| `simulations/plot_results.py` | Bar charts + per-fog utilization-over-time + multi-N comparison |
| `simulations/dqn_weights/` | Exported weights: `w1,b1,w2,b2,w3,b3,scaler.csv` |
| `simulations/omnetpp.ini` | Per-fog specs + policy configs |
| `simulations/FogNetwork.ned` | 15 IoT + 5 Fog, 5G links `delay = uniform(1ms,10ms)` |
| `src/ProcessorModel.h` | Shared CPU/GPU/QPU exec-time formula (single source of truth) |
| `src/IoTDevice.cc` | Policy logic, completion-time estimate, metrics |
| `src/FogServer.cc` | Heterogeneous execution, queue, utilization vector, ack |
| `src/DqnPolicy.h` | Pure C++ MLP inference (reads weight CSVs) |
| `src/Task.msg` | Task + completion-ack message |

---

## Step-by-Step Instructions

### Step 1: Source the OMNeT++ environment

```bash
cd /Users/admin/Personal/University/Project/ehsan/omnetpp-6.4.0
source setenv
```

> Do this in **every new terminal** before running any command below.

### Step 2: Build the simulation

```bash
cd samples/Tutorial/src
make               # compile (creates ../out/clang-release/src/Tutorial, linked to src/Tutorial)
```

If you changed `.msg`/`.cc` and get errors: `make clean && make`.

### Step 3: Generate a task set

```bash
cd ../simulations
python3 generate_tasks.py            # default 2000 tasks
python3 generate_tasks.py -n 1000    # any count
```

### Step 4: Train the DQN

```bash
python3 train_dqn.py            # 3000 episodes × 1000 tasks (~20 min, pure NumPy)
python3 train_dqn.py --quick    # smoke test (50 ep × 200 tasks, ~5 s)
```

Verify: `ls dqn_weights/` → `w1.csv b1.csv w2.csv b2.csv w3.csv b3.csv scaler.csv`.

### Step 5: Run the full experiment loop (recommended)

```bash
./run_experiments.sh                 # trains DQN + runs 100/200/500/1000 × 5 policies + plots
SKIP_TRAIN=1 ./run_experiments.sh    # reuse existing dqn_weights/
QUICK=1   ./run_experiments.sh        # quick DQN smoke train + full loop
```

This writes results to `results/n{100,200,500,1000}/` and plots to `results/plots/`.

### Step 6: Run a single policy manually

```bash
../src/Tutorial -u Cmdenv -c Greedy -n .:../src --seed-set=0
../src/Tutorial -u Cmdenv -c DQN    -n .:../src --seed-set=0
```

### Step 7: Plot the results

```bash
python3 plot_results.py                 # current results/ dir
python3 plot_results.py --multi         # multi-task-count comparison (100/200/500/1000)
```

Charts written to `results/plots/`:

| File | Chart |
|------|-------|
| `summary_dashboard.png` | 6-panel overview |
| `hit_ratio.png`, `avg_delay.png`, `energy.png` | Per-policy bars |
| `fog_utilization.png`, `per_fog_utilization.png` | Mean utilization (overall + per fog) |
| `fog_utilization_over_time_DQN.png` | Per-fog utilization over time (vector) |
| `local_vs_offload.png` | Local vs offloaded share |
| `multi_task_comparison.png` | Metrics vs task count (100/200/500/1000) |
| `training_curves.png` | DQN convergence: reward, loss, hit%, delay over episodes (also `dqn_weights/training_curves.png`, raw data `training_history.csv`) |

### Step 8 (Optional): Run with GUI

```bash
../src/Tutorial -c DQN -n .:../src
```

---

## Recorded metrics (spec Section 10 + teacher feedback)

| Metric | Where | Signal / scalar |
|--------|-------|-----------------|
| Deadline hit ratio | IoTDevice | `hitRatio`, `deadlineHit:mean` |
| End-to-end delay (latency) | IoTDevice | `e2eDelay:mean/max` |
| Energy per task | IoTDevice | `energy:mean/sum` (tx + fog exec) |
| Fog execution time | FogServer | `execTime:mean/max` |
| Fog exec energy | FogServer | `energy:mean/sum` |
| Fog utilization (over time) | FogServer | `fogUtilization` (vector), `utilization` (scalar) |
| Fog queue backlog | FogServer | `queueBacklog:mean/max` |
| Local vs offload | IoTDevice | `tasksLocal`, `tasksOffloaded` |

---

## Tuning

### DQN training (`train_dqn.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPISODES` | 3000 | Training episodes (teacher: ≥ 3000) |
| `NUM_TASKS_PER_EP` | 1000 | Tasks per episode (teacher: 1000) |
| `STATE_DIM` | 18 | `[W, Qs, slack, net×5, backlog×5, exec×5]` |
| `HIDDEN_DIM` | 64 | Hidden-layer width |
| `TRAIN_EVERY` | 4 | Gradient update cadence |
| `LR` / `GAMMA` | 0.001 / 0.99 | Learning rate / discount |

### Simulation (`omnetpp.ini`)

Per-fog specs (`*.fog[*].processorType`, `throughput`, `power`, …), 5G link delay
(`uniform(1ms,10ms)` in `FogNetwork.ned`), IoT local model (`localThroughput`,
`localPower`, `txPower`, `uplinkRate`).

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot open task file` | Run from `simulations/`; run `generate_tasks.py` first |
| `Cannot open dqn_weights/w1.csv` | Run `python3 train_dqn.py` first (DQN config only) |
| Build errors after changing `.msg`/`.cc` | `cd src && make clean && make` |
| `opp_configfilepath: Command not found` | You forgot `source setenv` (Step 1) |
| `ModuleNotFoundError: numpy` | `pip3 install numpy matplotlib` |
| Want fresh results | Delete `simulations/results/` first |
