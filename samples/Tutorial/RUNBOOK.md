# RUNBOOK: IoT–Fog Task Offloading Simulation (OMNeT++)

## Overview

This simulation models **15 IoT devices** offloading computational tasks to **5 fog servers**
(spec: *Intelligent Task Offloading for IoT–Fog Computing*). Each IoT device decides, per task,
one of **6 actions**:

- **Action 0**: process the task locally
- **Action 1–5**: offload to fog server 0–4

Five offloading policies are implemented **inside OMNeT++** and can be compared directly:

| Config | Policy | Logic |
|--------|--------|-------|
| `Local` | Local-only | Always process on the IoT device |
| `Random` | Random | Uniform random over {local, fog0..fog4} |
| `RoundRobin` | Round-robin | Always offload, cycling through fog servers |
| `Greedy` | Greedy (least completion time) | Pick min of `local` vs `backlog + 2·linkDelay + cpuDemand` |
| `DQN` | RL (trained DQN) | Neural-net policy trained offline in Python |

The DQN observes a **7-dimensional state**: `[cpuDemand, deadlineSlack, fogLoad0, …, fogLoad4]`.

---

## Architecture

```
╭─────────────────────────────────────────────────────╮
│  Python (offline)                                   │
│  ├─ generate_tasks.py  → tasks.csv (2000 tasks)     │
│  └─ train_dqn.py                                    │
│       ├─ Trains DQN (300 episodes, pure NumPy)      │
│       ├─ Evaluates baselines (Local/Random/         │
│       │   RoundRobin/Greedy/DQN)                    │
│       └─ Exports weights → dqn_weights/*.csv        │
╰──────────────────────┬──────────────────────────────╯
                       │ (tasks.csv + CSV weight files)
                       ▼
╭─────────────────────────────────────────────────────╮
│  OMNeT++ (online)                                   │
│  ├─ IoTDevice.cc: reads tasks.csv, picks action     │
│  │   per policy, tracks e2e delay / deadline /      │
│  │   energy, loads DQN via DqnPolicy.h              │
│  └─ FogServer.cc: FIFO queue (nextFreeTime),        │
│      broadcasts load every 100 ms, sends ack with   │
│      finish time, records utilization               │
╰─────────────────────────────────────────────────────╯
```

---

## Key Files

| File | Purpose |
|------|---------|
| `simulations/generate_tasks.py` | Generates `tasks.csv` (2000 tasks, seed 42, spec schema) |
| `simulations/tasks.csv` | Task set: id, device, arrival, type, size, cpu, deadline, priority, mem, energy |
| `simulations/train_dqn.py` | NumPy DQN trainer + weight exporter (env aligned to the sim) |
| `simulations/dqn_weights/` | Exported weights: `w1,b1,w2,b2,w3,b3,scaler.csv` |
| `simulations/omnetpp.ini` | Configs: `[General]`, `[Config Local/Random/RoundRobin/Greedy/DQN]` |
| `simulations/FogNetwork.ned` | Topology: 15 IoT + 5 Fog, fully connected, 10 ms links |
| `src/IoTDevice.cc` | Policy logic, local processing, metrics |
| `src/FogServer.cc` | FIFO queue, load broadcast, ack, utilization |
| `src/DqnPolicy.h` | Pure C++ MLP inference (reads weight CSVs) |
| `src/Task.msg` | Task + completion-ack message |
| `src/ServerStatus.msg` | Fog→IoT load-feedback message |

---

## Step-by-Step Instructions

### Step 1: Source the OMNeT++ environment

```bash
cd /Users/admin/Personal/University/Project/omnetpp-6.4.0
source setenv
```

> Do this in **every new terminal** before running any command below.

---

### Step 2: Build the simulation

```bash
cd samples/Tutorial
make makefiles     # regenerate Makefile (only needed once, or after adding files)
make               # compile
```

You should see:
```
Creating executable: ../out/clang-release/src/Tutorial
```

If you get errors, run `make clean && make`.

---

### Step 3: Generate the task set (2000 tasks)

```bash
cd simulations
python3 generate_tasks.py
```

This writes `tasks.csv` (2000 rows) with a fixed seed (42) and prints summary statistics.
Verify:
```bash
wc -l tasks.csv     # -> 2001 (header + 2000 tasks)
```

> Optional: change the count with `python3 generate_tasks.py -n 500 -o tasks.csv`.

---

### Step 4: Train the DQN and export weights

```bash
python3 train_dqn.py
```

Takes ~10–30 s (pure NumPy, no GPU). It trains for 300 episodes, prints a
baseline comparison, and exports weights to `dqn_weights/`.

Verify the weights exist:
```bash
ls dqn_weights/     # w1.csv b1.csv w2.csv b2.csv w3.csv b3.csv scaler.csv
```

---

### Step 5: Run each policy

Run from the `simulations/` directory:

```bash
../src/Tutorial -u Cmdenv -c Local
../src/Tutorial -u Cmdenv -c Random
../src/Tutorial -u Cmdenv -c RoundRobin
../src/Tutorial -u Cmdenv -c Greedy
../src/Tutorial -u Cmdenv -c DQN
```

Each run processes all 2000 tasks and writes results to `results/<Config>-#0.sca` / `.vec`.

> The bare `[General]` config (`-c General`) also runs and defaults to round-robin.

---

### Step 6: Compare results

The recorded metrics (spec Section 10) are:

| Metric | Where | Signal / scalar |
|--------|-------|-----------------|
| Deadline hit ratio | IoTDevice | `hitRatio` (scalar), `deadlineHit:mean` |
| End-to-end delay | IoTDevice | `e2eDelay:mean`, `e2eDelay:max` |
| Energy per task | IoTDevice | `energy:mean`, `energy:sum` |
| Local vs offload | IoTDevice | `tasksLocal`, `tasksOffloaded` |
| Fog utilization | FogServer | `utilization` (scalar) |
| Fog queue backlog | FogServer | `queueBacklog:mean/max` |
| Response time | FogServer | `respTime:mean/max` |

Quick side-by-side comparison from the shell:

```bash
printf "%-12s %-10s %-12s %-12s %-10s\n" Policy HitRatio AvgE2E(s) AvgEnergy FogUtil
for cfg in Local Random RoundRobin Greedy DQN; do
  f="results/${cfg}-#0.sca"; [ -f "$f" ] || continue
  hit=$(awk '$3=="hitRatio"{s+=$4;n++}END{if(n)printf "%.3f",s/n}' "$f")
  e2e=$(awk '$3=="e2eDelay:mean"{s+=$4;n++}END{if(n)printf "%.4f",s/n}' "$f")
  en=$(awk '$3=="energy:mean"{s+=$4;n++}END{if(n)printf "%.4f",s/n}' "$f")
  util=$(awk '$3=="utilization"{s+=$4;n++}END{if(n)printf "%.3f",s/n}' "$f")
  printf "%-12s %-10s %-12s %-12s %-10s\n" "$cfg" "$hit" "$e2e" "$en" "$util"
done
```

Example output (values vary with the trained model):
```
Policy       HitRatio   AvgE2E(s)    AvgEnergy    FogUtil
Local        1.000      0.1304       0.2292       0.000
Random       1.000      0.0869       10.4247      0.186
RoundRobin   1.000      0.0827       12.6689      0.230
Greedy       1.000      0.1071       10.3662      0.220
DQN          1.000      0.0965       12.6689      0.230
```

DQN (and the offloading baselines) achieve **lower end-to-end delay than Local**,
satisfying the spec success criterion. Local uses the least energy (no transmission)
but the slowest CPU.

Or use `scavetool` / the OMNeT++ IDE:
```bash
scavetool export results/DQN-*.sca -o dqn_scalars.csv
scavetool export results/Greedy-*.sca -o greedy_scalars.csv
# Or open General.anf in the IDE, add the .sca/.vec files, and plot.
```

---

### Step 7: Plot the results (for the report / presentation)

After running all five policy configs (Step 5), generate comparison charts:

```bash
python3 plot_results.py
```

This reads `results/<Policy>-#0.sca` and writes PNGs to `results/plots/`:

| File | Chart |
|------|-------|
| `summary_dashboard.png` | 6-panel overview (send this one to your teacher) |
| `hit_ratio.png` | Deadline hit ratio per policy |
| `avg_delay.png` | Average end-to-end delay per policy |
| `energy.png` | Average energy per task per policy |
| `fog_utilization.png` | Mean fog-server utilization per policy |
| `local_vs_offload.png` | Local vs offloaded task share per policy |

It also prints a summary table to the console. Example:
```
Policy          Hit%   Delay(ms)    Energy  FogUtil%   Local%
Local          100.0      130.35      0.23       0.0    100.0
Random         100.0       86.85     10.42      18.6     16.8
RoundRobin     100.0       82.75     12.67      23.0      0.0
Greedy         100.0       78.81     10.38      22.0     18.0
DQN            100.0       96.47     12.67      23.0      0.0
```

The offloading policies (and DQN) cut end-to-end delay roughly in half versus
Local, while Local uses far less energy (no transmission) — the classic
latency-vs-energy trade-off to discuss in the report.

---

### Step 8 (Optional): Run with GUI

```bash
../src/Tutorial -c DQN
```

Opens the **Qtenv** GUI to watch messages flow and inspect module state.

---

## Tuning

### DQN training (`train_dqn.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPISODES` | 300 | Training episodes |
| `NUM_TASKS_PER_EP` | 200 | Tasks per episode |
| `HIDDEN_DIM` | 128 | Hidden-layer width |
| `LR` | 0.001 | Learning rate |
| `EPSILON_DECAY` | 0.995 | Exploration decay |
| `GAMMA` | 0.99 | Discount factor |

The training environment (`FogOffloadEnv`) is aligned with the simulation
(fog rate 1.0, local rate 0.5, 10 ms links, cpuDemand ∈ [0.005, 0.2] s,
deadline slack ∈ [3, 18] s) so the learned policy transfers to OMNeT++ (spec §9.1, Option A).

### Task generation (`generate_tasks.py`)

Edit the constants at the top: `NUM_TASKS`, `NUM_DEVICES`, `SIM_WINDOW_S`,
`TASK_TYPES`, `PRIORITY_WEIGHTS`, `LOCAL_CPU_RATE`.

### Simulation (`omnetpp.ini` / `FogNetwork.ned`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `*.fog[*].statusInterval` | `0.1s` | Fog load broadcast interval |
| `*.iot[*].localCpuRate` | `0.5` | Local CPU speed (fog = 1.0) |
| `*.iot[*].linkDelay` | `0.01s` | Link delay used by the greedy estimate |
| `*.iot[*].txEnergyPerKb` | `0.05` | Transmit energy per KB when offloading |
| `sim-time-limit` | `100s` | Simulation duration |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot open task file` | Run from the `simulations/` directory; run `generate_tasks.py` first |
| `Cannot open dqn_weights/w1.csv` | Run `python3 train_dqn.py` first (DQN config only) |
| Build errors after changing `.msg`/`.cc` | `cd src && opp_makemake -f --deep && cd .. && make clean && make` |
| `ModuleNotFoundError: numpy` | `pip3 install numpy` |
| Want fresh results | Delete `simulations/results/` first |
