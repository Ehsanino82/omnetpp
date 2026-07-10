# Project Specification: Intelligent Task Offloading Simulation for IoT–Fog Computing Systems

**Document:** `spec.md`
**Type:** Bachelor Project Specification
**Simulator:** OMNeT++
**Author:** [Ehsan Rahmani]
**Supervisor:** [Dr.Safaei]

---

## 1. Overview

This project designs, implements, and evaluates a **task offloading simulation** for an IoT–Fog computing environment using **OMNeT++**. The system consists of **15 IoT devices** generating computational tasks and **5 fog servers** capable of processing offloaded tasks. The goal is to compare a set of offloading decision strategies — including a **Reinforcement-Learning (RL) based intelligent offloading policy** — against baseline strategies (Local-only, Random, Greedy), and to demonstrate that intelligent offloading reduces task delay, deadline violations, and energy consumption compared to naive strategies.

The project reuses and adapts two pieces of prior work:

1. **`Shortest_path.ipynb`** — a Python/PyTorch DQN-based implementation that already models (a) shortest-path routing delay estimation between nodes, and (b) a local-vs-offload decision environment (`TETRISOffloadEnv`) trained with a DQN, compared against Local/Random/Greedy baselines.
2. **`Quantum.md`** — a conceptual framework ("Intelligent Task Offloading in Heterogeneous GPU–Quantum Systems") describing a generic **feature extraction → suitability detection → execution-time prediction → completion-time-aware decision** pipeline. This project reuses the *architectural pattern* of that framework (not the quantum computing part) and applies it to the IoT–Fog domain.

---

## 2. Motivation & Related Work

IoT devices are resource-constrained (CPU, battery, memory) and often cannot execute latency-sensitive or compute-heavy tasks locally within their deadlines. Fog servers, located closer to the network edge than the cloud, can absorb such tasks — but naive offloading (e.g., always offload, or random selection) can overload fog queues and cause deadline violations.

`Quantum.md` proposes a generic decision architecture:

```
Task → Feature Extraction → Suitability Check → Execution-Time Prediction (per candidate processor)
     → Completion-Time Prediction (execution + queue wait) → Select processor with min completion time
```

This exact pattern is reused here, replacing "GPU vs QPU" with **"Local IoT device vs one of 5 Fog servers"**, and replacing the QPU-suitability classifier with a **local-execution feasibility check** (can the device meet the deadline locally, given battery/CPU state?).

`Shortest_path.ipynb` already implements a simplified version of this pipeline (local_delay vs path_delay+fog_delay, using a DQN). This project **extends** that notebook from a generic 2-action (local/offload) and 10-generic-node abstraction into a **concrete, OMNeT++-simulated 15-device / 5-fog-server system** with a 6-action decision space (local + 5 fog servers).

---

## 3. Objectives

1. Generate a realistic synthetic dataset of **2000 IoT tasks** (`tasks.csv`).
2. Build an OMNeT++ network model with **15 IoT devices** and **5 fog servers** connected through a network topology with realistic link delays and bandwidths.
3. Simulate task generation, transmission, queuing, and processing in OMNeT++, producing realistic delay/queue-backlog measurements (replacing the purely synthetic random values currently used in `Shortest_path.ipynb`).
4. Adapt the DQN-based offloading agent from `Shortest_path.ipynb` to:
   - Support 6 actions (Local, Fog1…Fog5) instead of 2 (Local/Offload).
   - Use real network/queue statistics derived from OMNeT++ instead of synthetic `np.random` values.
5. Implement and compare baseline offloading policies: **Local-only**, **Random**, **Greedy (least completion time)**, and **RL/DQN (TETRIS-style)**.
6. Evaluate all policies using standard metrics (average delay, deadline violation ratio, hit ratio, energy consumption, fog utilization, throughput).
7. Present final results (tables, plots) comparing all strategies for the thesis/report and presentation to the supervisor.

---

## 4. Scope

**In scope:**
- Network-level simulation in OMNeT++ (topology, transmission delay, queuing at fog servers).
- Synthetic task dataset generation (2000 tasks).
- Offloading decision logic: baseline heuristics + RL agent (trained in Python, either embedded into OMNeT++ via a trained policy lookup, or run as a parallel/offline evaluation using OMNeT++-derived statistics).
- Evaluation and result visualization.

**Out of scope (future work):**
- Actual quantum processing unit (QPU) integration — `Quantum.md` is used only as a conceptual template for the decision pipeline, not implemented in this project.
- Mobility of IoT devices (static topology assumed).
- Security/privacy aspects of offloading.

---

## 5. System Model

### 5.1 Network Topology

```
        [IoT Device 1] ---\
        [IoT Device 2] ----\
             ...             \--- [Access Network / Router] --- [Fog Server 1]
        [IoT Device 14] ---/                                  |- [Fog Server 2]
        [IoT Device 15] --/                                   |- [Fog Server 3]
                                                               |- [Fog Server 4]
                                                               |- [Fog Server 5]
```

- 15 `IoTDevice` modules, each connected to a shared `Router`/`Gateway` module (or directly to all fog servers via a switch, depending on NED design).
- 5 `FogServer` modules, each with its own task queue, processing rate, and utilization tracking.
- Link delays and data rates are configurable per link (`.ini` parameters), representing realistic wireless/wired IoT-to-fog latency (e.g., 3–5 ms per hop, similar to the ranges already used in `Shortest_path.ipynb`: `EDGE_DELAY_LOW=3e-3`, `EDGE_DELAY_HIGH=5e-3`).

### 5.2 IoT Device Model

Each `IoTDevice`:
- Generates tasks according to `tasks.csv` (or a stochastic arrival process, e.g., Poisson).
- Has limited local CPU capacity (fixed or randomized per device) and a local processing delay distribution (similar to `local_delay ~ Uniform(6ms, 8ms)` in the notebook).
- Makes an offloading decision per task using one of the policies (Section 8).
- Tracks per-task metrics: latency, deadline miss, energy consumed.

### 5.3 Fog Server Model

Each `FogServer`:
- Has a processing capacity (CPU cycles/sec), randomly assigned within a range (analogous to `CAPACITY_LOW=100`, `CAPACITY_HIGH=300` in the notebook).
- Maintains an FCFS (or priority) queue of incoming tasks (reuse `FogQueueFCFS` logic).
- Reports **current backlog** (queue waiting time) which is used by the Greedy and RL policies for completion-time estimation.

### 5.4 Task Model

Each task `T_i` (aligned with `Quantum.md` Phase 1, adapted for IoT-Fog):

```
T_i = (task_id, source_device_id, task_type, arrival_time,
       data_size_kb, cpu_cycles_required, deadline_ms,
       priority, memory_requirement_mb)
```

---

## 6. Data Generation — `tasks.csv`

### 6.1 Schema

| Column | Type | Description |
|---|---|---|
| `task_id` | int | Unique task identifier (0–1999) |
| `device_id` | int | Source IoT device (0–14) |
| `arrival_time_ms` | float | Simulation-time arrival (Poisson process) |
| `task_type` | string | e.g., `sensor_data`, `image_processing`, `control_signal`, `aggregation`, `alert` |
| `data_size_kb` | float | Task payload size (for transmission delay) |
| `cpu_cycles_million` | float | Required computation (MI) |
| `deadline_ms` | float | Max tolerable latency |
| `priority` | int (1–5) | Task priority |
| `memory_mb` | float | Required memory |
| `energy_local_est` | float | Estimated energy if processed locally |

### 6.2 Generation Rules (suggested distributions)

- `arrival_time_ms`: Poisson process, mean inter-arrival ~ 5–20 ms across 2000 tasks (or evenly spread across a simulation window, e.g., 60 seconds).
- `data_size_kb`: `Uniform(10, 500)`
- `cpu_cycles_million`: `Uniform(50, 500)` (varies by `task_type`)
- `deadline_ms`: `Uniform(5, 50)` depending on task type (control signals = tighter deadlines, aggregation = looser)
- `priority`: weighted random (1–5)
- `device_id`: uniform random among 15 devices

### 6.3 Deliverable

A Python script `generate_tasks.py` that:
- Uses a fixed random seed (consistent with `SEED = 42` used in the notebook, for reproducibility).
- Outputs `data/tasks.csv` with 2000 rows matching the schema above.
- Includes summary statistics printout (mean/std per column) for sanity-checking.

---

## 7. OMNeT++ Simulation Design

### 7.1 Project Structure

```
FogOffloadSim/
├── simulations/
│   ├── omnetpp.ini
│   └── Network.ned
├── src/
│   ├── IoTDevice.ned / IoTDevice.cc / IoTDevice.h
│   ├── FogServer.ned / FogServer.cc / FogServer.h
│   ├── Router.ned (optional)
│   ├── Task_m.msg
│   └── OffloadingPolicy.cc (baseline logic; RL policy imported as lookup table/ONNX or via file exchange)
├── data/
│   └── tasks.csv
├── python/
│   ├── generate_tasks.py
│   ├── Shortest_path.ipynb   (adapted RL training/eval)
│   └── results_analysis.py
└── results/
    ├── raw/
    └── plots/
```

### 7.2 NED Modules

- `IoTDevice`: reads assigned tasks from `tasks.csv`, decides offload target, sends `Task` messages.
- `FogServer`: receives `Task` messages, queues and processes them, sends completion/ack messages back.
- `Network`: top-level module wiring 15 `IoTDevice` + 5 `FogServer` (+ optional `Router`) with channel delays/datarates.

### 7.3 Message Definition (`Task_m.msg`)

```
message Task {
    int taskId;
    int deviceId;
    string taskType;
    double dataSizeKb;
    double cpuCyclesMillion;
    double deadlineMs;
    int priority;
    double arrivalTime;
    int assignedProcessor; // -1 = local, 0-4 = fog server index
}
```

### 7.4 Simulation Flow

1. At `t = arrivalTime`, `IoTDevice[i]` creates the `Task` (loaded from `tasks.csv`).
2. The device's **offloading policy module** decides: process locally OR send to one of the 5 fog servers.
3. If local → simulate local processing delay + energy cost, record result.
4. If offloaded → message travels over the network channel (propagation + transmission delay) → arrives at `FogServer[j]` → enqueued → processed FCFS → completion event sent back.
5. Each task's **end-to-end latency**, **deadline hit/miss**, and **energy** are logged to a results CSV.

### 7.5 Key `.ini` Parameters

```ini
[General]
network = Network
sim-time-limit = 60s
**.numIoTDevices = 15
**.numFogServers = 5
**.channelDelay = uniform(3ms, 5ms)
**.fogCapacity = uniform(100, 300)
**.localProcDelay = uniform(6ms, 8ms)
**.offloadingPolicy = "greedy" # local | random | greedy | rl
```

---

## 8. Offloading Decision Strategies

### 8.1 Baseline Policies (from `Shortest_path.ipynb`, reused directly)

| Policy | Logic |
|---|---|
| **Local** | Always process on the IoT device |
| **Random** | Randomly choose local or one of the 5 fog servers |
| **Greedy** | Estimate completion time for local and each fog server (`fog_backlog + path_delay + fog_delay`), pick the minimum |

### 8.2 RL-Based Policy (Adapted DQN, "TETRIS"-style)

Adapted from `TETRISOffloadEnv` / `OffloadDQN` in `Shortest_path.ipynb`:

- **State**: `[local_delay, path_delay_to_each_fog(5), fog_backlog(5)]` → dimension updated from 4 to **11** (was 2-action/4-dim state; must be redesigned for 6 actions).
- **Action space**: 6 (0 = local, 1–5 = Fog server 1–5) — changed from the original 2-action space.
- **Reward**: same structure as original (`-latency*100`, `-50` penalty on deadline miss), extended to penalize overloading a specific fog server.
- **Path delay estimation**: reuse `PathEstimator`/`RoutingDQN`/`TETRISGraphEnv` concept, but node graph resized to represent the actual IoT-device-to-fog-server topology (or replaced entirely with real per-link delays measured from OMNeT++).

### 8.3 Required Adaptation Work (Checklist)

- [ ] Change `ACTION_DIM` from 2 → 6 in `OffloadDQN`.
- [ ] Change `TETRISOffloadEnv` to track 5 independent `FogQueueFCFS` instances (one per fog server).
- [ ] Change `_get_obs()` to return an 11-dim (or agreed) state vector.
- [ ] Replace synthetic `np.random.uniform` delay generation with values imported from OMNeT++ simulation output (see Section 9).
- [ ] Update `GreedyPolicy` to compare 6 completion-time estimates instead of 2.

---

## 9. Integration Architecture: OMNeT++ ↔ Python RL

Two integration approaches are considered; **Option A is recommended** for bachelor-thesis scope due to simplicity and reproducibility.

### 9.1 Option A — Two-Stage Offline Integration (Recommended)

1. Run OMNeT++ simulation with a **simple/greedy policy** first to collect realistic link delays, fog processing rates, and queue backlog traces → export to CSV (`network_stats.csv`).
2. Feed these real statistics into the Python RL environment (`TETRISOffloadEnv`) in place of the synthetic `np.random.uniform(...)` calls, and train the DQN policy offline in `Shortest_path.ipynb`.
3. Export the trained policy's **decision rule / Q-table behavior** (or a simplified lookup/threshold policy distilled from it) back into OMNeT++ as `RLOffloadingPolicy.cc`.
4. Re-run the full OMNeT++ simulation using this policy and compare against Local/Random/Greedy runs — all within OMNeT++ for fair, consistent measurement.

### 9.2 Option B — Live Co-Simulation (Advanced, Optional)

- Use OMNeT++'s support for external process communication (sockets) to query a running Python (PyTorch) DQN model at decision time.
- More accurate but significantly more complex to implement and debug for a bachelor timeline — recommended only as a stretch goal.

### 9.3 Recommendation

Use **Option A**. Clearly document in the report that the RL model was trained on statistics derived from the OMNeT++ simulation, then its resulting policy was implemented/evaluated inside OMNeT++ for final comparison.

---

## 10. Evaluation Metrics

Aligned with both `Shortest_path.ipynb` metrics and `Quantum.md` Section 14:

| Metric | Description |
|---|---|
| Average end-to-end delay (ms) | Mean latency across all tasks |
| Deadline violation ratio | % of tasks exceeding their deadline |
| Hit ratio | 1 − deadline violation ratio |
| Local vs Offload ratio | % of tasks processed locally vs at each fog server |
| Fog server utilization | Per-server load over time |
| Throughput | Tasks completed per second |
| Energy consumption | Average energy per task (local vs offloaded) |
| Average hops / path delay | Network path delay to assigned fog server |

---

## 11. Experiment Plan

| Step | Description |
|---|---|
| 1 | Generate `tasks.csv` (2000 tasks) |
| 2 | Build baseline OMNeT++ model (topology, IoT devices, fog servers) with Greedy policy |
| 3 | Run baseline simulations (Local, Random, Greedy) — collect metrics |
| 4 | Export network/queue statistics from OMNeT++ run |
| 5 | Adapt & retrain DQN in `Shortest_path.ipynb` using exported statistics (6-action version) |
| 6 | Implement trained RL policy inside OMNeT++ |
| 7 | Run RL-policy simulation — collect metrics |
| 8 | Compare all four policies (tables + plots: delay, hit ratio, utilization) |
| 9 | Write final report & prepare presentation for supervisor |

---

## 12. Relation to `Quantum.md` Framework

`Quantum.md` is **not implemented** in this project (no actual QPU) but its **decision architecture is the conceptual blueprint** reused here:

| Quantum.md Concept | Fog Offloading Equivalent |
|---|---|
| Feature Extraction (Phase 3) | Task feature vector: data size, CPU cycles, deadline, priority |
| Quantum Suitability Detector (Phase 4) | *(Not strictly needed — all tasks are offload-candidates, but a "local feasibility check" can play an analogous role)* |
| GPU Execution Time Prediction (Phase 5) | Local processing delay prediction |
| QPU Execution Time Prediction (Phase 6) | Fog execution time prediction (per fog server) |
| Queue Waiting Time (Phase 7) | Fog queue backlog (`FogQueueFCFS`) |
| Completion Time Prediction (Phase 8) | `C = backlog + path_delay + processing_delay` |
| Intelligent Offloading Decision (Phase 9) | Select min completion-time among {Local, Fog1..Fog5} — implemented as Greedy & RL policies |

This mapping should be explicitly discussed in the thesis introduction/related-work section to justify the architectural choices and show awareness of the broader research context.

---

## 13. Deliverables

1. `generate_tasks.py` + `tasks.csv` (2000 tasks)
2. OMNeT++ project (`FogOffloadSim/`) with full source code, NED files, and `.ini` configs
3. Adapted `Shortest_path.ipynb` (6-action DQN version) + trained model weights
4. Results: CSV logs + plots (delay, hit ratio, utilization, comparison bar charts)
5. Final report (methodology, results, discussion, conclusion)
6. Presentation slides for supervisor meeting

---

## 14. Timeline / Milestones (suggested, adjust to your schedule)

| Week | Task |
|---|---|
| 1 | Finalize spec, set up OMNeT++ project skeleton |
| 2 | Implement task generator, `tasks.csv` |
| 3 | Implement IoTDevice/FogServer NED + basic simulation (Local & Random policies) |
| 4 | Implement Greedy policy, run baseline experiments |
| 5 | Export stats, adapt & retrain DQN (6-action) in notebook |
| 6 | Implement RL policy in OMNeT++, run experiments |
| 7 | Collect & analyze results, generate plots |
| 8 | Write report, prepare presentation |

---

## 15. Tools & Environment

- **OMNeT++** (version ≥ 6.0)
- **Python 3.10+**, PyTorch, NumPy, Pandas, Matplotlib
- **Jupyter Notebook** for RL training (`Shortest_path.ipynb`)
- Version control: Git

---

## 16. Risks & Assumptions

- Assumes static network topology (no mobility).
- Assumes task characteristics are independent across devices (no correlated bursts), unless deliberately modeled.
- RL training may require tuning (reward shaping, episode count) beyond values used in the original notebook, given the increased action space (2 → 6).
- OMNeT++ ↔ Python integration (Option A) assumes offline statistics are representative enough for policy training; if not, live co-simulation (Option B) may be required as a fallback.

---

## 17. Success Criteria

The project is considered successful if:
- The simulation runs end-to-end for all 2000 tasks across 15 IoT devices and 5 fog servers.
- All four policies (Local, Random, Greedy, RL) produce comparable, reproducible metrics.
- The RL-based policy demonstrates **measurable improvement** (lower average delay and/or higher hit ratio) over at least the Local and Random baselines, and ideally is competitive with or better than Greedy.
- Results are clearly visualized and explainable to the supervisor.