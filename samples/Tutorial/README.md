# RL-Based Task Offloading in Fog Computing

An OMNeT++ discrete-event simulation of **15 IoT devices** offloading computation tasks to **5 fog servers**, where each IoT device uses a **Deep Q-Network (DQN)** to learn the optimal offloading policy at runtime.

---

## Table of Contents

- [Motivation](#motivation)
- [System Model](#system-model)
- [RL Formulation](#rl-formulation)
- [Project Structure](#project-structure)
- [Architecture Diagram](#architecture-diagram)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Results & Analysis](#results--analysis)
- [Baseline Policies](#baseline-policies)
- [Future Work](#future-work)
- [References](#references)

---

## Motivation

In fog computing, IoT devices generate tasks that must be processed within strict deadlines. Each device faces a decision: **process locally** (slower CPU, no network delay) or **offload to a fog server** (faster CPU, but network latency and potential queuing delay). Choosing the wrong server — or offloading when servers are overloaded — leads to deadline violations and degraded quality of service.

Traditional approaches use static rules (round-robin, greedy) which cannot adapt to changing load conditions. This project uses **Deep Reinforcement Learning** to train each IoT device to make intelligent, load-aware offloading decisions that minimize task completion delay and maximize deadline hit ratio.

---

## System Model

### Network Topology

```
   IoT[0]  IoT[1]  ...  IoT[14]
     │╲      │╲           │╲
     │ ╲     │ ╲          │ ╲         (fully connected,
     │  ╲    │  ╲         │  ╲         10ms link delay)
     ▼   ▼   ▼   ▼       ▼   ▼
   Fog[0] Fog[1] Fog[2] Fog[3] Fog[4]
```

- **15 IoT devices**: Generate tasks from `tasks.csv` at specified arrival times. Each device has a local CPU (slower) and can offload to any fog server.
- **5 Fog servers**: Process offloaded tasks using a FIFO single-server queue model. Broadcast their current queue load to all IoT devices every 100ms.
- **Links**: Fully connected mesh with 10ms propagation delay per hop.

### Task Model

Each task has the following attributes:

| Field | Description |
|-------|-------------|
| `taskId` | Unique task identifier |
| `ownerDeviceId` | IoT device that generates this task (0–14) |
| `arrivalTime` | Simulation time when the task is released (seconds) |
| `cpuDemand` | CPU processing time required (seconds) |
| `deadline` | Absolute deadline by which the task must complete (seconds) |
| `sizeBytes` | Task data size in bytes |

### Processing Model

- **Local**: `processingTime = cpuDemand / localCpuRate` (default `localCpuRate = 0.5`, i.e., local CPU is half the speed of fog)
- **Fog**: `processingTime = cpuDemand` (fog CPU rate = 1.0), plus round-trip network delay (2 × 10ms), plus queuing wait if the server is busy
- Both local and fog use a **FIFO single-server queue**: `start = max(now, nextFreeTime)`, `finish = start + processingTime`

### Fog Load Feedback

Each fog server periodically broadcasts a `ServerStatus` message to all IoT devices containing its **estimated remaining queue work** (in seconds). IoT devices use this to build their state vector for the DQN.

---

## RL Formulation

### State Space (7 dimensions)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `cpuDemand` | CPU demand of the current task (seconds) |
| 1 | `deadlineSlack` | Time remaining until deadline: `deadline - now` (seconds) |
| 2–6 | `fogLoad[0..4]` | Estimated queue backlog of each fog server (seconds) |

### Action Space (6 discrete actions)

| Action | Meaning |
|--------|---------|
| 0 | Process task locally |
| 1 | Offload to Fog Server 0 |
| 2 | Offload to Fog Server 1 |
| 3 | Offload to Fog Server 2 |
| 4 | Offload to Fog Server 3 |
| 5 | Offload to Fog Server 4 |

### Reward Function

```
reward = -latency × 10
if latency > deadline_slack:
    reward -= 20    (deadline violation penalty)
```

### DQN Architecture

A 3-layer fully connected neural network:

```
Input (7) → Dense(128) → ReLU → Dense(128) → ReLU → Dense(6) → Q-values
```

Training uses:
- **Experience replay** buffer (20,000 transitions)
- **Target network** updated every 10 episodes
- **ε-greedy** exploration decaying from 1.0 to 0.05
- **Gradient clipping** at norm 1.0

### Offline Training → Online Inference

The DQN is trained **offline** in Python (`train_dqn.py`) using a simplified simulator. The trained weights are exported as CSV files. The OMNeT++ C++ simulation loads these weights at initialization and performs **pure forward inference** (no learning during simulation). This keeps the C++ side simple with zero Python/PyTorch dependency.

---

## Project Structure

```
Tutorial/
├── README.md                  # This file
├── RUNBOOK.md                 # Step-by-step execution guide
├── Makefile                   # Top-level build
├── src/
│   ├── IoTDevice.cc           # IoT device: RL decision, local processing, task dispatch
│   ├── FogServer.cc           # Fog server: FIFO queue, load broadcast, ack
│   ├── DqnPolicy.h            # Pure C++ MLP inference (reads CSV weights)
│   ├── Task.msg               # Task message definition
│   ├── ServerStatus.msg       # Fog load feedback message
│   ├── Task_m.cc/h            # (auto-generated from Task.msg)
│   ├── ServerStatus_m.cc/h    # (auto-generated from ServerStatus.msg)
│   ├── package.ned            # NED package declaration
│   └── Makefile               # Source-level build (generated by opp_makemake)
├── simulations/
│   ├── FogNetwork.ned         # Network topology: 15 IoT + 5 Fog, fully connected
│   ├── omnetpp.ini            # Configs: Local, Random, RoundRobin, Greedy, DQN
│   ├── generate_tasks.py      # Task generator (2000 tasks, seed 42)
│   ├── tasks.csv              # 2000 tasks (id, device, arrival, type, size, cpu, deadline, ...)
│   ├── train_dqn.py           # Python DQN trainer (pure NumPy, no PyTorch)
│   ├── dqn_weights/           # Exported model weights (after training)
│   │   ├── w1.csv, b1.csv     #   Layer 1 weights and biases
│   │   ├── w2.csv, b2.csv     #   Layer 2 weights and biases
│   │   ├── w3.csv, b3.csv     #   Layer 3 weights and biases
│   │   └── scaler.csv         #   Input normalization (mean, std)
│   ├── results/               # Simulation output (scalars, vectors)
│   └── package.ned            # NED package declaration
└── out/                       # Build artifacts
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE (Python)                         │
│                                                                 │
│   train_dqn.py                                                  │
│   ┌───────────────┐    ┌──────────────┐    ┌────────────────┐   │
│   │ FogOffloadEnv │───▶│  DQN Agent   │───▶│ Export Weights │   │
│   │ (simplified)  │◀───│ (ε-greedy)   │    │ to CSV files   │   │
│   └───────────────┘    └──────────────┘    └───────┬────────┘   │
│                                                    │            │
└────────────────────────────────────────────────────┼────────────┘
                                                     │
                              w1.csv, b1.csv, ...    │
                              scaler.csv             │
                                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ONLINE (OMNeT++ C++)                       │
│                                                                 │
│   ┌──────────────┐  ServerStatus   ┌──────────────┐             │
│   │  IoTDevice   │◀───────────────│  FogServer   │             │
│   │              │                 │              │             │
│   │ ┌──────────┐ │     Task        │ ┌──────────┐ │             │
│   │ │DqnPolicy │ │────────────────▶│ │FIFO Queue│ │             │
│   │ │(forward  │ │                 │ │          │ │             │
│   │ │ pass)    │ │      Ack        │ │          │ │             │
│   │ └──────────┘ │◀───────────────│ └──────────┘ │             │
│   │              │                 │              │             │
│   │ Local Queue  │                 │  Broadcasts  │             │
│   │ (fallback)   │                 │  load every  │             │
│   └──────────────┘                 │  100ms       │             │
│                                    └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

- **OMNeT++ 6.4.0** (already installed in this workspace)
- **Python 3.8+** with **NumPy** (no PyTorch required)
- **C++17** compiler (clang or g++)

---

## Quick Start

```bash
# 1. Source OMNeT++ environment
cd /path/to/omnetpp-6.4.0
source setenv

# 2. Build the simulation
cd samples/Tutorial
make makefiles
make

# 3. Train the DQN and export weights
cd simulations
python3 train_dqn.py

# 4. Run with round-robin baseline
../src/Tutorial -u Cmdenv -c General

# 5. Run with trained DQN policy
../src/Tutorial -u Cmdenv -c DQN

# 6. (Optional) Run with GUI
../src/Tutorial -c DQN
```

See [RUNBOOK.md](RUNBOOK.md) for detailed step-by-step instructions and troubleshooting.

---

## Configuration

### `omnetpp.ini`

| Config | Description |
|--------|-------------|
| `[General]` | Round-robin baseline — always offloads, cycles through fog servers |
| `[Config Local]` | Baseline — always process locally |
| `[Config Random]` | Baseline — random choice over local + 5 fog servers |
| `[Config RoundRobin]` | Baseline — always offload, cycle through fog servers |
| `[Config Greedy]` | Baseline — least estimated completion time |
| `[Config DQN]` | RL — trained DQN policy from `dqn_weights/` |

### Key Parameters

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `numIoT` | `omnetpp.ini` | 15 | Number of IoT devices |
| `numFog` | `omnetpp.ini` | 5 | Number of fog servers |
| `sim-time-limit` | `omnetpp.ini` | 100s | Simulation duration |
| `statusInterval` | `FogNetwork.ned` | 0.1s | Fog load broadcast interval |
| `localCpuRate` | `FogNetwork.ned` | 0.5 | Local CPU speed (1.0 = fog speed) |
| `decisionMode` | `FogNetwork.ned` | "roundRobin" | Policy: `local`\|`random`\|`roundRobin`\|`greedy`\|`dqn` |
| `modelDir` | `FogNetwork.ned` | "" | Path to DQN weight CSVs |

### Training Hyperparameters (`train_dqn.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EPISODES` | 300 | Number of training episodes |
| `NUM_TASKS_PER_EP` | 200 | Tasks generated per episode |
| `HIDDEN_DIM` | 128 | MLP hidden layer width |
| `LR` | 0.001 | Learning rate |
| `GAMMA` | 0.99 | Discount factor |
| `EPSILON_DECAY` | 0.995 | Exploration decay rate |
| `BUFFER_SIZE` | 20000 | Replay buffer capacity |
| `BATCH_SIZE` | 64 | Training batch size |

---

## Results & Analysis

After running both configurations, results are stored in `simulations/results/`. Key metrics:

| Metric | Signal | Recorded At |
|--------|--------|-------------|
| Messages sent per IoT | `msgSent` | IoTDevice |
| Messages received per fog | `msgReceived` | FogServer |
| Response time per task | `respTime` | FogServer |

### Extracting Results

```bash
# Export scalar results
scavetool export results/General-*.sca -o baseline_scalars.csv
scavetool export results/DQN-*.sca -o dqn_scalars.csv

# Export vector results (time series)
scavetool export results/General-*.vec -o baseline_vectors.csv
scavetool export results/DQN-*.vec -o dqn_vectors.csv
```

### Expected Observations

- **Round-Robin**: Distributes load evenly but ignores server busyness; can cause queuing delays under bursty workloads
- **DQN**: Learns to route tasks to less-loaded servers and occasionally process locally when all servers are busy; should achieve lower average response time and fewer deadline violations

---

## Baseline Policies

The Python trainer evaluates these baselines during training:

| Policy | Strategy | Typical Hit Ratio |
|--------|----------|-------------------|
| **Local** | Always process locally | Very low (~2%) — local CPU is too slow |
| **Random** | Uniform random choice among all 6 actions | ~92% |
| **Round-Robin** | Cycle through fog servers sequentially | ~98% |
| **Greedy** | Pick the fog server with lowest current load | ~98% |
| **DQN** | Learned policy based on state observation | ~96–100% |

The DQN's advantage is most visible under **high load** or **variable workloads** where static policies cannot adapt.

---

## Future Work

### Short-Term Improvements

1. **Online Learning in OMNeT++**
   Instead of training offline and loading fixed weights, implement online DQN training inside the C++ simulation. The IoT devices would learn and adapt during the simulation itself, using actual response times as reward feedback. This requires integrating experience replay and gradient updates in C++ or bridging to Python via sockets/shared memory.

2. **Energy-Aware Reward**
   Extend the reward function to include energy consumption: local processing uses more device battery, while offloading uses network energy. A weighted reward `R = -α·latency - β·energy` would allow tuning the latency-energy trade-off.

3. **Per-Device Heterogeneous Policies**
   Currently all IoT devices share the same DQN weights. In reality, devices differ in CPU capability, battery level, and workload patterns. Train per-device or per-class policies, or add device-specific features (battery level, local queue length) to the state vector.

4. **Dynamic Task Generation**
   Replace the static `tasks.csv` with stochastic task generation (Poisson arrivals, variable CPU demands) to test the policy's robustness under unpredictable workloads.

5. **Local Queue Load as State Feature**
   Add an 8th state dimension for the IoT device's own local queue backlog. This helps the DQN make better local-vs-offload decisions when tasks arrive in bursts.

### Medium-Term Extensions

6. **Multi-Hop Fog Routing**
   Extend the network to include intermediate relay nodes or hierarchical fog tiers (edge → fog → cloud). The DQN would learn routing paths, not just single-hop offloading decisions. This connects to the TETRIS routing approach in the `Shortest_path.ipynb` notebook.

7. **Multi-Agent Reinforcement Learning (MARL)**
   Instead of independent per-device policies, use cooperative MARL where IoT devices share information or coordinate decisions. Approaches include centralized training with decentralized execution (CTDE), or communication-based methods where devices exchange intent signals.

8. **Prioritized Experience Replay**
   Replace uniform replay sampling with prioritized experience replay (PER), where transitions with higher TD-error are sampled more often. This typically accelerates learning and improves final policy quality.

9. **Dueling DQN / Double DQN**
   Upgrade the DQN architecture to Dueling DQN (separate value and advantage streams) or Double DQN (decouple action selection and evaluation) for more stable and accurate Q-value estimation.

10. **Realistic Network Model**
    Add variable link delays (based on congestion), packet-level simulation, bandwidth constraints, and link failures. Use OMNeT++'s INET framework for TCP/IP-level modeling.

### Long-Term Research Directions

11. **Transfer Learning Across Topologies**
    Train on one network topology and test on different configurations (more/fewer fog servers, different connectivity). Investigate how well the learned policy generalizes or whether fine-tuning is needed.

12. **Federated Learning for Privacy**
    Each IoT device trains its own model on local task data, then periodically aggregates weights with other devices (federated averaging). This preserves data privacy while enabling collaborative learning.

13. **Integration with Real Fog Platforms**
    Export the trained policy to real edge/fog platforms (Raspberry Pi clusters, AWS Greengrass, Azure IoT Edge) and validate simulation results against real-world performance.

14. **Comparison with Optimization-Based Methods**
    Benchmark the DQN approach against classical optimization methods (integer linear programming, Hungarian algorithm, convex optimization) to quantify the RL advantage and identify scenarios where each approach excels.

15. **Curriculum Learning**
    Start training with simple scenarios (few tasks, light load) and progressively increase difficulty (more tasks, tighter deadlines, higher contention). This can lead to faster convergence and better final policies.

---

## References

- **DQN**: Mnih, V., et al. "Human-level control through deep reinforcement learning." *Nature* 518.7540 (2015): 529-533.
- **TETRIS**: Inspired by the joint task routing and offloading approach in `Shortest_path.ipynb`.
- **OMNeT++**: Varga, A. "OMNeT++." *Modeling and Tools for Network Simulation*. Springer, 2010.
- **Fog Computing Offloading**: Mao, Y., et al. "A survey on mobile edge computing." *IEEE Communications Surveys & Tutorials* 19.4 (2017): 2322-2358.

---

## License

This project is licensed under the GNU Lesser General Public License v3.0. See the source file headers for details.
