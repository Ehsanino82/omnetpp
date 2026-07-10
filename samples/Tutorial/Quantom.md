Here is the PDF content converted to Markdown format:

---

# Intelligent Task Offloading in Heterogeneous GPU–Quantum Computing Systems Using Quantum Suitability Detection and Execution Time Prediction

**Pouria**

*June 26, 2026*

## Abstract

Future computing infrastructures will integrate Quantum Processing Units (QPUs) alongside classical accelerators like GPUs. However, QPUs are beneficial only for specific problem classes, making task-to-processor assignment a critical challenge. This project proposes an intelligent scheduler that automatically determines whether a task is quantum-suitable, predicts execution and waiting times on both GPU and QPU, and offloads tasks to the processor expected to complete them earliest. The framework integrates a quantum suitability classifier, separate execution-time predictors, and a completion-time-aware scheduling policy to minimize overall makespan in hybrid heterogeneous systems.

## Contents

1. Project Description — 3
   - 1.1 Project Title — 3
   - 1.2 Motivation — 3
2. Objectives — 3
3. System Architecture — 3
4. Phase 1: Incoming Tasks — 4
5. Phase 2: Supported Task Types — 5
   - 5.1 Quantum-Optimizable Tasks — 5
   - 5.2 GPU-Oriented Tasks — 5
6. Phase 3: Feature Extraction — 6
7. Phase 4: Quantum Suitability Detection — 6
8. Phase 5: GPU Execution Time Prediction — 7
9. Phase 6: QPU Execution Time Prediction — 7
10. Phase 7: Queue Waiting Time — 8
11. Phase 8: Completion Time Prediction — 8
12. Phase 9: Intelligent Offloading Decision — 8
13. Machine Learning Components — 8
14. Evaluation Metrics — 9
15. Expected Contributions — 9

---

## 1 Project Description

### 1.1 Project Title

**Intelligent Task Offloading in Heterogeneous GPU–Quantum Computing Systems Using Quantum Suitability Detection and Execution Time Prediction**

### 1.2 Motivation

Future computing infrastructures will not rely solely on CPUs and GPUs. Quantum Processing Units (QPUs) are becoming available through cloud platforms and hybrid architectures, where quantum processors work alongside classical accelerators. However, unlike GPUs, QPUs only provide advantages for specific classes of computational problems, while many conventional workloads still execute more efficiently on GPUs.

A major challenge is therefore deciding which processor should execute each incoming task. Sending unsuitable tasks to a QPU wastes scarce quantum resources, whereas sending quantum-friendly optimization problems to a GPU may lead to longer execution times. Existing schedulers typically assume that the execution platform has already been selected and do not jointly consider quantum suitability, execution time prediction, and queueing delay.

This project proposes an intelligent scheduler that automatically determines whether a task is suitable for quantum execution and dynamically assigns it to either a GPU or a QPU to minimize overall completion time.

---

## 2 Objectives

The proposed scheduler has four primary objectives:

1. Analyze incoming tasks and extract computational characteristics.
2. Determine whether a task is quantum-optimizable.
3. Predict execution and waiting times on both GPU and QPU.
4. Offload the task to the processor expected to complete it earliest.

---

## 3 System Architecture

The overall architecture of the proposed scheduler is illustrated below:

```
                Incoming Tasks
                      |
                      v
              Task Feature Extractor
                      |
                      v
           Quantum Suitability Detector
                      |
          +-----------+-----------+
          |                       |
  Not Quantum-Friendly     Quantum-Friendly
          |                       |
          v                       v
   GPU Candidate           Estimate Both GPU & QPU
                                   |
                                   v
                       Execution Time Prediction
                      +-----------+-----------+
                      |                       |
                      v                       v
              GPU Completion           QPU Completion
                      |                       |
                      +-----------+-----------+
                                  |
                                  v
                       Intelligent Scheduler
                                  |
                      +-----------+-----------+
                      |                       |
                      v                       v
                  GPU Queue               QPU Queue
```

---

## 4 Phase 1: Incoming Tasks

Every arriving task is represented as

$$T_i = (ID, Type, InputSize, Priority, Deadline)$$

Each task also contains computational metadata:

- Task category
- Number of variables
- Number of constraints
- Memory requirement
- FLOPs estimate
- Data size
- Graph size
- Search space size

---

## 5 Phase 2: Supported Task Types

### 5.1 Quantum-Optimizable Tasks

These problems have known quantum algorithms or are naturally formulated for quantum optimization.

**Table 1: Quantum-Optimizable Tasks and Their Suitability**

| Task | Quantum Algorithm | Suitability |
|---|---|---|
| Integer Factorization | Shor's Algorithm | Very High |
| Unstructured Search | Grover's Algorithm | High |
| Traveling Salesman Problem | QAOA | High |
| Vehicle Routing Problem | QAOA | High |
| Max-Cut | QAOA | High |
| Graph Coloring | QAOA | High |
| Knapsack Problem | QAOA | High |
| SAT / Max-SAT | QAOA | High |
| Portfolio Optimization | Quantum Annealing/QAOA | High |
| Job-Shop Scheduling | Quantum Annealing | High |
| Molecular Simulation | VQE | Very High |
| Quantum Chemistry | VQE | Very High |
| Monte Carlo Simulation | Quantum Amplitude Estimation | Medium–High |
| Sparse Linear Systems | HHL | Medium |
| Quantum Machine Learning Optimization | QML | Medium |

### 5.2 GPU-Oriented Tasks

**Table 2: GPU-Oriented Tasks and Their Preferred Processor**

| Task | Preferred Processor |
|---|---|
| Image Classification | GPU |
| Object Detection | GPU |
| CNN Training | GPU |
| Transformer Inference | GPU |
| Video Processing | GPU |
| Rendering | GPU |
| Ray Tracing | GPU |
| Matrix Multiplication | GPU |
| FFT | GPU |
| Signal Processing | GPU |
| Image Filtering | GPU |
| Physics Simulation | GPU |
| Fluid Simulation | GPU |
| Deep Learning Training | GPU |

---

## 6 Phase 3: Feature Extraction

Instead of relying only on task names, each task is converted into a feature vector

$$X = [x_1, x_2, \ldots, x_n]$$

Example features include:

- Problem category
- Optimization problem
- NP-hard
- Search problem
- Graph-based
- Number of variables
- Number of constraints
- Search-space size
- FLOPs
- Memory requirement
- Data transfer size
- QUBO formulation possible
- Circuit depth estimate
- Estimated qubits

---

## 7 Phase 4: Quantum Suitability Detection

The scheduler predicts a quantum suitability score

$$Q_s \in [0, 1]$$

**Table 3: Example Quantum Suitability Scores**

| Task | $Q_s$ |
|---|---|
| TSP | 0.97 |
| Knapsack | 0.93 |
| Max-Cut | 0.95 |
| Molecular Simulation | 0.99 |
| CNN Training | 0.08 |
| Video Rendering | 0.02 |

**Decision rule**

$$Q_s > \theta$$

where

$$\theta = 0.7$$

Only tasks above the threshold are considered for QPU execution.

---

## 8 Phase 5: GPU Execution Time Prediction

The GPU execution time is estimated as

$$E_{GPU} = T_{copy} + T_{kernel} + T_{copyback}$$

or

$$E_{GPU} = \frac{F}{P_{GPU} \times \eta}$$

where

- $F$ = required floating-point operations
- $P_{GPU}$ = GPU throughput
- $\eta$ = GPU efficiency

---

## 9 Phase 6: QPU Execution Time Prediction

The QPU execution time is modeled as

$$E_{QPU} = T_{encoding} + T_{compile} + T_{execution} + T_{measurement} + T_{decode}$$

For variational algorithms

$$E_{QPU} = Iterations \times CircuitTime$$

Circuit execution depends on

- Number of qubits
- Circuit depth
- Number of gates
- Number of shots
- Gate latency

---

## 10 Phase 7: Queue Waiting Time

GPU waiting time

$$W_{GPU} = \sum_{k=1}^{m} E_k$$

QPU waiting time

$$W_{QPU} = \sum_{k=1}^{n} E_k$$

---

## 11 Phase 8: Completion Time Prediction

GPU completion time

$$C_{GPU} = W_{GPU} + E_{GPU}$$

QPU completion time

$$C_{QPU} = W_{QPU} + E_{QPU}$$

---

## 12 Phase 9: Intelligent Offloading Decision

The scheduler follows the algorithm below.

1. Extract task features.
2. Compute the quantum suitability score $Q_s$.
3. If $Q_s < \theta$, send the task directly to the GPU.
4. Otherwise estimate $C_{GPU}$ and $C_{QPU}$.
5. Select the processor with the smallest completion time.

---

## 13 Machine Learning Components

Three prediction models are employed.

**Table 4: Machine Learning Models for the Scheduler**

| Model | Input | Output |
|---|---|---|
| Quantum Suitability Classifier | Task Features | $Q_s$ |
| GPU Time Predictor | Task + GPU State | $E_{GPU}$ |
| QPU Time Predictor | Task + QPU State | $E_{QPU}$ |

Possible machine learning algorithms include:

- Decision Trees
- Random Forest
- XGBoost
- LightGBM
- Neural Networks

---

## 14 Evaluation Metrics

The proposed scheduler can be evaluated using:

- Average completion time (Makespan)
- Average waiting time
- Throughput
- GPU utilization
- QPU utilization
- Load balancing
- Scheduling overhead
- Deadline satisfaction
- Energy consumption (optional)

---

## 15 Expected Contributions

The proposed framework contributes:

1. A task taxonomy distinguishing quantum-optimizable and GPU-oriented workloads.
2. A quantum suitability classifier for identifying tasks that can benefit from quantum execution.
3. Separate execution-time prediction models for GPU and QPU.
4. A completion-time-aware offloading strategy that jointly considers execution time and queue waiting time.
5. A hybrid heterogeneous scheduling framework for future GPU–QPU computing platforms.

Unlike traditional schedulers, the proposed framework integrates task classification, execution-time prediction, and intelligent offloading into a unified decision-making architecture for heterogeneous quantum-classical computing systems.

---

## Acknowledgments

The author gratefully acknowledges the support and guidance received from the research group at Tecnun, University of Navarra, and the facilities provided through the IBM Quantum System II.

---

## References

1. IBM Quantum. (2024). IBM Quantum System II Documentation. IBM Quantum.

2. Farhi, E., Goldstone, J., & Gutmann, S. (2014). A quantum approximate optimization algorithm. arXiv preprint arXiv:1411.4028.

3. Grover, L. K. (1996). A fast quantum mechanical algorithm for database search. Proceedings of the 28th Annual ACM Symposium on Theory of Computing, 212–219.

4. Shor, P. W. (1997). Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer. SIAM Journal on Computing, 26(5), 1484–1509.

5. Peruzzo, A., McClean, J., Shadbolt, P., et al. (2014). A variational eigenvalue solver on a photonic quantum processor. Nature Communications, 5, 4213.

6. Huang, A., et al. (2021). Quantum computing for optimization: A survey. ACM Computing Surveys, 54(5), 1–35.