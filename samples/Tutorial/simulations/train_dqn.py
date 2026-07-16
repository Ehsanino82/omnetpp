#!/usr/bin/env python3
"""
train_dqn.py — Train a DQN for IoT task offloading onto a HETEROGENEOUS fog
(CPU/GPU/QPU), mirroring the OMNeT++ ProcessorModel.h exactly.

Teacher feedback:
  * fog servers are heterogeneous: 2x QPU (IBM, IonQ), 2x GPU (RTX 3090), 1x CPU.
  * selection must be intelligent: estimate completion time = network + queue +
    execution time, where execution time follows Quantom.md formulas (CPU/GPU/QPU).
  * DRL: 1000 tasks/episode, >= 3000 episodes.
  * 5G network delay random 1-10 ms.

Actions: 0 = local, 1..5 = fog server i-1.
State (18-dim): [workload, Qs, deadlineSlack,
                 netDelay0..4, fogBacklog0..4, execEst0..4]
Exports weight matrices + scaler to CSV for C++ inference (src/DqnPolicy.h).
"""

import numpy as np
import os
import csv
import random
import argparse
import math
from collections import deque

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ========== Configuration ==========
NUM_FOG = 5
NUM_ACTIONS = NUM_FOG + 1   # 0=local, 1..5=fog servers
STATE_DIM = 3 + NUM_FOG + 2 * NUM_FOG   # W, Qs, slack, net(5), backlog(5), exec(5) = 18

EPISODES = 3000
NUM_TASKS_PER_EP = 1000
BATCH_SIZE = 64
BUFFER_SIZE = 50000
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.997
TARGET_UPDATE = 10        # episodes
HIDDEN_DIM = 64
TRAIN_EVERY = 4           # gradient update every N env steps (standard DQN)

# ========== Processor model (mirrors src/ProcessorModel.h) ==========
PROC_CPU, PROC_GPU, PROC_QPU = 0, 1, 2
QPU_THETA = 0.7


def compute_exec_time(spec, workload, qs, data_bytes):
    eps = 1e-9
    t = spec["type"]
    if t == PROC_GPU:
        copy = (2.0 * data_bytes / spec["gpuBandwidth"]) if spec["gpuBandwidth"] > eps else 0.0
        kernel = workload / (spec["throughput"] * spec["efficiency"]) if spec["throughput"] * spec["efficiency"] > eps else 1e6
        return spec["gpuLaunchOverhead"] + copy + kernel
    if t == PROC_QPU:
        if qs >= QPU_THETA:
            eff = spec["throughput"] * qs * qs + eps
            return spec["qpuOverhead"] + workload / eff
        # non-quantum task: QPU falls back to its slow classical host (finite)
        return spec["qpuOverhead"] + workload / 50.0
    return workload / spec["throughput"] if spec["throughput"] > eps else 1e6
    return workload / spec["throughput"] if spec["throughput"] > eps else 1e6


# Heterogeneous fog specs — MUST match simulations/omnetpp.ini
FOG_SPECS = [
    {"type": PROC_QPU, "throughput": 200000, "power": 15000, "qpuOverhead": 0.0015},   # IBM
    {"type": PROC_QPU, "throughput": 200000, "power": 12000, "qpuOverhead": 0.0020},   # IonQ
    {"type": PROC_GPU, "throughput": 35000,  "power": 350,   "efficiency": 0.55,
     "gpuBandwidth": 936e6, "gpuLaunchOverhead": 0.0001},                              # RTX3090
    {"type": PROC_GPU, "throughput": 35000,  "power": 350,   "efficiency": 0.55,
     "gpuBandwidth": 936e6, "gpuLaunchOverhead": 0.0001},                              # RTX3090b
    {"type": PROC_CPU, "throughput": 1000,   "power": 65},                              # CPU-11core
]

# IoT local model — MUST match omnetpp.ini
LOCAL_THROUGHPUT = 300.0      # MFLOP/s
LOCAL_POWER = 2.0             # W
TX_POWER = 1.5                # W
UPLINK_RATE = 100e6           # bps

# Task-type profiles (mirrors generate_tasks.py)
TASK_TYPES = {
    "tsp":              (0.97, (10, 60),  (10, 40)),
    "knapsack":         (0.93, (8, 50),   (10, 40)),
    "maxcut":           (0.95, (10, 60),  (10, 40)),
    "portfolio":        (0.90, (8, 45),   (10, 40)),
    "molecular_sim":    (0.99, (5, 30),   (15, 50)),
    "image_processing": (0.05, (20, 80),  (8, 30)),
    "cnn_inference":    (0.08, (30, 80),  (8, 30)),
    "video_processing": (0.02, (25, 80),  (10, 35)),
    "sensor_data":      (0.15, (2, 15),   (5, 25)),
    "control_signal":   (0.10, (1, 8),    (4, 15)),
    "aggregation":      (0.35, (5, 25),   (8, 30)),
    "alert":            (0.10, (1, 6),    (4, 12)),
}
TYPE_NAMES = list(TASK_TYPES.keys())
TYPE_WEIGHTS = [0.10, 0.08, 0.08, 0.07, 0.07,
                0.12, 0.10, 0.08,
                0.10, 0.08, 0.07, 0.05]


def gen_task(rng):
    ttype = rng.choice(TYPE_NAMES, p=TYPE_WEIGHTS)
    qs, wrange, srange = TASK_TYPES[ttype]
    qs = float(np.clip(qs + rng.normal(0, 0.02), 0.0, 1.0))
    workload = float(rng.uniform(*wrange))
    slack = float(rng.uniform(*srange)) / 1000.0          # -> seconds
    data_kb = float(rng.uniform(10.0, 500.0))
    return workload, qs, slack, data_kb


def tx_energy(data_kb):
    bits = data_kb * 1024.0 * 8.0
    return TX_POWER * bits / UPLINK_RATE


# ========== Environment ==========
class FogOffloadEnv:
    def __init__(self, num_tasks=NUM_TASKS_PER_EP, seed=None):
        self.num_tasks = num_tasks
        self.rng = np.random.default_rng(seed) if seed is not None else np.random
        self.reset()

    def reset(self):
        self.fog_free_at = np.zeros(NUM_FOG)
        self.local_free_at = 0.0
        self.now = 0.0
        self.task_id = 0
        self.total_delay = 0.0
        self.total_energy = 0.0
        self.dropped = 0
        self.processed = 0
        self.local_count = 0
        self.offload_count = 0
        self._gen_task()
        return self._get_obs()

    def _gen_task(self):
        self.workload, self.qs, self.deadline_slack, self.data_kb = gen_task(self.rng)
        self.data_bytes = self.data_kb * 1024.0
        # 5G network delay per fog, random 1-10 ms (teacher feedback)
        self.net_delays = self.rng.uniform(0.001, 0.010, size=NUM_FOG)

    def _exec_est(self, j):
        return compute_exec_time(FOG_SPECS[j], self.workload, self.qs, self.data_bytes)

    def allowed_actions(self):
        """Quantum-suitability mask (Quantom.md phase 4): a non-quantum task
        (Qs < theta) must not be sent to a QPU. Returns the feasible action set."""
        allowed = [0]  # local always allowed
        for j in range(NUM_FOG):
            is_qpu = (FOG_SPECS[j]["type"] == PROC_QPU)
            if is_qpu and self.qs < QPU_THETA:
                continue
            allowed.append(j + 1)
        return allowed

    def _get_obs(self):
        backlog = np.maximum(0.0, self.fog_free_at - self.now)
        execs = np.array([self._exec_est(j) for j in range(NUM_FOG)])
        return np.array([
            self.workload,
            self.qs,
            self.deadline_slack,
            *self.net_delays,
            *backlog,
            *execs,
        ], dtype=np.float32)

    def step(self, action):
        if self.task_id >= self.num_tasks:
            return self._get_obs(), 0.0, True, {}

        if action == 0:
            proc = self.workload / LOCAL_THROUGHPUT
            start = max(self.now, self.local_free_at)
            finish = start + proc
            self.local_free_at = finish
            latency = finish - self.now
            energy = LOCAL_POWER * proc
            self.local_count += 1
        else:
            j = action - 1
            exec_t = self._exec_est(j)
            net = self.net_delays[j]
            start = max(self.now + net, self.fog_free_at[j])
            finish = start + exec_t
            self.fog_free_at[j] = finish
            latency = (finish - self.now) + net          # + ack return delay
            energy = tx_energy(self.data_kb) + FOG_SPECS[j]["power"] * exec_t
            self.offload_count += 1

        self.total_delay += latency
        self.total_energy += energy
        self.processed += 1
        self.task_id += 1

        # Reward: latency-dominant, strong deadline-miss penalty, small energy
        # term. Shaped so the agent prioritizes meeting deadlines (matching the
        # completion-time-minimizing greedy policy) while still factoring energy.
        reward = -latency * 200.0 - 0.005 * energy
        if latency > self.deadline_slack:
            self.dropped += 1
            reward -= 80.0

        self.now += float(self.rng.uniform(0.01, 0.05))
        done = self.task_id >= self.num_tasks
        if not done:
            self._gen_task()
        return self._get_obs(), reward, done, {}

    def metrics(self):
        p = max(1, self.processed)
        return {
            "avg_delay": self.total_delay / p,
            "hit_ratio": 1.0 - self.dropped / p,
            "avg_energy": self.total_energy / p,
            "local_ratio": self.local_count / p,
            "offload_ratio": self.offload_count / p,
        }


# ========== NumPy MLP DQN ==========
class NumpyMLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        scale3 = np.sqrt(2.0 / hidden_dim)
        self.W1 = np.random.randn(hidden_dim, input_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = np.random.randn(output_dim, hidden_dim).astype(np.float32) * scale3
        self.b3 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x):
        h1 = np.maximum(0, x @ self.W1.T + self.b1)
        h2 = np.maximum(0, h1 @ self.W2.T + self.b2)
        return h2 @ self.W3.T + self.b3

    def predict(self, x):
        q = self.forward(x.reshape(1, -1))
        return int(np.argmax(q))

    def predict_masked(self, x, allowed):
        q = self.forward(x.reshape(1, -1))[0]
        best = allowed[0]
        bestQ = q[best]
        for a in allowed:
            if q[a] > bestQ:
                bestQ = q[a]
                best = a
        return best

    def copy_from(self, other):
        for n in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(self, n, getattr(other, n).copy())

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2, self.W3, self.b3 = [p.copy() for p in params]


class ReplayBuffer:
    def __init__(self, cap=BUFFER_SIZE):
        self.buf = deque(maxlen=cap)

    def add(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, float(d)))

    def sample(self, batch):
        data = random.sample(self.buf, batch)
        s, a, r, ns, d = zip(*data)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


def train_step(policy, target, buf, lr=LR, gamma=GAMMA):
    bs, ba, br, bns, bd = buf.sample(BATCH_SIZE)
    h1_pre = bs @ policy.W1.T + policy.b1
    h1 = np.maximum(0, h1_pre)
    h2_pre = h1 @ policy.W2.T + policy.b2
    h2 = np.maximum(0, h2_pre)
    q_all = h2 @ policy.W3.T + policy.b3

    q_sa = q_all[np.arange(len(ba)), ba]
    nq_all = target.forward(bns)
    target_q = br + gamma * nq_all.max(axis=1) * (1 - bd)

    error = q_sa - target_q
    loss = float(np.mean(error ** 2))

    dq = np.zeros_like(q_all)
    dq[np.arange(len(ba)), ba] = 2 * error / len(ba)

    dW3 = dq.T @ h2
    db3 = dq.sum(axis=0)
    dh2 = dq @ policy.W3
    dh2_pre = dh2 * (h2_pre > 0)
    dW2 = dh2_pre.T @ h1
    db2 = dh2_pre.sum(axis=0)
    dh1 = dh2_pre @ policy.W2
    dh1_pre = dh1 * (h1_pre > 0)
    dW1 = dh1_pre.T @ bs
    db1 = dh1_pre.sum(axis=0)

    for grad in [dW1, db1, dW2, db2, dW3, db3]:
        norm = np.linalg.norm(grad)
        if norm > 1.0:
            grad *= 1.0 / norm

    policy.W1 -= lr * dW1
    policy.b1 -= lr * db1
    policy.W2 -= lr * dW2
    policy.b2 -= lr * db2
    policy.W3 -= lr * dW3
    policy.b3 -= lr * db3
    return loss


def train(episodes=EPISODES, tasks_per_ep=NUM_TASKS_PER_EP):
    buf = ReplayBuffer()
    policy = NumpyMLP(STATE_DIM, HIDDEN_DIM, NUM_ACTIONS)
    target = NumpyMLP(STATE_DIM, HIDDEN_DIM, NUM_ACTIONS)
    target.copy_from(policy)

    epsilon = EPSILON_START
    all_states = []

    print(f"\n{'='*72}")
    print("TRAINING DQN OFFLOADING POLICY (NumPy) — heterogeneous CPU/GPU/QPU fog")
    print(f"{'='*72}")
    print(f"State dim: {STATE_DIM}, Actions: {NUM_ACTIONS}, Hidden: {HIDDEN_DIM}")
    print(f"Episodes: {episodes}, Tasks/ep: {tasks_per_ep}")
    print(f"{'Ep':<7}{'Delay(s)':<11}{'Hit%':<8}{'Energy':<9}{'Local%':<9}"
          f"{'Loss':<12}{'Reward':<10}")
    print("-" * 72)

    best_score = -1e9
    best_params = None

    for ep in range(episodes):
        env = FogOffloadEnv(num_tasks=tasks_per_ep)
        state = env.reset()
        ep_reward = 0.0
        ep_losses = []
        step = 0

        while True:
            all_states.append(state.copy())
            allowed = env.allowed_actions()
            if random.random() < epsilon:
                action = random.choice(allowed)
            else:
                action = policy.predict_masked(state, allowed)
            ns, reward, done, _ = env.step(action)
            buf.add(state, action, reward, ns, done)
            state = ns
            ep_reward += reward
            # Gradient update every TRAIN_EVERY steps (standard DQN cadence;
            # keeps 3000 episodes x 1000 tasks but ~4x fewer gradient steps).
            if (len(buf) > BATCH_SIZE) and (step % TRAIN_EVERY == 0):
                ep_losses.append(train_step(policy, target, buf))
            step += 1
            if done:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        if ep % TARGET_UPDATE == 0:
            target.copy_from(policy)

        m = env.metrics()
        # Combined objective: high hit ratio, low delay
        score = m["hit_ratio"] * 100 - m["avg_delay"] * 50
        if score >= best_score:
            best_score = score
            best_params = [p.copy() for p in policy.get_params()]

        if (ep + 1) % 100 == 0 or ep == 0:
            avg_loss = np.mean(ep_losses) if ep_losses else 0
            print(f"{ep+1:<7}{m['avg_delay']:<11.4f}{m['hit_ratio']*100:<8.1f}"
                  f"{m['avg_energy']:<9.3f}{m['local_ratio']*100:<9.1f}"
                  f"{avg_loss:<12.6f}{ep_reward:<10.1f}")

    if best_params:
        policy.set_params(best_params)
    print(f"\nBest combined score (hit% - 50*delay): {best_score:.2f}")
    return policy, np.array(all_states)


# ========== Export weights to CSV ==========
def export_weights(policy, all_states, output_dir="dqn_weights"):
    os.makedirs(output_dir, exist_ok=True)
    params = policy.get_params()
    names = [("w1", "b1"), ("w2", "b2"), ("w3", "b3")]
    for i, (wname, bname) in enumerate(names):
        W = params[i * 2]
        b = params[i * 2 + 1]
        with open(os.path.join(output_dir, f"{wname}.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            for row in W:
                writer.writerow([f"{v:.8f}" for v in row])
        with open(os.path.join(output_dir, f"{bname}.csv"), "w", newline="") as f:
            for v in b:
                f.write(f"{v:.8f}\n")

    mean = np.mean(all_states, axis=0)
    std = np.std(all_states, axis=0)
    std[std < 1e-8] = 1.0
    with open(os.path.join(output_dir, "scaler.csv"), "w", newline="") as f:
        f.write("mean,std\n")
        for mm, s in zip(mean, std):
            f.write(f"{mm:.8f},{s:.8f}\n")

    print(f"\n✅ Weights exported to: {output_dir}/")
    print(f"   Files: w1.csv, b1.csv, w2.csv, b2.csv, w3.csv, b3.csv, scaler.csv")
    print(f"   State dim: {STATE_DIM} (must match DqnPolicy input in IoTDevice.cc)")


# ========== Evaluate baselines ==========
def evaluate_baselines(trained_policy=None, episodes=30):
    print(f"\n{'='*72}")
    print(f"BASELINE COMPARISON ({episodes} episodes x {NUM_TASKS_PER_EP} tasks)")
    print(f"{'='*72}")
    print(f"{'Policy':<12}{'Hit%':<8}{'Delay(s)':<11}{'Energy':<9}{'Local%':<9}")
    print("-" * 72)

    rr = [0]

    def local_fn(s): return 0
    def random_fn(s): return random.randint(0, NUM_ACTIONS - 1)
    def rr_fn(s):
        rr[0] = (rr[0] % NUM_FOG) + 1
        return rr[0]
    def greedy_fn(s):
        # s = [W, Qs, slack, net(5), backlog(5), exec(5)]
        backlog = s[3 + NUM_FOG: 3 + 2 * NUM_FOG]
        net = s[3: 3 + NUM_FOG]
        execs = s[3 + 2 * NUM_FOG:]
        costs = list(net + backlog + execs)
        costs.insert(0, max(0.0, 0.0) + s[0] / LOCAL_THROUGHPUT)  # local
        return int(np.argmin(costs))

    baselines = [
        ("Local", local_fn),
        ("Random", random_fn),
        ("RoundRobin", rr_fn),
        ("Greedy", greedy_fn),
    ]
    for name, fn in baselines:
        hits, delays, energies, loc = [], [], [], []
        for _ in range(episodes):
            env = FogOffloadEnv()
            s = env.reset()
            done = False
            while not done:
                s, _, done, _ = env.step(fn(s))
            m = env.metrics()
            hits.append(m["hit_ratio"]); delays.append(m["avg_delay"])
            energies.append(m["avg_energy"]); loc.append(m["local_ratio"])
        print(f"{name:<12}{np.mean(hits)*100:<8.1f}{np.mean(delays):<11.4f}"
              f"{np.mean(energies):<9.3f}{np.mean(loc)*100:<9.1f}")

    if trained_policy is not None:
        hits, delays, energies, loc = [], [], [], []
        for _ in range(episodes):
            env = FogOffloadEnv()
            s = env.reset()
            done = False
            while not done:
                s, _, done, _ = env.step(trained_policy.predict_masked(s, env.allowed_actions()))
            m = env.metrics()
            hits.append(m["hit_ratio"]); delays.append(m["avg_delay"])
            energies.append(m["avg_energy"]); loc.append(m["local_ratio"])
        print(f"{'DQN':<12}{np.mean(hits)*100:<8.1f}{np.mean(delays):<11.4f}"
              f"{np.mean(energies):<9.3f}{np.mean(loc)*100:<9.1f}")


# ========== Main ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--tasks", type=int, default=NUM_TASKS_PER_EP)
    ap.add_argument("--quick", action="store_true",
                    help="small run for smoke-testing (episodes=50, tasks=200)")
    args = ap.parse_args()

    eps = 50 if args.quick else args.episodes
    tasks = 200 if args.quick else args.tasks

    print("=" * 72)
    print("DQN Task Offloading Trainer — heterogeneous CPU/GPU/QPU fog")
    print("=" * 72)
    print(f"Fog: IBM(QPU), IonQ(QPU), RTX3090(GPU)x2, CPU-11core")
    print(f"State dim: {STATE_DIM}, Actions: {NUM_ACTIONS}, Hidden: {HIDDEN_DIM}")
    print(f"Episodes: {eps}, Tasks/ep: {tasks}")

    policy, all_states = train(episodes=eps, tasks_per_ep=tasks)
    evaluate_baselines(policy)
    export_weights(policy, all_states, output_dir="dqn_weights")

    print("\n" + "=" * 72)
    print("DONE! Now run the simulation with DQN:")
    print("  ../src/Tutorial -u Cmdenv -c DQN")
    print("=" * 72)


if __name__ == "__main__":
    main()
