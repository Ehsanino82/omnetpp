#!/usr/bin/env python3
"""
Train a DQN for IoT task offloading decisions (pure NumPy, no PyTorch needed).
Actions: 0 = process locally, 1..NUM_FOG = offload to fog server i-1.
State:   [cpuDemand, deadlineSlack, fogLoad0, ..., fogLoad4]
Exports: weight matrices + scaler to CSV files for C++ inference.
"""

import numpy as np
import os
import csv
import random
from collections import deque
from copy import deepcopy

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ========== Configuration ==========
NUM_FOG = 5
NUM_ACTIONS = NUM_FOG + 1   # 0=local, 1..5=fog servers
STATE_DIM = 2 + NUM_FOG     # cpuDemand, deadlineSlack, fogLoad[0..4]

EPISODES = 300
NUM_TASKS_PER_EP = 200
BATCH_SIZE = 64
BUFFER_SIZE = 20000
GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
HIDDEN_DIM = 128


# ========== Environment ==========
class FogOffloadEnv:
    # Parameters aligned with the OMNeT++ simulation (spec Section 9.1, Option A):
    #   fog CPU rate = 1.0, local CPU rate = 0.5, one-way link delay = 10ms,
    #   task/deadline distributions match generate_tasks.py.
    def __init__(self, num_tasks=NUM_TASKS_PER_EP):
        self.num_tasks = num_tasks
        self.fog_rates = np.ones(NUM_FOG)               # fog processing rate = 1.0
        self.local_rate = 0.5                           # matches *.iot[*].localCpuRate
        self.link_delays = np.full(NUM_FOG, 0.01)       # 10ms one-way link (FogNetwork.ned)
        self.reset()

    def reset(self):
        self.fog_free_at = np.zeros(NUM_FOG)
        self.local_free_at = 0.0
        self.now = 0.0
        self.task_id = 0
        self.total_delay = 0.0
        self.dropped = 0
        self.processed = 0
        self.local_count = 0
        self.offload_count = 0
        self._gen_task()
        return self._get_obs()

    def _gen_task(self):
        # Ranges match generate_tasks.py (cpuDemand in seconds, deadline slack in seconds)
        self.cpu_demand = np.random.uniform(0.005, 0.2)
        self.deadline_slack = np.random.uniform(3.0, 18.0)

    def _get_obs(self):
        fog_loads = np.maximum(0, self.fog_free_at - self.now)
        return np.array([
            self.cpu_demand,
            self.deadline_slack,
            *fog_loads
        ], dtype=np.float32)

    def step(self, action):
        if self.task_id >= self.num_tasks:
            return self._get_obs(), 0.0, True, {}

        if action == 0:
            proc_time = self.cpu_demand / self.local_rate
            start = max(self.now, self.local_free_at)
            finish = start + proc_time
            self.local_free_at = finish
            latency = finish - self.now
            self.local_count += 1
        else:
            fog_idx = action - 1
            proc_time = self.cpu_demand / self.fog_rates[fog_idx]
            network_delay = self.link_delays[fog_idx] * 2
            start = max(self.now + self.link_delays[fog_idx], self.fog_free_at[fog_idx])
            finish = start + proc_time
            self.fog_free_at[fog_idx] = finish
            latency = (finish - self.now) + self.link_delays[fog_idx]
            self.offload_count += 1

        self.total_delay += latency
        self.processed += 1
        self.task_id += 1

        reward = -latency * 10.0
        if latency > self.deadline_slack:
            self.dropped += 1
            reward -= 20.0

        self.now += np.random.uniform(0.01, 0.1)

        done = self.task_id >= self.num_tasks
        if not done:
            self._gen_task()

        return self._get_obs(), reward, done, {}

    def metrics(self):
        p = max(1, self.processed)
        return {
            "avg_delay": self.total_delay / p,
            "hit_ratio": 1.0 - self.dropped / p,
            "local_ratio": self.local_count / p,
            "offload_ratio": self.offload_count / p,
        }


# ========== NumPy MLP DQN ==========
class NumpyMLP:
    """Simple 3-layer MLP with ReLU, using NumPy only."""
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
        """x: (batch, input_dim) -> (batch, output_dim)"""
        h1 = np.maximum(0, x @ self.W1.T + self.b1)          # ReLU
        h2 = np.maximum(0, h1 @ self.W2.T + self.b2)         # ReLU
        return h2 @ self.W3.T + self.b3

    def predict(self, x):
        """x: (input_dim,) -> action"""
        q = self.forward(x.reshape(1, -1))
        return int(np.argmax(q))

    def copy_from(self, other):
        self.W1 = other.W1.copy()
        self.b1 = other.b1.copy()
        self.W2 = other.W2.copy()
        self.b2 = other.b2.copy()
        self.W3 = other.W3.copy()
        self.b3 = other.b3.copy()

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
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(ns, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


def train_step(policy, target, buf, lr=LR, gamma=GAMMA):
    """One gradient step using MSE loss on Q-values (NumPy backprop)."""
    bs, ba, br, bns, bd = buf.sample(BATCH_SIZE)

    # Forward pass through policy
    h1_pre = bs @ policy.W1.T + policy.b1
    h1 = np.maximum(0, h1_pre)
    h2_pre = h1 @ policy.W2.T + policy.b2
    h2 = np.maximum(0, h2_pre)
    q_all = h2 @ policy.W3.T + policy.b3

    # Get Q(s,a) for taken actions
    q_sa = q_all[np.arange(len(ba)), ba]

    # Target Q
    nq_all = target.forward(bns)
    nq_max = nq_all.max(axis=1)
    target_q = br + gamma * nq_max * (1 - bd)

    # Loss gradient: d/dq of (q - target)^2 = 2*(q - target)
    error = q_sa - target_q           # (batch,)
    loss = float(np.mean(error ** 2))

    # Backprop through layer 3
    dq = np.zeros_like(q_all)         # (batch, output_dim)
    dq[np.arange(len(ba)), ba] = 2 * error / len(ba)

    dW3 = dq.T @ h2                   # (output_dim, hidden_dim)
    db3 = dq.sum(axis=0)
    dh2 = dq @ policy.W3              # (batch, hidden_dim)

    # ReLU backward layer 2
    dh2_pre = dh2 * (h2_pre > 0)
    dW2 = dh2_pre.T @ h1
    db2 = dh2_pre.sum(axis=0)
    dh1 = dh2_pre @ policy.W2

    # ReLU backward layer 1
    dh1_pre = dh1 * (h1_pre > 0)
    dW1 = dh1_pre.T @ bs
    db1 = dh1_pre.sum(axis=0)

    # Gradient clipping
    for grad in [dW1, db1, dW2, db2, dW3, db3]:
        norm = np.linalg.norm(grad)
        if norm > 1.0:
            grad *= 1.0 / norm

    # SGD update
    policy.W1 -= lr * dW1
    policy.b1 -= lr * db1
    policy.W2 -= lr * dW2
    policy.b2 -= lr * db2
    policy.W3 -= lr * dW3
    policy.b3 -= lr * db3

    return loss


def train():
    env = FogOffloadEnv()
    buf = ReplayBuffer()
    policy = NumpyMLP(STATE_DIM, HIDDEN_DIM, NUM_ACTIONS)
    target = NumpyMLP(STATE_DIM, HIDDEN_DIM, NUM_ACTIONS)
    target.copy_from(policy)

    epsilon = EPSILON_START
    all_states = []

    print(f"\n{'='*70}")
    print("TRAINING DQN OFFLOADING POLICY (NumPy)")
    print(f"{'='*70}")
    print(f"State dim: {STATE_DIM}, Actions: {NUM_ACTIONS}, Hidden: {HIDDEN_DIM}")
    print(f"{'Ep':<6}{'Delay':<10}{'Hit%':<10}{'Local%':<10}{'Loss':<12}{'Reward':<10}")
    print("-" * 70)

    best_hit = 0.0
    best_params = None

    for ep in range(EPISODES):
        state = env.reset()
        ep_reward = 0.0
        ep_losses = []

        while True:
            all_states.append(state.copy())

            if random.random() < epsilon:
                action = random.randint(0, NUM_ACTIONS - 1)
            else:
                action = policy.predict(state)

            ns, reward, done, _ = env.step(action)
            buf.add(state, action, reward, ns, done)
            state = ns
            ep_reward += reward

            if len(buf) > BATCH_SIZE:
                loss = train_step(policy, target, buf)
                ep_losses.append(loss)

            if done:
                break

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if ep % TARGET_UPDATE == 0:
            target.copy_from(policy)

        m = env.metrics()
        if m["hit_ratio"] >= best_hit:
            best_hit = m["hit_ratio"]
            best_params = [p.copy() for p in policy.get_params()]

        if (ep + 1) % 20 == 0:
            avg_loss = np.mean(ep_losses) if ep_losses else 0
            print(f"{ep:<6}{m['avg_delay']:<10.4f}{m['hit_ratio']*100:<10.1f}"
                  f"{m['local_ratio']*100:<10.1f}{avg_loss:<12.6f}{ep_reward:<10.1f}")

    if best_params:
        policy.set_params(best_params)

    print(f"\nBest hit ratio during training: {best_hit*100:.1f}%")
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
        for m, s in zip(mean, std):
            f.write(f"{m:.8f},{s:.8f}\n")

    print(f"\n✅ Weights exported to: {output_dir}/")
    print(f"   Files: w1.csv, b1.csv, w2.csv, b2.csv, w3.csv, b3.csv, scaler.csv")


# ========== Evaluate baselines ==========
def evaluate_baselines(trained_policy=None):
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON (30 episodes each)")
    print(f"{'='*70}")
    print(f"{'Policy':<15}{'Hit%':<10}{'Delay':<10}{'Local%':<10}")
    print("-" * 70)

    rr_counter = [0]
    baselines = [
        ("Local",      lambda s: 0),
        ("Random",     lambda s: random.randint(0, NUM_ACTIONS - 1)),
        ("RoundRobin", lambda s: (rr_counter.__setitem__(0, (rr_counter[0] % NUM_FOG) + 1), rr_counter[0])[1]),
        ("Greedy",     lambda s: int(np.argmin(s[2:])) + 1),
    ]

    for name, choose_fn in baselines:
        hits, delays, locals_ = [], [], []
        for _ in range(30):
            env = FogOffloadEnv()
            state = env.reset()
            done = False
            while not done:
                action = choose_fn(state)
                state, _, done, _ = env.step(action)
            m = env.metrics()
            hits.append(m["hit_ratio"])
            delays.append(m["avg_delay"])
            locals_.append(m["local_ratio"])
        print(f"{name:<15}{np.mean(hits)*100:<10.1f}{np.mean(delays):<10.4f}"
              f"{np.mean(locals_)*100:<10.1f}")

    if trained_policy:
        hits, delays, locals_ = [], [], []
        for _ in range(30):
            env = FogOffloadEnv()
            state = env.reset()
            done = False
            while not done:
                action = trained_policy.predict(state)
                state, _, done, _ = env.step(action)
            m = env.metrics()
            hits.append(m["hit_ratio"])
            delays.append(m["avg_delay"])
            locals_.append(m["local_ratio"])
        print(f"{'DQN':<15}{np.mean(hits)*100:<10.1f}{np.mean(delays):<10.4f}"
              f"{np.mean(locals_)*100:<10.1f}")


# ========== Main ==========
if __name__ == "__main__":
    print("=" * 70)
    print("DQN Task Offloading Trainer for OMNeT++ Fog Simulation")
    print("=" * 70)
    print(f"Fog servers: {NUM_FOG}, Actions: {NUM_ACTIONS} (local + {NUM_FOG} fog)")
    print(f"State dim: {STATE_DIM}, Hidden: {HIDDEN_DIM}")
    print(f"Episodes: {EPISODES}, Tasks/ep: {NUM_TASKS_PER_EP}")

    policy, all_states = train()
    evaluate_baselines(policy)
    export_weights(policy, all_states, output_dir="dqn_weights")

    print("\n" + "=" * 70)
    print("DONE! Now run the simulation with DQN:")
    print("  ./Tutorial -u Cmdenv -c DQN")
    print("=" * 70)
