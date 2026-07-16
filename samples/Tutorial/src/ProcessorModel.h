#ifndef __PROCESSOR_MODEL_H
#define __PROCESSOR_MODEL_H

// Heterogeneous processor execution-time model (teacher feedback, Quantom.md
// phases 5-8). Single source of truth shared by FogServer.cc (actual exec) and
// IoTDevice.cc (completion-time estimate for greedy / DQN state).
//
// Processor types:
//   PROC_CPU (0): 11-core CPU          E = F / P_CPU
//   PROC_GPU (1): RTX 3090 GPU         E = T_launch + 2*data/BW + F/(P_GPU*eta)
//   PROC_QPU (2): IBM / IonQ QPU       E = T_overhead + F / (P_QPU * Qs^2)
//
// For QPU, tasks whose quantum suitability Qs is below `qpuTheta` are heavily
// penalized (effective throughput ~0), modeling that a QPU is wasteful for
// non-quantum-optimizable workloads (Quantom.md phase 4 decision rule).

static const int PROC_CPU = 0;
static const int PROC_GPU = 1;
static const int PROC_QPU = 2;

struct ProcSpec {
    int type = PROC_CPU;        // PROC_CPU / PROC_GPU / PROC_QPU
    double throughput = 0.0;    // MFLOP/s
    double power = 0.0;         // Watts (energy = power * execTime)
    double efficiency = 1.0;    // GPU eta
    double gpuBandwidth = 0.0;  // bytes/s
    double gpuLaunchOverhead = 0.0; // s
    double qpuOverhead = 0.0;   // s
    double qpuTheta = 0.7;      // quantum-suitability threshold
};

inline double computeExecTime(const ProcSpec &s, double workload, double qs, double dataBytes)
{
    const double eps = 1e-9;
    switch (s.type) {
        case PROC_GPU: {
            double copyTime = (s.gpuBandwidth > eps)
                              ? (2.0 * dataBytes / s.gpuBandwidth)
                              : 0.0;
            double kernel = (s.throughput * s.efficiency > eps)
                            ? (workload / (s.throughput * s.efficiency))
                            : 1e6;
            return s.gpuLaunchOverhead + copyTime + kernel;
        }
        case PROC_QPU: {
            // Quantum-optimizable tasks (Qs >= theta): quantum speedup, exec time
            // shrinks with Qs^2. Non-quantum tasks (Qs < theta) cannot use the
            // quantum algorithm, so the QPU falls back to its slow classical host
            // controller (finite penalty, not a division-by-zero explosion).
            if (qs >= s.qpuTheta) {
                double eff = s.throughput * qs * qs + eps;
                return s.qpuOverhead + workload / eff;
            }
            const double qpuClassicalThroughput = 50.0;   // MFLOP/s (QPU host CPU)
            return s.qpuOverhead + workload / qpuClassicalThroughput;
        }
        case PROC_CPU:
        default:
            return (s.throughput > eps) ? (workload / s.throughput) : 1e6;
    }
}

#endif
