package tutorial.simulations;

//
// Heterogeneous IoT-Fog network (teacher feedback / Quantom.md).
//   5 fog servers: 2x QPU (IBM, IonQ), 2x GPU (RTX 3090), 1x CPU (11-core).
//   5G links with random 1-10 ms one-way delay.
//
simple IoTDevice
{
    parameters:
        @display("i=device/pocketpc_s");
        int numFogServers;
        string taskFile = default("tasks.csv");
        string decisionMode = default("roundRobin");  // local | random | roundRobin | greedy | dqn
        string modelDir = default("");                 // path to exported DQN weights

        // Local IoT execution model — Raspberry Pi 4B (hospital/edge domain).
        //   SoC: Broadcom BCM2711, quad-core Cortex-A72 @ 1.5 GHz.
        //   Peak FP32 (NEON, 4 cores) ~ 12 GFLOP/s; derated to an EFFECTIVE
        //   sustained-available throughput of 600 MFLOP/s under the device's
        //   OS + sensing stack and battery power constraint. Active power ~5 W.
        //   Energy = power * execTime,  execTime = workload / localThroughput.
        double localThroughput = default(600);          // MFLOP/s (Pi 4B effective sustained)
        double localPower = default(5.0);               // Watts (Pi 4B active CPU)
        double txPower = default(1.5);                  // 5G uplink transmit power (W)
        double uplinkRate = default(100e6);             // 5G uplink data rate (bps)
        double linkDelay @unit(s) = default(0.005s);    // fallback one-way delay
        double qpuTheta = default(0.7);                 // quantum-suitability threshold (local-feasibility)

        @signal[msgSent](type=long);
        @signal[e2eDelay](type=double);
        @signal[deadlineHit](type=long);
        @signal[energy](type=double);
        @statistic[msgSent](title="messages sent"; record=count,vector);
        @statistic[e2eDelay](title="end-to-end delay"; record=mean,max,vector);
        @statistic[deadlineHit](title="deadline hit (1=met)"; record=mean,count);
        @statistic[energy](title="energy per task"; record=mean,sum,vector);
    gates:
        input  in[];
        output out[];
}

simple FogServer
{
    parameters:
        @display("i=device/server");
        double statusInterval @unit(s) = default(0.1s);  // load/utilization broadcast period

        // Heterogeneous processor specs (overridden per fog in omnetpp.ini)
        int processorType = default(0);          // 0=CPU, 1=GPU, 2=QPU
        string procName = default("CPU-11core");
        double throughput = default(1000);        // MFLOP/s
        double power = default(65);               // Watts (energy = power * execTime)
        double efficiency = default(1.0);         // GPU eta
        double gpuBandwidth = default(936e6);     // bytes/s (GPU copy BW)
        double gpuLaunchOverhead @unit(s) = default(0.0001s);
        double qpuOverhead @unit(s) = default(0.0015s);
        double qpuTheta = default(0.7);           // quantum-suitability threshold

        @signal[msgReceived](type=long);
        @signal[respTime](type=simtime_t);
        @signal[queueWait](type=double);
        @signal[serverTurnaround](type=double);
        @signal[queueBacklog](type=double);
        @signal[execTime](type=double);
        @signal[energy](type=double);
        @signal[fogUtilization](type=double);
        @statistic[msgReceived](title="messages received"; record=count,vector);
        @statistic[respTime](title="response time net+queue+exec"; record=mean,max,vector);
        @statistic[queueWait](title="queue waiting time (s)"; record=mean,max,vector);
        @statistic[serverTurnaround](title="server turnaround = queue+exec (s)"; record=mean,max,vector);
        @statistic[queueBacklog](title="queue backlog (s)"; record=mean,max,vector);
        @statistic[execTime](title="execution time"; record=mean,max,vector);
        @statistic[energy](title="fog exec energy"; record=mean,sum,vector);
        @statistic[fogUtilization](title="fog utilization"; record=mean,max,vector);
    gates:
        input  in[];
        output out[];
}

network FogNetwork
{
    parameters:
        int numIoT    = default(15);
        int numFog    = default(5);
    submodules:
        iot[numIoT]: IoTDevice {
            numFogServers = parent.numFog;
        }
        fog[numFog]: FogServer;
    connections:
        // Fully connect every IoT device to every fog server over a 5G link with
        // random 1-10 ms one-way delay (teacher: "network delay random 1-10 ms").
        for i=0..numIoT-1, for j=0..numFog-1 {
            iot[i].out++ --> { delay = uniform(1ms, 10ms); } --> fog[j].in++;
            fog[j].out++ --> { delay = uniform(1ms, 10ms); } --> iot[i].in++;
        }
}
