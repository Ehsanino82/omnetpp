#include <omnetpp.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "Task_m.h"
#include "ServerStatus_m.h"
#include "DqnPolicy.h"
#include "ProcessorModel.h"

using namespace omnetpp;

struct TaskConfig {
    int taskId, ownerDeviceId, sizeBytes, priority;
    double arrivalTime, workload, quantumSuitability, deadline;
    double dataSizeKb, energyLocalEst;
    std::string taskType;
};

class IoTDevice : public cSimpleModule
{
  private:
    int numFogServers;
    int myId;
    int nextServer = 0;                     // round-robin fallback
    std::vector<TaskConfig> myTasks;
    size_t nextTaskIdx = 0;
    cMessage *releaseTimer = nullptr;
    simsignal_t msgSentSignal;
    simsignal_t e2eDelaySignal;
    simsignal_t deadlineHitSignal;
    simsignal_t energySignal;

    // Decision mode: "local", "random", "roundRobin", "greedy", "dqn"
    std::string decisionMode;

    // DQN policy
    DqnPolicy dqnPolicy;

    // Per-fog processor specs (read from the fog submodules) + measured 5G link
    // delays. Both are used to estimate completion time (greedy) and to build
    // the DQN state.
    std::vector<ProcSpec> fogSpecs;
    std::vector<double> fogLinkDelay;       // one-way network delay to each fog (s)
    std::vector<double> fogLoads;           // queue backlog from ServerStatus (s)

    // Local processing model
    double localThroughput;                 // MFLOP/s
    double localPower;                      // Watts
    double txPower;                         // 5G uplink transmit power (W)
    double uplinkRate;                      // 5G uplink data rate (bps)
    double linkDelayFallback;               // if channel delay not readable
    double qpuTheta;                        // quantum-suitability threshold
    simtime_t localNextFreeTime = SIMTIME_ZERO;

    // Statistics
    int tasksProcessedLocally = 0;
    int tasksOffloaded = 0;
    int deadlinesMet = 0;
    int deadlinesMissed = 0;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    void loadTasks(const char *filename);
    void scheduleNextTask();
    void sendTask(const TaskConfig &tc);
    int chooseAction(const TaskConfig &tc);
    double estimateFogExec(int j, const TaskConfig &tc) const;
    double localExecTime(const TaskConfig &tc) const;
    double localEnergy(const TaskConfig &tc) const;
    double txEnergy(const TaskConfig &tc) const;
    bool isQuantumTask(const TaskConfig &tc) const;
    bool isGpuTaskType(const std::string &t) const;
    bool localAllowed(const TaskConfig &tc) const;   // GPU/quantum tasks must offload
    int pickBestFog(const TaskConfig &tc) const;      // greedy argmin fog index
    void processLocally(const TaskConfig &tc);
    void recordCompletion(double e2eDelay, bool deadlineMet, double energy);
    virtual void finish() override;
    ~IoTDevice() { cancelAndDelete(releaseTimer); }
};

Define_Module(IoTDevice);

void IoTDevice::initialize()
{
    numFogServers  = par("numFogServers");
    myId           = getIndex();
    decisionMode   = par("decisionMode").stdstringValue();
    localThroughput= par("localThroughput").doubleValue();
    localPower     = par("localPower").doubleValue();
    txPower        = par("txPower").doubleValue();
    uplinkRate     = par("uplinkRate").doubleValue();
    linkDelayFallback = par("linkDelay").doubleValue();
    qpuTheta       = par("qpuTheta").doubleValue();

    msgSentSignal     = registerSignal("msgSent");
    e2eDelaySignal    = registerSignal("e2eDelay");
    deadlineHitSignal = registerSignal("deadlineHit");
    energySignal      = registerSignal("energy");

    fogSpecs.resize(numFogServers);
    fogLinkDelay.resize(numFogServers, linkDelayFallback);
    fogLoads.resize(numFogServers, 0.0);

    // Read each fog server's processor specs from the submodule so the device's
    // estimate uses the exact same model the fog executes (single source of
    // truth in ProcessorModel.h / the fog params).
    cModule *net = getParentModule();
    for (int j = 0; j < numFogServers; j++) {
        cModule *fog = net->getSubmodule("fog", j);
        if (fog) {
            ProcSpec &s = fogSpecs[j];
            s.type              = fog->par("processorType").intValue();
            s.throughput        = fog->par("throughput").doubleValue();
            s.power             = fog->par("power").doubleValue();
            s.efficiency        = fog->par("efficiency").doubleValue();
            s.gpuBandwidth      = fog->par("gpuBandwidth").doubleValue();
            s.gpuLaunchOverhead = fog->par("gpuLaunchOverhead").doubleValue();
            s.qpuOverhead       = fog->par("qpuOverhead").doubleValue();
            s.qpuTheta          = fog->par("qpuTheta").doubleValue();
        }
        // Read the actual 5G link delay configured on this iot->fog channel.
        cGate *g = gate("out", j);
        if (g) {
            cChannel *ch = g->getChannel();
            if (ch) {
                try {
                    cPar &dp = ch->par("delay");
                    if (dp.isSet())
                        fogLinkDelay[j] = dp.doubleValue();
                } catch (const std::exception&) {
                    // channel has no delay parameter -> keep fallback
                }
            }
        }
    }

    if (decisionMode == "dqn") {
        std::string modelDir = par("modelDir").stdstringValue();
        try {
            dqnPolicy.load(modelDir);
            EV << getName() << " loaded DQN model from " << modelDir << "\n";
        } catch (const std::exception& e) {
            EV_WARN << getName() << " failed to load DQN model: " << e.what()
                    << ", falling back to greedy\n";
            decisionMode = "greedy";
        }
    }

    loadTasks(par("taskFile").stringValue());

    releaseTimer = new cMessage("releaseTimer");
    scheduleNextTask();
}

void IoTDevice::loadTasks(const char *filename)
{
    std::ifstream f(filename);
    if (!f.is_open())
        throw cRuntimeError("Cannot open task file: %s", filename);

    std::string line;
    std::getline(f, line);                  // skip header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        TaskConfig tc;
        // Schema: taskId,ownerDeviceId,arrivalTime,taskType,dataSizeKb,
        //         workload,quantumSuitability,deadline,priority,memoryMb,energyLocalEst
        std::getline(ss, cell, ','); tc.taskId             = std::stoi(cell);
        std::getline(ss, cell, ','); tc.ownerDeviceId      = std::stoi(cell);
        std::getline(ss, cell, ','); tc.arrivalTime        = std::stod(cell);
        std::getline(ss, cell, ','); tc.taskType           = cell;   // task type (local-feasibility rule)
        std::getline(ss, cell, ','); tc.dataSizeKb        = std::stod(cell);
        std::getline(ss, cell, ','); tc.workload          = std::stod(cell);
        std::getline(ss, cell, ','); tc.quantumSuitability= std::stod(cell);
        std::getline(ss, cell, ','); tc.deadline          = std::stod(cell);
        std::getline(ss, cell, ','); tc.priority          = std::stoi(cell);
        std::getline(ss, cell, ',');                    /* memoryMb (unused) */
        std::getline(ss, cell, ','); tc.energyLocalEst    = std::stod(cell);
        tc.sizeBytes = (int)std::lround(tc.dataSizeKb * 1024.0);

        if (tc.ownerDeviceId == myId)
            myTasks.push_back(tc);
    }
    EV << getName() << " loaded " << myTasks.size() << " tasks\n";
}

void IoTDevice::scheduleNextTask()
{
    if (nextTaskIdx < myTasks.size())
        scheduleAt(myTasks[nextTaskIdx].arrivalTime, releaseTimer);
}

double IoTDevice::localExecTime(const TaskConfig &tc) const
{
    return (localThroughput > 1e-9) ? (tc.workload / localThroughput) : 1e6;
}

double IoTDevice::localEnergy(const TaskConfig &tc) const
{
    // E_local = P_local * T_local  (dynamic in task size/workload)
    return localPower * localExecTime(tc);
}

double IoTDevice::txEnergy(const TaskConfig &tc) const
{
    // E_tx = P_tx * D / R_up   (D = payload bits, R_up = 5G uplink rate)
    double bits = tc.sizeBytes * 8.0;
    return (uplinkRate > 1e-9) ? (txPower * bits / uplinkRate) : 0.0;
}

double IoTDevice::estimateFogExec(int j, const TaskConfig &tc) const
{
    double dataBytes = (double)tc.sizeBytes;
    return computeExecTime(fogSpecs[j], tc.workload, tc.quantumSuitability, dataBytes);
}

// Task classification for the local-feasibility rule (teacher feedback):
//   - Quantum task   (Qs >= qpuTheta)               -> MUST offload (QPU/GPU)
//   - GPU-oriented task (image/cnn/video)           -> MUST offload (GPU)
//   - CPU/control task (sensor/control/aggr/alert)  -> may run locally
// Rationale: the battery-powered IoT device is too weak/slow for heavy GPU or
// quantum-optimizable work; running those locally would blow deadlines and drain
// the battery. Only light CPU/control tasks are local-feasible.
bool IoTDevice::isQuantumTask(const TaskConfig &tc) const
{
    return tc.quantumSuitability >= qpuTheta;
}

bool IoTDevice::isGpuTaskType(const std::string &t) const
{
    return t == "image_processing" || t == "cnn_inference" || t == "video_processing";
}

bool IoTDevice::localAllowed(const TaskConfig &tc) const
{
    if (isQuantumTask(tc))  return false;
    if (isGpuTaskType(tc.taskType)) return false;
    return true;            // CPU / control / sensor task -> local is feasible
}

// Greedy completion-time fog pick (network + queue + exec), restricted to fogs
// that are suitable for the task (non-quantum tasks skip QPU fogs). Used to
// force-offload GPU/quantum tasks that may not run locally.
int IoTDevice::pickBestFog(const TaskConfig &tc) const
{
    int best = -1;
    double bestCost = 1e18;
    for (int j = 0; j < numFogServers; j++) {
        bool isQpu = (fogSpecs[j].type == PROC_QPU);
        if (isQpu && tc.quantumSuitability < fogSpecs[j].qpuTheta)
            continue;                       // non-quantum task -> skip QPU
        double cost = fogLinkDelay[j] + fogLoads[j] + estimateFogExec(j, tc);
        if (cost < bestCost) {
            bestCost = cost;
            best = j;
        }
    }
    if (best < 0) best = 0;                 // fallback (should not happen)
    return best;
}

void IoTDevice::handleMessage(cMessage *msg)
{
    if (msg == releaseTimer) {
        sendTask(myTasks[nextTaskIdx]);
        nextTaskIdx++;
        scheduleNextTask();
        return;
    }

    // Handle ServerStatus messages (fog load feedback)
    ServerStatus *ss = dynamic_cast<ServerStatus *>(msg);
    if (ss) {
        int sid = ss->getServerId();
        if (sid >= 0 && sid < numFogServers)
            fogLoads[sid] = ss->getLoadSec();
        delete msg;
        return;
    }

    // Handle completion ack from a fog server
    Task *ack = dynamic_cast<Task *>(msg);
    if (ack && ack->isAck()) {
        double e2e = ack->getFinishTime() - ack->getReleaseTime();
        bool met  = ack->getFinishTime() <= ack->getDeadline();
        // Total energy for an offloaded task = transmission + fog execution
        double energy = txEnergy(TaskConfig{0, 0, ack->getSizeBytes(), 0,
                                            0, ack->getWorkload(),
                                            ack->getQuantumSuitability(),
                                            ack->getDeadline(),
                                            ack->getDataSizeKb(), 0.0})
                        + ack->getEnergy();
        recordCompletion(e2e, met, energy);
        EV << getName() << " task " << ack->getTaskId() << " completed at fog["
           << ack->getTargetServer() << "] (" << (ack->getProcessorType()==2?"QPU":
                ack->getProcessorType()==1?"GPU":"CPU") << ")"
           << " e2e=" << e2e * 1000.0 << "ms "
           << (met ? "HIT" : "MISS") << "\n";
        delete msg;
        return;
    }

    delete msg;
}

int IoTDevice::chooseAction(const TaskConfig &tc)
{
    // ---- Local-feasibility rule (teacher feedback) -------------------------
    // GPU-oriented and quantum-optimizable tasks MUST be offloaded (the IoT
    // device is too weak / battery-bound to run them). Only CPU/control tasks
    // may be processed locally. This is enforced for EVERY policy below by
    // removing action 0 (local) from the feasible set when !localAllowed(tc).
    const bool canLocal = localAllowed(tc);

    if (decisionMode == "local") {
        // Local-preferred baseline: run CPU tasks locally; GPU/quantum tasks
        // cannot run locally, so offload them to the best (greedy) fog.
        if (canLocal) return 0;
        return pickBestFog(tc) + 1;
    }

    if (decisionMode == "random") {
        // Uniform random over the feasible set: local (if allowed) + all fogs.
        std::vector<int> pool;
        if (canLocal) pool.push_back(0);
        for (int j = 0; j < numFogServers; j++) pool.push_back(j + 1);
        return pool[intuniform(0, (int)pool.size() - 1)];
    }

    // RoundRobin: dumb baseline — always offload, cycle through fog servers
    // (no execution-time / queue awareness, per teacher note). Never uses local.
    if (decisionMode == "roundRobin") {
        int chosen = nextServer + 1;
        nextServer = (nextServer + 1) % numFogServers;
        return chosen;
    }

    // Intelligent completion-time selection (teacher feedback / Quantom.md
    // phase 8-9): C = network_delay + queue_backlog + execution_time, pick min.
    if (decisionMode == "greedy") {
        std::vector<double> costs;
        std::vector<int> actions;
        if (canLocal) {
            costs.push_back(std::max(0.0, (localNextFreeTime - simTime()).dbl())
                            + localExecTime(tc));                       // action 0
            actions.push_back(0);
        }
        for (int j = 0; j < numFogServers; j++) {
            bool isQpu = (fogSpecs[j].type == PROC_QPU);
            if (isQpu && tc.quantumSuitability < fogSpecs[j].qpuTheta)
                continue;                       // non-quantum task -> skip QPU
            costs.push_back(fogLinkDelay[j] + fogLoads[j] + estimateFogExec(j, tc));
            actions.push_back(j + 1);
        }

        double bestCost = *std::min_element(costs.begin(), costs.end());
        std::vector<int> candidates;
        for (int a = 0; a < (int)costs.size(); a++)
            if (costs[a] <= bestCost + 1e-9)
                candidates.push_back(actions[a]);
        return candidates[intuniform(0, (int)candidates.size() - 1)];
    }

    if (decisionMode == "dqn" && dqnPolicy.isLoaded()) {
        // State (18-dim): [workload, Qs, deadlineSlack,
        //                  netDelay0..4, fogBacklog0..4, execEst0..4]
        std::vector<double> state;
        state.push_back(tc.workload);
        state.push_back(tc.quantumSuitability);
        state.push_back(tc.deadline - simTime().dbl());
        for (int j = 0; j < numFogServers; j++) state.push_back(fogLinkDelay[j]);
        for (int j = 0; j < numFogServers; j++) state.push_back(fogLoads[j]);
        for (int j = 0; j < numFogServers; j++) state.push_back(estimateFogExec(j, tc));

        // Action mask (Quantom.md phase 4 + local-feasibility rule):
        //   - local (0) only for CPU/control tasks
        //   - QPU fogs only for quantum-suitable tasks
        std::vector<int> allowed;
        if (canLocal) allowed.push_back(0);
        for (int j = 0; j < numFogServers; j++) {
            bool isQpu = (fogSpecs[j].type == PROC_QPU);
            if (isQpu && tc.quantumSuitability < fogSpecs[j].qpuTheta)
                continue;                          // non-quantum task -> skip QPU
            allowed.push_back(j + 1);
        }
        return dqnPolicy.actMasked(state, allowed);   // 0=local, 1..5=fog[0..4]
    }

    // roundRobin fallback: always offload, cycle through fog servers
    int chosen = nextServer + 1;
    nextServer = (nextServer + 1) % numFogServers;
    return chosen;
}

void IoTDevice::processLocally(const TaskConfig &tc)
{
    double localProcTime = localExecTime(tc);
    simtime_t start  = std::max(simTime(), localNextFreeTime);
    simtime_t finish = start + localProcTime;
    localNextFreeTime = finish;

    double e2eDelay = (finish - simTime()).dbl();
    bool met = finish.dbl() <= tc.deadline;
    tasksProcessedLocally++;

    recordCompletion(e2eDelay, met, localEnergy(tc));

    EV << getName() << " processing task " << tc.taskId
       << " LOCALLY (proc=" << localProcTime * 1000.0 << "ms) "
       << (met ? "HIT" : "MISS") << "\n";
}

void IoTDevice::recordCompletion(double e2eDelay, bool deadlineMet, double energy)
{
    emit(e2eDelaySignal, e2eDelay);
    emit(deadlineHitSignal, deadlineMet ? 1L : 0L);
    emit(energySignal, energy);
    if (deadlineMet) deadlinesMet++;
    else             deadlinesMissed++;
}

void IoTDevice::sendTask(const TaskConfig &tc)
{
    int action = chooseAction(tc);

    if (action == 0) {
        processLocally(tc);
        return;
    }

    // Offload to fog server (action 1..numFogServers -> fog index 0..numFogServers-1)
    int chosen = action - 1;
    if (chosen < 0 || chosen >= numFogServers) {
        chosen = nextServer;
        nextServer = (nextServer + 1) % numFogServers;
    }

    Task *t = new Task("task");
    t->setTaskId(tc.taskId);
    t->setOwnerDeviceId(tc.ownerDeviceId);
    t->setTargetServer(chosen);
    t->setWorkload(tc.workload);
    t->setCpuDemand(tc.workload);
    t->setQuantumSuitability(tc.quantumSuitability);
    t->setDataSizeKb(tc.dataSizeKb);
    t->setSizeBytes(tc.sizeBytes);
    t->setDeadline(tc.deadline);
    t->setNetDelay(fogLinkDelay[chosen]);
    t->setReleaseTime(simTime().dbl());

    EV << getName() << " sending task " << tc.taskId
       << " -> fog[" << chosen << "] (W=" << tc.workload
       << " Qs=" << tc.quantumSuitability << ")\n";

    t->setTimestamp(simTime());
    send(t, "out", chosen);
    emit(msgSentSignal, 1);
    tasksOffloaded++;
}

void IoTDevice::finish()
{
    int total = deadlinesMet + deadlinesMissed;
    double hitRatio = (total > 0) ? (double)deadlinesMet / total : 0.0;
    recordScalar("hitRatio", hitRatio);
    recordScalar("tasksLocal", tasksProcessedLocally);
    recordScalar("tasksOffloaded", tasksOffloaded);
    EV << getName() << " stats: local=" << tasksProcessedLocally
       << " offloaded=" << tasksOffloaded
       << " deadlinesMet=" << deadlinesMet
       << " deadlinesMissed=" << deadlinesMissed
       << " hitRatio=" << hitRatio << "\n";
}
