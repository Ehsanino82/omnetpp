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
        std::getline(ss, cell, ',');                    /* taskType (unused) */
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
    if (decisionMode == "local") {
        return 0;
    }

    if (decisionMode == "random") {
        return intuniform(0, numFogServers);   // 0=local, 1..numFog
    }

    // RoundRobin: dumb baseline — always offload, cycle through fog servers
    // (no execution-time / queue awareness, per teacher note).
    if (decisionMode == "roundRobin") {
        int chosen = nextServer + 1;
        nextServer = (nextServer + 1) % numFogServers;
        return chosen;
    }

    // Intelligent completion-time selection (teacher feedback / Quantom.md
    // phase 8-9): C = network_delay + queue_backlog + execution_time, pick min.
    if (decisionMode == "greedy") {
        std::vector<double> costs;
        costs.push_back(std::max(0.0, (localNextFreeTime - simTime()).dbl())
                        + localExecTime(tc));                          // action 0
        for (int j = 0; j < numFogServers; j++)
            costs.push_back(fogLinkDelay[j] + fogLoads[j] + estimateFogExec(j, tc));

        double bestCost = *std::min_element(costs.begin(), costs.end());
        std::vector<int> candidates;
        for (int a = 0; a < (int)costs.size(); a++)
            if (costs[a] <= bestCost + 1e-9)
                candidates.push_back(a);
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

        // Quantum-suitability mask (Quantom.md phase 4): a task with Qs below
        // theta must NOT be sent to a QPU fog. Restrict the action set so the
        // DQN only argmaxes over feasible placements.
        std::vector<int> allowed;
        allowed.push_back(0);                      // local always allowed
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
