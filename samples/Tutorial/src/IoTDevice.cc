#include <omnetpp.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "Task_m.h"
#include "ServerStatus_m.h"
#include "DqnPolicy.h"

using namespace omnetpp;

struct TaskConfig {
    int taskId, ownerDeviceId, sizeBytes, priority;
    double arrivalTime, cpuDemand, deadline, dataSizeKb, energyLocalEst;
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

    // Fog server loads (updated via ServerStatus messages)
    std::vector<double> fogLoads;

    // Local processing model
    double localCpuRate;
    double linkDelay;                       // one-way link delay used for greedy estimate
    double txEnergyPerKb;                   // transmit energy per KB when offloading
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
    void processLocally(const TaskConfig &tc);
    void recordCompletion(double e2eDelay, bool deadlineMet, double energy);
    virtual void finish() override;
    ~IoTDevice() { cancelAndDelete(releaseTimer); }
};

Define_Module(IoTDevice);

void IoTDevice::initialize()
{
    numFogServers = par("numFogServers");
    myId          = getIndex();
    decisionMode  = par("decisionMode").stdstringValue();
    localCpuRate  = par("localCpuRate").doubleValue();
    linkDelay     = par("linkDelay").doubleValue();
    txEnergyPerKb = par("txEnergyPerKb").doubleValue();

    msgSentSignal     = registerSignal("msgSent");
    e2eDelaySignal    = registerSignal("e2eDelay");
    deadlineHitSignal = registerSignal("deadlineHit");
    energySignal      = registerSignal("energy");

    fogLoads.resize(numFogServers, 0.0);

    if (decisionMode == "dqn") {
        std::string modelDir = par("modelDir").stdstringValue();
        try {
            dqnPolicy.load(modelDir);
            EV << getName() << " loaded DQN model from " << modelDir << "\n";
        } catch (const std::exception& e) {
            EV_WARN << getName() << " failed to load DQN model: " << e.what()
                    << ", falling back to roundRobin\n";
            decisionMode = "roundRobin";
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
        //         cpuDemand,deadline,priority,memoryMb,energyLocalEst
        std::getline(ss, cell, ','); tc.taskId         = std::stoi(cell);
        std::getline(ss, cell, ','); tc.ownerDeviceId  = std::stoi(cell);
        std::getline(ss, cell, ','); tc.arrivalTime    = std::stod(cell);
        std::getline(ss, cell, ',');                    /* taskType (unused) */
        std::getline(ss, cell, ','); tc.dataSizeKb     = std::stod(cell);
        std::getline(ss, cell, ','); tc.cpuDemand      = std::stod(cell);
        std::getline(ss, cell, ','); tc.deadline       = std::stod(cell);
        std::getline(ss, cell, ','); tc.priority       = std::stoi(cell);
        std::getline(ss, cell, ',');                    /* memoryMb (unused) */
        std::getline(ss, cell, ','); tc.energyLocalEst = std::stod(cell);
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
        double energy = txEnergyPerKb * (ack->getSizeBytes() / 1024.0);
        recordCompletion(e2e, met, energy);
        EV << getName() << " task " << ack->getTaskId() << " completed at fog["
           << ack->getTargetServer() << "] e2e=" << e2e << "s "
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

    if (decisionMode == "greedy") {
        // Compare estimated completion time of local vs each fog server:
        //   local: local_backlog + cpuDemand/localCpuRate
        //   fog_j: fog_backlog[j] + 2*linkDelay + cpuDemand  (fog rate = 1.0)
        std::vector<double> costs;
        costs.push_back(std::max(0.0, (localNextFreeTime - simTime()).dbl())
                        + tc.cpuDemand / localCpuRate);           // action 0 = local
        for (int j = 0; j < numFogServers; j++)
            costs.push_back(fogLoads[j] + 2.0 * linkDelay + tc.cpuDemand);

        // Find minimum cost, then break ties randomly so that when several
        // servers report equal (or stale/idle) backlog the load is spread out
        // instead of always hitting the first candidate.
        double bestCost = *std::min_element(costs.begin(), costs.end());
        std::vector<int> candidates;
        for (int a = 0; a < (int)costs.size(); a++)
            if (costs[a] <= bestCost + 1e-9)
                candidates.push_back(a);
        return candidates[intuniform(0, (int)candidates.size() - 1)];
    }

    if (decisionMode == "dqn" && dqnPolicy.isLoaded()) {
        // State: [cpuDemand, deadlineSlack, fogLoad0, ..., fogLoad4]
        std::vector<double> state;
        state.push_back(tc.cpuDemand);
        state.push_back(tc.deadline - simTime().dbl());
        for (int i = 0; i < numFogServers; i++)
            state.push_back(fogLoads[i]);
        return dqnPolicy.act(state);   // 0=local, 1..5=fog[0..4]
    }

    // roundRobin (default): always offload, cycle through fog servers
    int chosen = nextServer + 1;  // action 1..numFogServers
    nextServer = (nextServer + 1) % numFogServers;
    return chosen;
}

void IoTDevice::processLocally(const TaskConfig &tc)
{
    double localProcTime = tc.cpuDemand / localCpuRate;
    simtime_t start  = std::max(simTime(), localNextFreeTime);
    simtime_t finish = start + localProcTime;
    localNextFreeTime = finish;

    double e2eDelay = (finish - simTime()).dbl();
    bool met = finish.dbl() <= tc.deadline;
    tasksProcessedLocally++;

    recordCompletion(e2eDelay, met, tc.energyLocalEst);

    EV << getName() << " processing task " << tc.taskId
       << " LOCALLY (proc=" << localProcTime << "s, finish=" << finish << "s) "
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
        // Process locally
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
    t->setCpuDemand(tc.cpuDemand);
    t->setDeadline(tc.deadline);
    t->setSizeBytes(tc.sizeBytes);
    t->setReleaseTime(simTime().dbl());

    EV << getName() << " sending task " << tc.taskId
       << " -> fog[" << chosen << "] (cpu=" << tc.cpuDemand << ")\n";

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
