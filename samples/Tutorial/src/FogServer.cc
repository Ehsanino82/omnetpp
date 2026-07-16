#include <omnetpp.h>
#include <algorithm>
#include "Task_m.h"
#include "ServerStatus_m.h"
#include "ProcessorModel.h"

using namespace omnetpp;

class FogServer : public cSimpleModule
{
  private:
    int myId;
    int numIoT;
    simtime_t nextFreeTime = SIMTIME_ZERO;
    double statusInterval;
    cMessage *statusTimer = nullptr;

    // --- Processor specs (read from NED/ini, defaults = heterogeneous fog) ---
    ProcSpec spec;                 // CPU/GPU/QPU specs (see ProcessorModel.h)
    std::string procName;

    // Utilization tracking
    simtime_t totalBusyTime = SIMTIME_ZERO;
    simtime_t lastSampleTime = SIMTIME_ZERO;
    simtime_t lastBusyTime = SIMTIME_ZERO;
    long tasksProcessed = 0;

    simsignal_t msgReceivedSignal;
    simsignal_t respTimeSignal;
    simsignal_t queueLenSignal;
    simsignal_t execTimeSignal;
    simsignal_t energySignal;
    simsignal_t fogUtilizationSignal;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;
    void broadcastStatus();
    double computeEnergy(double execTime) const { return spec.power * execTime; }
    ~FogServer() { cancelAndDelete(statusTimer); }

  public:
    const char *procTypeName() const {
        return spec.type == PROC_QPU ? "QPU" :
               spec.type == PROC_GPU ? "GPU" : "CPU";
    }
};

Define_Module(FogServer);

void FogServer::initialize()
{
    myId = getIndex();
    numIoT = getParentModule()->par("numIoT");
    statusInterval = par("statusInterval").doubleValue();

    spec.type             = par("processorType").intValue();
    procName              = par("procName").stdstringValue();
    spec.throughput       = par("throughput").doubleValue();
    spec.power            = par("power").doubleValue();
    spec.efficiency       = par("efficiency").doubleValue();
    spec.gpuBandwidth     = par("gpuBandwidth").doubleValue();
    spec.gpuLaunchOverhead= par("gpuLaunchOverhead").doubleValue();
    spec.qpuOverhead      = par("qpuOverhead").doubleValue();
    spec.qpuTheta         = par("qpuTheta").doubleValue();

    msgReceivedSignal      = registerSignal("msgReceived");
    respTimeSignal         = registerSignal("respTime");
    queueLenSignal         = registerSignal("queueBacklog");
    execTimeSignal         = registerSignal("execTime");
    energySignal           = registerSignal("energy");
    fogUtilizationSignal   = registerSignal("fogUtilization");

    lastSampleTime = simTime();
    lastBusyTime   = SIMTIME_ZERO;

    EV << getName() << " [" << procName << " / " << procTypeName() << "]"
       << " throughput=" << spec.throughput << " MFLOP/s power=" << spec.power << "W\n";

    statusTimer = new cMessage("statusTimer");
    scheduleAt(simTime() + statusInterval, statusTimer);
}

// Execution-time prediction per processor type (Quantom.md phases 5-6, 8):
//   CPU:  E = F / P_CPU
//   GPU:  E = T_copy + F/(P_GPU*eta) + T_copyback   (F=workload, copy ~ data size)
//   QPU:  E = T_overhead + F / (P_QPU * Qs^2)        (Qs = quantum suitability)
// Implemented once in ProcessorModel.h and shared with IoTDevice.cc.

void FogServer::handleMessage(cMessage *msg)
{
    if (msg == statusTimer) {
        broadcastStatus();
        scheduleAt(simTime() + statusInterval, statusTimer);
        return;
    }

    Task *t = check_and_cast<Task *>(msg);
    emit(msgReceivedSignal, 1);

    double workload = t->getWorkload();
    double qs       = t->getQuantumSuitability();
    double dataBytes = (double)t->getSizeBytes();

    double procTime = computeExecTime(spec, workload, qs, dataBytes);
    double execEnergy = computeEnergy(procTime);

    // FIFO single-server queue model
    simtime_t start  = std::max(simTime(), nextFreeTime);
    simtime_t finish = start + procTime;
    nextFreeTime = finish;

    totalBusyTime += procTime;
    tasksProcessed++;

    simtime_t responseTime = finish - t->getTimestamp();
    emit(respTimeSignal, responseTime);
    emit(execTimeSignal, procTime);
    emit(energySignal, execEnergy);
    emit(queueLenSignal, std::max(0.0, (nextFreeTime - simTime()).dbl()));

    EV << getName() << " [" << procName << "/" << procTypeName() << "] task "
       << t->getTaskId() << " W=" << workload << " Qs=" << qs
       << " exec=" << procTime * 1000.0 << "ms E=" << execEnergy << "J\n";

    // Send the task back as a completion ack once processing finishes,
    // carrying the finish time + exec/energy so the IoT device can compute
    // end-to-end delay and total energy.
    Task *ack = new Task("ack");
    ack->setTaskId(t->getTaskId());
    ack->setOwnerDeviceId(t->getOwnerDeviceId());
    ack->setTargetServer(myId);
    ack->setWorkload(t->getWorkload());
    ack->setCpuDemand(t->getCpuDemand());
    ack->setQuantumSuitability(t->getQuantumSuitability());
    ack->setDataSizeKb(t->getDataSizeKb());
    ack->setSizeBytes(t->getSizeBytes());
    ack->setDeadline(t->getDeadline());
    ack->setReleaseTime(t->getReleaseTime());
    ack->setNetDelay(t->getNetDelay());
    ack->setExecTime(procTime);
    ack->setEnergy(execEnergy);
    ack->setProcessorType(spec.type);
    ack->setFinishTime(finish.dbl());
    ack->setIsAck(true);
    sendDelayed(ack, (finish - simTime()).dbl(), "out", msg->getArrivalGate()->getIndex());
    delete t;
}

void FogServer::broadcastStatus()
{
    // Instantaneous utilization over the last status interval (teacher: "utilization
    // rate of each fog at every moment"). Emitted as a vector at each tick.
    simtime_t now = simTime();
    simtime_t deltaT = now - lastSampleTime;
    double util = (deltaT > 0) ? (totalBusyTime - lastBusyTime).dbl() / deltaT.dbl() : 0.0;
    util = std::max(0.0, std::min(1.0, util));
    emit(fogUtilizationSignal, util);
    lastSampleTime = now;
    lastBusyTime   = totalBusyTime;

    double load = std::max(0.0, (nextFreeTime - now).dbl());

    // Send backlog to every connected IoT device
    int outSize = this->gateSize("out");
    for (int i = 0; i < outSize; i++) {
        ServerStatus *ss = new ServerStatus("serverStatus");
        ss->setServerId(myId);
        ss->setLoadSec(load);
        send(ss, "out", i);
    }
}

void FogServer::finish()
{
    double simDuration = simTime().dbl();
    double utilization = (simDuration > 0) ? (totalBusyTime.dbl() / simDuration) : 0.0;
    recordScalar("utilization", utilization);
    recordScalar("tasksProcessed", tasksProcessed);
    EV << getName() << " [" << procName << "/" << procTypeName() << "]"
       << " utilization=" << utilization
       << " tasksProcessed=" << tasksProcessed << "\n";
}
