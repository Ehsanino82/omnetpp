#include <omnetpp.h>
#include "Task_m.h"
#include "ServerStatus_m.h"

using namespace omnetpp;

class FogServer : public cSimpleModule
{
  private:
    int myId;
    int numIoT;
    simtime_t nextFreeTime = SIMTIME_ZERO;
    double statusInterval;
    cMessage *statusTimer = nullptr;

    // Utilization tracking
    simtime_t totalBusyTime = SIMTIME_ZERO;
    long tasksProcessed = 0;

    simsignal_t msgReceivedSignal;
    simsignal_t respTimeSignal;
    simsignal_t queueLenSignal;

  protected:
    virtual void initialize() override;
    virtual void handleMessage(cMessage *msg) override;
    virtual void finish() override;
    void broadcastStatus();
    ~FogServer() { cancelAndDelete(statusTimer); }
};

Define_Module(FogServer);

void FogServer::initialize()
{
    myId = getIndex();
    numIoT = getParentModule()->par("numIoT");
    statusInterval = par("statusInterval").doubleValue();

    msgReceivedSignal = registerSignal("msgReceived");
    respTimeSignal    = registerSignal("respTime");
    queueLenSignal    = registerSignal("queueBacklog");

    statusTimer = new cMessage("statusTimer");
    scheduleAt(simTime() + statusInterval, statusTimer);
}

void FogServer::handleMessage(cMessage *msg)
{
    if (msg == statusTimer) {
        broadcastStatus();
        scheduleAt(simTime() + statusInterval, statusTimer);
        return;
    }

    Task *t = check_and_cast<Task *>(msg);
    emit(msgReceivedSignal, 1);

    double procTime = t->getCpuDemand();

    // FIFO single-server queue model
    simtime_t start  = std::max(simTime(), nextFreeTime);
    simtime_t finish = start + procTime;
    nextFreeTime = finish;

    totalBusyTime += procTime;
    tasksProcessed++;

    simtime_t responseTime = finish - t->getTimestamp();
    emit(respTimeSignal, responseTime);
    emit(queueLenSignal, std::max(0.0, (nextFreeTime - simTime()).dbl()));

    EV << getName() << " processing task " << t->getTaskId()
       << " (proc=" << procTime << "s, finish=" << finish << "s)\n";

    // Send the task back as a completion ack once processing finishes,
    // carrying the finish time so the IoT device can compute end-to-end delay.
    Task *ack = new Task("ack");
    ack->setTaskId(t->getTaskId());
    ack->setOwnerDeviceId(t->getOwnerDeviceId());
    ack->setTargetServer(myId);
    ack->setCpuDemand(t->getCpuDemand());
    ack->setDeadline(t->getDeadline());
    ack->setSizeBytes(t->getSizeBytes());
    ack->setReleaseTime(t->getReleaseTime());
    ack->setFinishTime(finish.dbl());
    ack->setIsAck(true);
    sendDelayed(ack, (finish - simTime()).dbl(), "out", msg->getArrivalGate()->getIndex());
    delete t;
}

void FogServer::broadcastStatus()
{
    double load = std::max(0.0, (nextFreeTime - simTime()).dbl());

    // Send status to every connected IoT device
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
    EV << getName() << " utilization=" << utilization
       << " tasksProcessed=" << tasksProcessed << "\n";
}
