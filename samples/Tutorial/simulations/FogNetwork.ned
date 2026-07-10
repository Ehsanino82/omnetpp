package tutorial.simulations;

simple IoTDevice
{
    parameters:
        @display("i=device/pocketpc_s");
        int numFogServers;
        string taskFile = default("tasks.csv");
        string decisionMode = default("roundRobin");  // local | random | roundRobin | greedy | dqn
        string modelDir = default("");                 // path to exported DQN weights
        double localCpuRate = default(1.0);            // local CPU speed factor
        double linkDelay @unit(s) = default(0.01s);    // one-way link delay (greedy estimate)
        double txEnergyPerKb = default(0.05);          // transmit energy per KB when offloading
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
        double statusInterval @unit(s) = default(0.1s);  // how often to broadcast load
        @signal[msgReceived](type=long);
        @signal[respTime](type=simtime_t);
        @signal[queueBacklog](type=double);
        @statistic[msgReceived](title="messages received"; record=count,vector);
        @statistic[respTime](title="processing time"; record=mean,max,vector);
        @statistic[queueBacklog](title="queue backlog (s)"; record=mean,max,vector);
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
        // Fully connect every IoT device to every fog server
        for i=0..numIoT-1, for j=0..numFog-1 {
            iot[i].out++ --> { delay = 10ms; } --> fog[j].in++;
            fog[j].out++ --> { delay = 10ms; } --> iot[i].in++;
        }
}
