//
// Task message — carries an offloaded computation from an IoT device to a fog
// server and the completion ack back. Fields extended for the heterogeneous
// CPU/GPU/QPU fog model (teacher feedback, Quantom.md phases 5-9):
//
//   workload            : required compute in MFLOP (million FLOP) -- the "F"
//   quantumSuitability  : Q_s in [0,1]; only high-Q_s tasks benefit from a QPU
//   dataSizeKb          : payload size (drives GPU copy time + tx energy)
//   netDelay            : one-way 5G network delay actually used (seconds)
//   execTime            : execution time on the assigned processor (set by fog)
//   energy              : energy spent executing on the assigned processor (fog)
//   processorType       : 0=CPU, 1=GPU, 2=QPU of the assigned fog server
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see http://www.gnu.org/licenses/.
//
message Task
{
    int    taskId;
    int    ownerDeviceId;
    int    targetServer;          // -1 = "let device decide"
    double cpuDemand;             // workload in MFLOP (alias of `workload`)
    double workload;              // required compute (MFLOP) -- the "F"
    double quantumSuitability;    // Q_s in [0,1]
    double dataSizeKb;            // payload size in KB
    double netDelay;              // one-way network delay used (s)
    double execTime;              // exec time on assigned processor (s, set by fog)
    double energy;                // exec energy on assigned processor (J, set by fog)
    int    processorType;         // 0=CPU, 1=GPU, 2=QPU (set by fog)
    double deadline;
    int    sizeBytes;
    double releaseTime;           // time the task was generated at the IoT device
    double finishTime;            // set by fog server; time processing completed (ack)
    bool   isAck = false;         // true when the message is a completion ack from a fog
}
