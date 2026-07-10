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
    int    targetServer;     // -1 = "let device decide"
    double cpuDemand;
    double deadline;
    int    sizeBytes;
    double releaseTime;      // time the task was generated at the IoT device
    double finishTime;       // set by fog server; time processing completed (ack)
    bool   isAck = false;    // true when the message is a completion ack from a fog server
}