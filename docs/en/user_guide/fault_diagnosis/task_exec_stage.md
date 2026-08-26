# Task Dispatch and Execution Stage

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T04:00:19.333Z pushedAt=2026-08-14T07:33:33.435Z -->

## Troubleshooting Approach

After communicator initialization and parameter plane link establishment are completed, HCCL performs task orchestration and dispatch of communication operators. During communication operator task orchestration, a notify synchronization mechanism is used before data communication to ensure that the peer is ready to receive data from the local end. Therefore, if any rank experiences a process hang or exit due to an exception, a network fault, or inconsistent communication operator calls, most ranks will encounter execution wait timeout. When such issues occur, the primary task for troubleshooting is to locate the fault point. The overall troubleshooting approach is shown in the following figure.

![Troubleshooting Approach for Task Dispatch and Execution Phase Errors](figures/task_exec_error_debug.png)

During the communication operator task dispatch and execution phase, HCCL provides the following DFX mechanisms to facilitate rapid issue locating:

- HCCL has a cluster heartbeat mechanism. When a rank node detects an exception, the exception is propagated to every node in the cluster through the heartbeat mechanism. Therefore, you can first search for heartbeat exception event information in the CANN logs of any node in the cluster. For details about the mechanism and log information, see [Cluster Heartbeat Mechanism](#cluster-heartbeat-mechanism).

- If no heartbeat exception event log is retrieved, check for inconsistent cluster behavior issues through task exception error information. For the troubleshooting method, see [task exception mechanism](#task-exception-mechanism).

## Cluster Heartbeat Mechanism

### Troubleshooting Approach

HCCL establishes independent maintenance and diagnostic links with adjacent ranks based on existing communicator information, thereby providing the cluster with single-point fault broadcast and diffusion capabilities. (Note: HCCL controls the number of maintenance and diagnostic links and the communication data volume, so users do not need to worry about performance loss on communication links.) This ensures that the plog of any rank contains the fault root node information. For the currently supported fault detection capabilities, see the following table.

| Priority | Exception Type | Status (Run Log - run/plog) | ExceptionType (Debug Log - debug/plog) | Criteria |
| --- | --- | --- | --- | --- |
| 1 | Network issue | ERROR CQE | Error cqe Occurred | Periodically polls ROCE driver retransmission timeout events and maps the remote IP through QPN. |
| 2 | Process hang | STUCK | Stuck Occurred | Polls the ingress/egress counts of all operators at intervals of 1/3 of HCCL_EXEC_TIMEOUT to analyze whether a hang has occurred. |
| 3 | Process exit | LOST | Heartbeat Lost Occurred | No heartbeat packet received from the remote end within 30 seconds. |

To control the number of printed entries, the current Cluster Exception ERROR log prints only the first three valid events, with the priority order being ERROR CQE \> STUCK \> LOST. If all heartbeat events need to be confirmed, search the logs in the run directory.

- After detecting an exception event, HCCL diffuses and forwards the information within the cluster. During operation, HCCL prints the received exception events to the run log.

    **The log format is:** [HeartbeatAbnormal]local rank[IP/ID]:crimer rank[IP/ID] status[exception event type]by informer rank[IP/ID].

  - HeartbeatAbnormal: Represents a heartbeat exception event.

  - local rank: Information about the current node.

  - crimer rank: Information about the root node.

  - status: Exception event type.

  - by informer rank: The reporter of the cluster fault.

    You can search using the keyword "HeartbeatAbnormal" combined with the status value. The following shows a log example:

    ```text
    [INFO] HCCL(686,python):2025-10-23-07:52:59.191.363 [heartbeat.cc:951] [8970][TaskExecStage][HeartbeatAbnormal]local rank [127.10.0.1/1]: crimer rank [127.10.0.2/2] by informer rank [127.10.0.3/3]
    ```

- If an operator execution error report occurs subsequently and the task exception callback function is invoked to notify HCCL, HCCL infers the most likely single point of failure based on the received exception events and timeout configurations such as HCCL_EXEC_TIMEOUT, and prints the result in the ERROR log.

    **The log format is**: \[TaskExecStage\]\[HeartbeatAbnormal\]Cluster Exception Location\[IP/ID\], Arrival Time:\[Day Mon DD HH:MM:SS YYYY\], Discoverer:\[IP/ID\], ExceptionType:\[Exception Type\], Possible Reason:possible cause.

  - \[TaskExecStage\]\[HeartbeatAbnormal\]: indicates that the cluster fault occurred during the operator execution stage and is a heartbeat exception event.

  - Cluster Exception Location: Location where the cluster fault occurred.

  - Arrival Time: Time when the cluster fault occurred.

  - Discoverer: Node that discovered the cluster fault.

  - ExceptionType: Exception type of the cluster fault.

  - Possible Reason: Possible cause of the cluster fault.

Users can search for the keyword "HeartbeatAbnormal". A log example is as follows:

    ```text
    [ERROR]HCCL(835695,all_reduce_test):2025-10-23-17:28:06.049.385[task_exception_handler.cc:610][835695][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 17:25:58 2025], Discoverer:[127.10.0.1/2], ExceptionType:[Heartbeat Lost Occurred], Possible Reason:1. Process has exited, 2. Network Disconnected
    ```

If no exception event accompanies the timeout, the issue may be caused by cluster behavior consistency problems. Prioritize troubleshooting factors such as scripts, versions, and datasets. If necessary, enable the HCCL_ENTRY_LOG_ENABLE environment variable to perform operator-level behavior tracking.

> [!NOTE]Note
>
> 1. If the training/inference task is killed before the notify timeout, or if the task exception mechanism fails to call the callback function to notify HCCL in time for some reason, HCCL does not print exception information. Users can still locate the root node by analyzing exception events recorded in the run log during system operation. In this case, exception events must be identified. Generally, LOST/ERROR CQE events occurring near the time when the system hangs are considered the cause of the system halt. Note that the STUCK detection time is (1/3 to 2/3) * HCCL_EXEC_TIMEOUT.
> 2. Both network anomalies and process exits may simultaneously cause LOST and ERROR CQE events. Analyze these events in conjunction with the specific heartbeat events. For example, check whether both ranks report the peer as LOST.

### Example: Process Hang or Peer Heartbeat Loss

#### Symptoms

The keyword "Cluster Exception Location" appears in CANN logs, as shown below:

Peer heartbeat loss:

```text
[ERROR]HCCL(835695,all_reduce_test):2025-10-23-17:28:06.049.385[task_exception_handler.cc:610][835695][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 17:25:58 2025], Discoverer:[127.10.0.1/2], ExceptionType:[Heartbeat Lost Occurred], Possible Reason:1. Process has exited, 2. Network Disconnected
```

Process hang:

```text
[ERROR]HCCL(1219039,all_reduce_test):2025-10-23-21:05:09.859.568[task_exception_handler.cc:610] [1219039][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 21:03:19 2025], ExceptionType:[Stuck Occurred], Possible Reason:1.Host process is stuck, 2.Device task is stuck
```

#### Possible Causes and Troubleshooting Approach

The exception type and the node information where the exception occurred can be identified from the error log:

- Cluster Exception Location: Indicates the node information where the exception occurred.

- Arrival Time: Indicates the time when the exception was broadcast to the local end.

- ExceptionType: Exception type, including heartbeat loss (Heartbeat Lost Occurred), process hang (Stuck Occurred), network packet loss (Error CQE Occurred), and others.

- Possible Reason: Possible causes of the exception and troubleshooting approach:

  - Heartbeat Lost Occurred: Check whether the node where the exception occurred has already exited early before the exception was broadcast to the local end, or whether a network anomaly between nodes prevents connection.

  - Stuck Occurred: Check whether the service process on the node where the exception occurred is stuck on a certain node or has encountered a deadlock.

  - Error CQE Occurred: Check whether a CQE error has occurred on the node where the exception occurred.

## Task Exception Mechanism

### Troubleshooting Approach

After the task orchestration of an HCCL communication operator is complete, the tasks are dispatched to the device side for asynchronous execution. If the task dispatched by HCCL fails during execution, a callback function is invoked to notify HCCL of the exception task information (stream and taskId). HCCL then uses this information to retrieve the task details recorded at dispatch time and prints the detailed information of the failed task along with the operator to which it belongs. For Atlas A3 Training Series/Atlas A3 Inference Series and Atlas A2 Training Series/Atlas A2 Inference Series, task-level tracking must be manually enabled through [HCCL_DIAGNOSE_ENABLE](../hccl_env/HCCL_DIAGNOSE_ENABLE.md).

In this case, the key log entries for task exception printed in the CANN log are **"Task run failed"** or **"TaskExecStage"**, as shown below:

```text
[ERROR] HCCL(2111667,all_reduce_test):2025-10-24-11:18:29.597.044 [task_exception_handler.cc:908] [2111667][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[2], taskID[21], tag[AllReduce_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
[ERROR] HCCL(2111667,all_reduce_test):2025-10-24-11:18:29.597.054 [task_exception_handler.cc:771] [2111667][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[2].
[ERROR] HCCL(2111667,all_reduce_test):2025-10-24-11:18:29.597.083 [task_exception_handler.cc:704] [2111667][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.490.253], deviceId[2], index[21], count[256], reduceType[sum], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
```

First, the key information of the communication operator can be identified from the task exception information:

- base information: The stream and task ID where the HCCL operator resides, as well as the operator tag. The HCCL operator that reported the error can be identified based on the tag.

- groupRank information: The communicator name (group), the size of the communicator (rankSize), and the rankId of the current card within the communicator.

- opData information: The input parameter information of the current operator, including the deviceId where it resides, the index of the operator within the communicator (index), the data volume (count), the reduce type (reduceType), and the addresses of the source (src) and destination (dst).

Generally, only the following two types of tasks may fail:

- Notify: Commonly occurs when waiting for a remote peer times out during the operator execution phase.

- SDMA: Typically occurs in scenarios such as HCCS link anomalies or multi-bit ECC errors, and may also be triggered with a low probability during a remote core dump.

### Notify Wait Timeout (EI0002)

Since collective communication is a globally coordinated behavior within a communicator, if the communication operators, data volumes, etc. issued among ranks within the communicator are inconsistent, execution timeout may occur due to task mismatch among ranks. Alternatively, if one of the ranks encounters another error, the other ranks will wait for the errored rank to time out and consequently fail. The overall troubleshooting approach is as follows:

![Notify-wait timeout error troubleshooting approach](figures/notify_wait_timeout_debug.png)

#### Confirming the Locations of All Rank Nodes in the Communication Domain

First, it is necessary to confirm the node processes where all ranks in the communicator reside. Since HCCL prints default logs when a communicator is created, the communicator name from the error information can be used to locate the node processes where all ranks in the communicator reside. Search in the log directories of all job nodes using `grep -r "Entry-" run/plog/ | grep "communication_domain_name"`, for example:

```bash
grep -r "Entry-" run/plog/ | grep "127.10.0.1%enp_60000_0_1761275812718970"
```

The search results are as follows:

```text
run/plog/plog-2111667_20251024111652406.log:[INFO] HCCL(2111667,all_reduce_test):2025-10-24-11:16:52.724.374 [op_base.cc:1292] [2111667]Entry-HcclCommInitRootInfoInner:ranks[4], rank[2], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[2]
run/plog/plog-2111668_20251024111652406.log:[INFO] HCCL(2111668,all_reduce_test):2025-10-24-11:16:52.725.226 [op_base.cc:1292] [2111668]Entry-HcclCommInitRootInfoInner:ranks[4], rank[3], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[3]
run/plog/plog-2111665_20251024111652405.log:[INFO] HCCL(2111665,all_reduce_test):2025-10-24-11:16:52.719.213 [op_base.cc:1292] [2111665]Entry-HcclCommInitRootInfoInner:ranks[4], rank[0], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[0]
run/plog/plog-2111666_20251024111652405.log:[INFO] HCCL(2111666,all_reduce_test):2025-10-24-11:16:52.719.502 [op_base.cc:1292] [2111666]Entry-HcclCommInitRootInfoInner:ranks[4], rank[1], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[1]
```

#### Checking Whether a Full Timeout Has Occurred on Other Ranks in the Communication Domain

1. If a rank in the communicator has other error reports, the cause of the error report on that rank must be troubleshot first.

2. If all ranks in the communicator report HCCL communication operator execution errors, it is necessary to check whether the operators, data volumes, and data types are consistent across all ranks in the communicator.

    In the following case, rank 0 under the same communication reports an error on the AllReduce operator, while rank 1 reports an error on the allgather operator. In this scenario, the root cause of inconsistent operator dispatch among different ranks within the same communicator must be further investigated from the service perspective.

    ```text
    rank0:
    [ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.235 [task_exception_handler.cc:908] [2111665][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[2], taskID[21], tag[AllReduce_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
    [ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.247 [task_exception_handler.cc:771] [2111665][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[0].
    [ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.283 [task_exception_handler.cc:704] [2111665][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.493.816], deviceId[0], index[21], count[256], reduceType[sum], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
    
    rank1:
    [ERROR] HCCL(2111666,all_reduce_test):2025-10-24-11:18:29.513.755 [task_exception_handler.cc:908] [2111666][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[2], taskID[21], tag[AllGather_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
    [ERROR] HCCL(2111666,all_reduce_test):2025-10-24-11:18:29.513.764 [task_exception_handler.cc:771] [2111666][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[1].
    [ERROR] HCCL(2111666,all_reduce_test):2025-10-24-11:18:29.513.787 [task_exception_handler.cc:704] [2111666][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.489.331], deviceId[1], index[21], count[256], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
    ```

3. If the operators and data volumes dispatched within the communicator are consistent, check whether the error report time interval between ranks in the communicator exceeds the timeout period configured by HCCL_EXEC_TIMEOUT, which defaults to 1836 seconds.

    In the following case, both ranks reported errors for the allreduce operator in the communicator 127.10.0.1%enp_60000_0_1761275812718970, but the error times differed by 5 minutes and 40 seconds, while the currently configured HCCL_EXEC_TIMEOUT was only 300 seconds. As a result, both ranks eventually timed out within the timeout duration.

    ```text
    rank0:
    [ERROR] HCCL(2111665,all_reduce_test):2025-10-24-22:03:14.946.261 [task_exception_handler.cc:908] [2111665][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[2], taskID[21], tag[AllReduce_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
    [ERROR] HCCL(2111665,all_reduce_test):2025-10-24-22:03:14.946.269 [task_exception_handler.cc:771] [2111665][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[0].
    [ERROR] HCCL(2111665,all_reduce_test):2025-10-24-22:03:14.946.310 [task_exception_handler.cc:704] [2111665][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.493.816], deviceId[0], index[21], count[256], reduceType[sum], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
    
    rank1:
    [ERROR] HCCL(2111666,all_reduce_test):2025-10-24-22:08:58.890.365 [task_exception_handler.cc:908] [2111666][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[2], taskID[21], tag[AllReduce_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
    [ERROR] HCCL(2111666,all_reduce_test):2025-10-24-22:08:58.890.383 [task_exception_handler.cc:771] [2111666][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[1].
    [ERROR] HCCL(2111666,all_reduce_test):2025-10-24-22:08:58.890.392 [task_exception_handler.cc:704] [2111666][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.489.331], deviceId[1], index[21], count[256], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
    ```

    If the timeout duration is exceeded, it is necessary to troubleshoot from the service perspective whether the operator dispatch time interval between ranks exceeding the timeout duration is expected. If it is expected, an appropriate timeout duration can be specified through the HCCL_EXEC_TIMEOUT environment variable. The currently configured timeout duration can be retrieved from the log:

    ```bash
    grep -r "HCCL_EXEC_TIMEOUT" run/plog
    ```

#### Checking Whether the Communication Operator Dispatch Behavior Is Abnormal

For operator execution errors that are difficult to troubleshoot, enable the `HCCL_ENTRY_LOG_ENABLE` environment variable and reproduce the test case. After this environment variable is enabled, a log entry recording the input parameters of the communication operator dispatch is printed in the log files under the `log/run/plog` directory each time a communication operator is dispatched. If the test case fails, you can then check whether the communication operators dispatched on each rank are abnormal.

```text
[INFO] HCCL(3015875,python):2025-03-07-11:43:32.305.623 [hccl_opbase_atrace_info.cc:56][3017221]Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth_60000_0_1741318944927847], sendBuf[0x1241d3dcdc00], recvBuf[0x124702f40200], count[10746295], dataType[float32], op[sum], localRank[0], streamId[7],comm[0xfffe380078d0], deviceLogicId[0]
[INFO] HCCL(3015875,python):2025-03-07-11:43:32.306.413 [hccl_opbase_atrace_info.cc:56][3017183]Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth_60000_0_1741318944927847], sendBuf[0x1244bfffe000], recvBuf[0x1244bfffb400], count[1024], dataType[float32], op[sum], localRank[0], streamId[2],comm[0xfffe380078d0], deviceLogicId[0]
```

The preceding log indicates that the service dispatches two AllReduce operators in the communicator `127.10.0.1%eth_60000_0_1741318944927847`, but on two different streams: `streamId[7]` and `streamId[2]`. Multiple streams execute concurrently on the NPU. If the service does not correctly implement a synchronization mechanism for stream execution, these two AllReduce operators in the same communicator will execute concurrently. Since HCCL reuses communication operator resources within the same communicator, the concurrent execution of two AllReduce operators causes resources such as notify to be consumed incorrectly, resulting in unpredictable errors, such as execution timeout errors or precision anomalies.

### SDMA ERROR (EI0012)

#### Symptoms

The console log contains the error code EI0012 with the keyword "Execution_Error_SDMA", as shown below:

```text
[PID: 3480365] 2025-12-24-14:10:31.094.189 Execution_Error_SDMA(EI0012): SDMA memory copy task exception occurred. Remote rank: [4800]. Base information: [streamID:[351], taskID[5], taskType[Memcpy], tag[], AlgType(level 0-1-2):[null-null-null].]. Task information: [src:[0x12c180000000], dst:[0x12c041800000], size:[0x80], notify id:[0xffffffffffffffff], link type:[HCCS], remote rank:[0]]. Communicator information: [group:[], user define information[], rankSize[0], rankId[0]].
```

In addition, the CANN log contains the keyword "**fftsplus sdma error**", as shown below:

```text
[ERROR] RUNTIME(57096,python3.10):2025-05-12-20:55:44.705.025 [task_info.cc:1170]57288 PrintSdmaErrorInfoForFftsPlusTask:fftsplus task execute failed, dev_id=0, stream_id=50, task_id=21, context_id=18, thread_id=0, err_type=4[fftsplus sdma error]
[ERROR] RUNTIME(57096,python3.10):2025-05-12-20:55:44.705.031 [task_info.cc:1270]57288 TaskFailCallBackForFftsPlusTask:fftsplus streamId=50, taskId=21, context_id=18, expandtype=1, rtCode=0x715006c,[fftsplus task exception], psStart=0x0, kernel_name=not found kernel name, binHandle=(nil), binSize=0.
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.132 [task_exception_handler.cc:947] [57288][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[32], taskID[21], tag[AllGather_group_name_0], AlgType(level 0-1-2):[fullmesh-ring-H-D].
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.140 [task_exception_handler.cc:810] [57288][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[group_name_0], user define information[Unspecified], rankSize[8], rankId[0].
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.163 [task_exception_handler.cc:737] [57288][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-05-12-20:54:51.268.778], deviceId[0], index[4], count[3397632], src[0x12c25487ac00], dst[0x12c255000000], dataType[uint8].
```

#### Possible Causes

A page table translation failure occurred during the execution of an SDMA memory copy task. Specifically, the input or output address of the memory copy was not allocated memory, the allocated memory was smaller than the memory copy size, or the allocated memory had already been freed.

Common root causes include the following scenarios:

- After a communication operator is dispatched, the communicator is destroyed without performing stream synchronization to confirm that the communication operator has completed execution. Since destroying the communicator releases the HCCL buffer addresses used for collective communication, this causes a page table translation failure during SDMA memory copy.

    The time when the communicator is destroyed can be retrieved from the `run` directory of the CANN logs:

    ```bash
    grep -r "Entry-HcclCommDestroy" log/run/plog
    ```

- On Atlas A3 Training Series/Atlas A3 Inference Series products, a network link fault can also cause an SDMA ERROR. In this case, check the link status between the two ends.

- When an HCCL communication operator is called, the actual allocated memory size of the passed input or output address is smaller than the passed data volume Count.

### ERROR CQE Error Report (EI0013)

ERROR CQE in HCCL indicates a retransmission timeout of RoCE packets. When this error occurs, it is inevitably accompanied by a cluster hang that leads to a timeout. HCCL periodically polls the RoCE driver to obtain its events. Users can query whether an ERROR CQE error report has occurred through the **HcclGetCommAsyncError** API.

#### Symptoms

The error code EI0013 is printed in the console logs, with the keyword "Error ROCE CQE", as shown below:

```text
[PID: 3448331] 2025-12-04-21:59:08.232.310 Execution Error ROCE CQE(EI0013): An error CQE occurred during operator execution. Local information: server 127.0.0.1, device ID 0, device IP 127.10.0.1. Peer information: server 127.0.0.2, device ID 1, device IP 127.10.0.2.
Possible Cause: 1. The network between two devices is abnormal. For example, the network port is intermittently disconnected.2. The peer process exits abnormally in advance. As a result, the local end cannot receive the response from the peer end.
Solution: 1. Check whether the network devices between the two ends are abnormal.2. Check whether the peer process exits first. If yes, check the cause of the process exit.
```

In addition, the keyword "error cqe status" appears in the CANN logs, as shown below:

```text
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.612 [hns_roce_lite.c:630]hns_roce_lite_handle_error_cqe(630) : error cqe status: 0x15
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.622 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000000): 0x00041580
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.627 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000001): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.630 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000002): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.634 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000003): 0x1500047c
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.637 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000004): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.640 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000005): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.644 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000006): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.647 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000007): 0x00000000
[ERROR] HCCP(2040034,alltoall_test):2025-09-15-08:38:12.776.650 [ra_hdc_lite.c:794]tid:2040458,ra_hdc_lite_period_poll_cqe : [create][ra_hdc_period_poll]failed CQE status[12], wr[0]
[ERROR] HCCL(2040034,alltoall_test):2025-09-15-08:38:13.607.432 [heartbeat.cc:1229] [2040666][TaskExecStage][HeartbeatAbnormal][ROCE CQE ERROR]cqe error status[12], time:[2025-09-15 08:38:12.776654],localInfo{server[127.10.0.1],deviceId[127.10.0.1],deviceIp[127.11.0.1]}, remoteIP{server[127.10.0.2],deviceId[127.10.0.2],deviceIp[127.11.0.2]}
```

#### Possible Causes

The root cause of an ERROR CQE event is that after the local end sends packets to the peer, it does not receive an acknowledgment reply from the peer within a specified time period, which triggers an ERROR CQE error report on the local end. This indicates that the network channel between the local end and the peer is abnormal, or the peer has disconnected, or the connection quality is poor and unable to respond. In addition to network factors, abnormal process exit on the peer side can also prevent the local end from receiving a reply, thereby causing an ERROR CQE error report.

#### Solution

First, identify the remote end of the ERROR CQE based on the error information.

```text
[ERROR] HCCL(2040034,alltoall_test):2025-09-15-08:38:13.607.432 [heartbeat.cc:1229] [2040666][TaskExecStage][HeartbeatAbnormal][ROCE CQE ERROR]cqe error status[12], time:[2025-09-15 08:38:12.776654],localInfo{server[127.10.0.1],deviceId[127.10.0.1],deviceIp[127.11.0.1]}, remoteIP{server[127.10.0.2],deviceId[127.10.0.2],deviceIp[127.11.0.2]}
```

Here, localIP and remoteIP represent the device IPs of the local end and the remote end, respectively. Locate the compute node or log where the corresponding rank resides based on the hardware resource information.

1. Check for network issues. Use the hccn_tool to query whether there are any network port flapping records. The following result indicates that a port link-down event occurred at 10:13:50 2025. If a collective communication operator was being executed at that time, an ERROR CQE would be generated, and further investigation into the cause of the port flapping is required.

    ```bash
    $ hccn_tool -i 0 -link_stat -g
    [devid 0]current time        : Tue Oct 28 21:46:46 2025
    [devid 0]link up count       : 2
    [devid 0]link down count     : 1
    [devid 0]link change records :
    [devid 0]    Sun Oct  5 10:13:51 2025    LINK UP
    [devid 0]    Sun Oct  5 10:13:50 2025    LINK DOWN
    [devid 0]    Sun Oct  5 10:13:35 2025    LINK UP
    ```

2. Check whether the peer service process exited abnormally or had already entered the resource destruction process before the local ERROR CQE occurred. This can be determined by examining the peer's service logs or plog to confirm whether the abnormal exit time of the peer process precedes the local ERROR CQE occurrence.

3. If the HCCL_RDMA_TIMEOUT retransmission timeout and HCCL_RDMA_RETRY_CNT retry count configured in the environment variables are relatively small, ERROR CQE errors are likely to occur when the link status is poor. Increase these environment variable values accordingly.

Among them, status\[12\] indicates a RoCE packet retransmission timeout. Other status codes are extremely rare. If encountered, contact technical support.

### AIV Communication Operator Execution Failure

#### Symptoms

For Atlas A2 training products/Atlas A2 inference products, after AIV mode is enabled via export HCCL_OP_EXPANSION_MODE="AIV", HCCL implements the orchestration and execution of HCCL communication operators in kernel execution mode in certain scenarios. In such cases, if a communication operator execution exception occurs, the following critical log line is printed: "fault kernel_name=aiv_all_reduce_***", indicating that the HCCL AIV operator execution has failed:

```text
[ERROR] RUNTIME(699131,python):2025-04-24-21:54:17.707.236 [davinci_kernel_task.cc:1268]699131 PrintErrorInfoForDavinciTask:[INIT][DEFAULT]Aicore kernel execute failed, device_id=0, stream_id=2, report_stream_id=2, task_id=55873, flip_num=2073, fault kernel_name=aiv_all_reduce_***, fault kernel info ext=aiv_all_reduce_910b_bfloat16_t, program id=42, hash=9645272693770703471.
```

In addition, the same task exception information described above is also printed. The root cause of the task failure can still be analyzed using the troubleshooting approach in [Notify Wait Timeout (EI0002)](#notify-wait-timeout-ei0002), such as whether a full-scale timeout has occurred or whether a node in the cluster experienced an exception first.
