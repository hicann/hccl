# 任务下发执行阶段

## 定位思路

完成通信域初始化及参数面建链后，HCCL会进行通信算子的任务编排及下发，在通信算子的任务编排上，进行数据通信前会有notify同步机制来确保对端已准备好接收本端的数据，因此如果有rank由于某种异常导致进程卡死或退出、网络故障或调用的通信算子不一致，会导致大部分rank出现执行等待超时。遇到此类问题，定位的首要条件是找到故障点位置，整体定位思路如下图。

![任务下发执行阶段报错定位思路](figures/task_exec_error_debug.png)

HCCL在通信算子任务下发执行阶段提供了以下DFX机制来辅助问题快速定位：

- HCCL存在集群心跳机制，当某个rank节点发现异常时，会通过心跳机制扩散到集群的每个节点上，因此可以先在集群中的任意节点的CANN日志中检索是否有心跳的异常事件信息打印，机制说明及日志信息可以参考[集群心跳机制](#集群心跳机制)。
- 若未检索到心跳的异常事件信息日志打印，可通过task exception报错信息排查是否有集群行为不一致问题，排查方法可参考[task exception机制](#task-exception机制)。

## 集群心跳机制

### 定位思路

HCCL会基于已有的通信域信息与邻近的rank建立起独立的维测链路，以此提供集群的单点故障广播扩散能力（注：HCCL会控制维测链路数量和通信数据量，用户不用担心其对通信链路造成的性能损失），使任意rank的plog日志中均包含故障根节点信息。当前支持的故障探测能力请参见下表。

| 优先级 | 异常类型 | status（运行日志-run/plog） | ExceptionType（调试日志-debug/plog） | 判断标准 |
| --- | --- | --- | --- | --- |
| 1 | 网络问题 | ERROR CQE | Error cqe Occurred | 定期轮询ROCE驱动重传超次事件，通过QPN映射远端IP |
| 2 | 进程卡死 | STUCK | Stuck Occurred | 每隔1/3的HCCL_EXEC_TIMEOUT时间轮询所有算子的入/出次数分析是否卡住 |
| 3 | 进程退出 | LOST | Heartbeat Lost Occurred | 30s时间内未收到远端的心跳报文 |

为了控制打印数量，当前Cluster Exception ERROR日志只打印有效事件的前三个，且优先级为ERROR CQE \> STUCK \> LOST，若需要确认全量的心跳事件，可在run目录日志中检索。

- HCCL在检测到异常事件后，会在集群中进行信息的扩散转发，HCCL在运行过程中会将收到的异常事件打印到run日志。

    **日志格式为：**\[HeartbeatAbnormal\]local rank\[IP/ID\]：crimer rank\[IP/ID\] status\[4异常事件类型\] by by informer rank\[IP/ID\]。

  - HeartbeatAbnormal：代表为心跳异常事件。
  - local rank：当前节点的信息。
  - crimer rank：根节点信息。
  - status：异常事件类型。
  - by by informer rank：集群故障上报者信息。

    用户可结合关键字“HeartbeatAbnormal”与status状态进行检索，日志示例如下：

    ```text
    [INFO] HCCL(686,python):2025-10-23-07:52:59.191.363 [heartbeat.cc:951] [8970][TaskExecStage][HeartbeatAbnormal]local rank [127.10.0.1/1]: crimer rank [127.10.0.2/2] status[LOST] by informer rank [127.10.0.3/3]
    ```

- 如果后续出现算子执行报错，并且调用了task exception回调函数通知了HCCL，HCCL会根据已经收到的异常事件，结合HCCL\_EXEC\_TIMEOUT等超时事件配置，推测出最可能的单点故障原因，打印在ERROR日志中。

    **日志格式为**：\[TaskExecStage\]\[HeartbeatAbnormal\]Cluster Exception Location\[IP/ID\], Arrival Time:\[星期 月 日 时:分:秒 年\], Discoverer:\[IP/ID\], ExceptionType:\[异常类型\], Possible Reason:可能原因。

  - \[TaskExecStage\]\[HeartbeatAbnormal\]：代表集群故障发生在算子执行阶段，为心跳异常事件。
  - Cluster Exception Location：集群故障发生位置。
  - Arrival Time：集群故障发生时间。
  - Discoverer：集群故障发现节点。
  - ExceptionType：集群故障的异常类型。
  - Possible Reason：集群故障发生的可能原因。

    用户可检索关键字“HeartbeatAbnormal”，日志示例如下所示：

    ```text
    [ERROR]HCCL(835695,all_reduce_test):2025-10-23-17:28:06.049.385[task_exception_handler.cc:610][835695][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 17:25:58 2025], Discoverer:[127.10.0.1/2], ExceptionType:[Heartbeat Lost Occurred], Possible Reason:1. Process has exited, 2. Network Disconnected
    ```

如果超时后未伴随异常事件，则有可能为集群行为一致性问题，请优先排查脚本、版本、数据集等因素，如果有需要，可以通过开启HCCL\_ENTRY\_LOG\_ENABLE环境变量进行算子级行为跟踪。

> [!NOTE]说明
>
> 1. 如果训练/推理任务被notify超时前提前杀掉，或task exception机制由于某种原因未及时调用callback函数通知HCCL，使HCCL没有打印异常信息。用户依然可以通过run日志中系统运行过程中的异常事件进行根节点定位。此时需要对异常事件进行甄别，一般我们认为，对于系统卡住时间附近的LOST/ERROR CQE事件即为导致系统停止的原因，而STUCK检测时间为（1/3\~2/3 ）\* HCCL\_EXEC\_TIMEOUT，需要注意。
> 2. 网络异常和进程退出均有可能会同时导致LOST和ERROR CQE事件，请结合心跳事件具体情况来看，比如：如果两端rank互报对端LOST。

### 示例：进程卡死或对端心跳丢失

#### 问题现象

在CANN日志中存在关键字"Cluster Exception Location"，如下所示：

对端心跳丢失：

```text
[ERROR]HCCL(835695,all_reduce_test):2025-10-23-17:28:06.049.385[task_exception_handler.cc:610][835695][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 17:25:58 2025], Discoverer:[127.10.0.1/2], ExceptionType:[Heartbeat Lost Occurred], Possible Reason:1. Process has exited, 2. Network Disconnected
```

进程卡死：

```text
[ERROR]HCCL(1219039,all_reduce_test):2025-10-23-21:05:09.859.568[task_exception_handler.cc:610] [1219039][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 21:03:19 2025], ExceptionType:[Stuck Occurred], Possible Reason:1.Host process is stuck, 2.Device task is stuck
```

#### 可能原因及定位思路

可从报错日志中识别异常类型及异常所在的节点信息：

- Cluster Exception Location：表示异常所在的节点信息。
- Arrival Time：表示异常广播到本端的时间。
- ExceptionType：异常类型，包括心跳丢失（Heartbeat Lost Occurred）、进程卡死（Stuck Occurred）、网络丢包（Error CQE Occurred）等。
- Possible Reason：异常可能的发生原因及排查思路：
  - Heartbeat Lost Occurred：排查异常所在的节点在异常广播到本端的时间是否已经提前退出或者节点间网络异常无法连接。
  - Stuck Occurred：排查异常所在的节点的业务进程是否卡死在某个节点或者发生了死锁。
  - Error cqe Occurred：排查异常所在的节点是否发生了cqe error。

## task exception机制

### 定位思路

HCCL通信算子的任务编排完成后会下发到Device侧上异步执行，若此时HCCL下发的任务执行失败，会通过调用回调函数通知HCCL异常task信息（stream和taskId），HCCL会以此检索下发时的task信息，打印失败task的详细信息及其所在的算子信息。针对Atlas A3 训练系列产品/Atlas A3 推理系列产品与Atlas A2 训练系列产品/Atlas A2 推理系列产品，如果要跟踪task级信息，需要通过[HCCL_DIAGNOSE_ENABLE](../hccl_env/HCCL_DIAGNOSE_ENABLE.md)手动开启。

此时在CANN日志中打印的task exception的关键日志为**"Task run failed"或"TaskExecStage"**，如下所示：

```text
[ERROR] HCCL(2111667,all_reduce_test):2025-10-24-11:18:29.597.044 [task_exception_handler.cc:908] [2111667][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[2], taskID[21], tag[AllReduce_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
[ERROR] HCCL(2111667,all_reduce_test):2025-10-24-11:18:29.597.054 [task_exception_handler.cc:771] [2111667][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[2].
[ERROR] HCCL(2111667,all_reduce_test):2025-10-24-11:18:29.597.083 [task_exception_handler.cc:704] [2111667][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.490.253], deviceId[2], index[21], count[256], reduceType[sum], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
```

首先可以从task exception信息中识别出通信算子关键信息：

- base information：HCCL算子所在的stream、taskid，以及算子的tag，可根据tag识别当前报错的HCCL算子。
- groupRank information：通信域名（group）、通信域的大小（rankSize）以及当前卡在通信域内的rankId。
- opData information：当前算子的入参信息，所在的deviceId，该通信域下的第几个算子（index）、数据量（count）、reduce类型（reduceType）以及输入（src）和输出（dst）的地址。

一般情况下，当前只有两种task可能会出现失败：

- Notify：常见于算子执行阶段等待远端超时。
- SDMA：一般在HCCS链路异常、多bit ecc等场景出现，也有较低概率在远端core dump时被触发。

### notify wait超时（EI0002）

由于集合通信是一个通信域内的全局协同行为，若通信域内rank之间下发的通信算子、数据量等不一致，则会由于rank之间的任务不匹配导致执行超时，或者其中一个rank发生了其他报错，则其他rank则会等待报错rank超时从而失败。整体的定位思路如下：

![notify-wait超时报错定位思路](figures/notify_wait_timeout_debug.png)

#### 确认通信域内全部rank节点所在位置

首先需要确认通信域内所有rank所在的节点进程，由于HCCL在通信域创建的时候会有默认日志打印，因此可以结合报错信息中的通信域名找到通信域内所有rank所在节点进程，在作业所有节点的log目录下检索`grep -r "Entry-" run/plog/ | grep "通信域名"`，如：

```bash
grep -r "Entry-" run/plog/ | grep "127.10.0.1%enp_60000_0_1761275812718970"
```

检索结果如下所示：

```text
run/plog/plog-2111667_20251024111652406.log:[INFO] HCCL(2111667,all_reduce_test):2025-10-24-11:16:52.724.374 [op_base.cc:1292] [2111667]Entry-HcclCommInitRootInfoInner:ranks[4], rank[2], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[2]
run/plog/plog-2111668_20251024111652406.log:[INFO] HCCL(2111668,all_reduce_test):2025-10-24-11:16:52.725.226 [op_base.cc:1292] [2111668]Entry-HcclCommInitRootInfoInner:ranks[4], rank[3], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[3]
run/plog/plog-2111665_20251024111652405.log:[INFO] HCCL(2111665,all_reduce_test):2025-10-24-11:16:52.719.213 [op_base.cc:1292] [2111665]Entry-HcclCommInitRootInfoInner:ranks[4], rank[0], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[0]
run/plog/plog-2111666_20251024111652405.log:[INFO] HCCL(2111666,all_reduce_test):2025-10-24-11:16:52.719.502 [op_base.cc:1292] [2111666]Entry-HcclCommInitRootInfoInner:ranks[4], rank[1], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[1]
```

#### 确认通信域内其他rank的行为，是否为全量超时

1. 若通信域内某个rank存在其他报错，则需要先排查对应rank报错的原因。
2. 若通信域内的全部rank都是HCCL通信算子执行报错，需要排查通信域内的所有rank的算子、数据量、数据类型是否一致。

    如下案例中，同一个通信下的rank0报错在AllReduce算子，而rank1报错在allgather算子，则需要从业务上进一步排查同一个通信域不同rank之间下发算子不一致的根因。

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

3. 若通信域内下发的算子、数据量均等一致，可排查通信域内rank之间的报错时间间隔是否超过了HCCL_EXEC_TIMEOUT配置的超时时间，默认为1836秒。

    如下案例中，两个rank都是通信域127.10.0.1%enp_60000_0_1761275812718970下的allreduce算子报错，但是报错的时间差了5分40秒，而当前设置的HCCL_EXEC_TIMEOUT超时时间仅为300秒，因此最终两个rank均在超时时间内等待超时。

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

    若超过了超时时间，需从业务上排查rank之间的算子下发时间间隔超过超时时间是否符合预期，若符合预期则可通过HCCL_EXEC_TIMEOUT环境变量指定合适的超时时间。可在log日志中检索当前配置的超时时间：

    ```bash
    grep -r "HCCL_EXEC_TIMEOUT" run/plog
    ```

#### 确认通信算子下发行为是否异常

若遇到较难排查的算子执行报错问题，可开启"HCCL_ENTRY_LOG_ENABLE"环境变量后，再复现一次用例，该环境变量使用后会在每次通信算子下发后，在log/run/plog目录下的日志文件中打印一次日志记录通信算子下发的入参信息，用例执行失败后便可排查每个rank上下发的通信算子是否存在异常。

```text
[INFO] HCCL(3015875,python):2025-03-07-11:43:32.305.623 [hccl_opbase_atrace_info.cc:56][3017221]Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth_60000_0_1741318944927847], sendBuf[0x1241d3dcdc00], recvBuf[0x124702f40200], count[10746295], dataType[float32], op[sum], localRank[0], streamId[7],comm[0xfffe380078d0], deviceLogicId[0]
[INFO] HCCL(3015875,python):2025-03-07-11:43:32.306.413 [hccl_opbase_atrace_info.cc:56][3017183]Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth_60000_0_1741318944927847], sendBuf[0x1244bfffe000], recvBuf[0x1244bfffb400], count[1024], dataType[float32], op[sum], localRank[0], streamId[2],comm[0xfffe380078d0], deviceLogicId[0]
```

如上日志表明业务在127.10.0.1%eth\_60000\_0\_1741318944927847通信域中下发两个AllReduce算子，但是下发在了两条不同的stream上，streamId\[7\]和streamId\[2\]，npu上多流并发执行，若业务上没有正确的实现流执行的同步机制，这两个同一个通信域下的AllReduce算子会并发执行，由于HCCL在同一个通信域下的通信算子资源复用，两个AllReduce算子并发执行会导致notify等资源被错误的消耗，因此会有无法预期的报错产生，如执行超时报错或者精度异常等。

### SDMA ERROR（EI0012）

#### 问题现象

在打屏日志中会有EI0012的错误码打印，关键字为"Execution_Error_SDMA"，如下所示：

```text
[PID: 3480365] 2025-12-24-14:10:31.094.189 Execution_Error_SDMA(EI0012): SDMA memory copy task exception occurred. Remote rank: [4800]. Base information: [streamID:[351], taskID[5], taskType[Memcpy], tag[], AlgType(level 0-1-2):[null-null-null].]. Task information: [src:[0x12c180000000], dst:[0x12c041800000], size:[0x80], notify id:[0xffffffffffffffff], link type:[HCCS], remote rank:[0]]. Communicator information: [group:[], user define information[], rankSize[0], rankId[0]].
```

且在CANN日志中存在关键字"**fftsplus sdma error**"，如下所示：

```text
[ERROR] RUNTIME(57096,python3.10):2025-05-12-20:55:44.705.025 [task_info.cc:1170]57288 PrintSdmaErrorInfoForFftsPlusTask:fftsplus task execute failed, dev_id=0, stream_id=50, task_id=21, context_id=18, thread_id=0, err_type=4[fftsplus sdma error]
[ERROR] RUNTIME(57096,python3.10):2025-05-12-20:55:44.705.031 [task_info.cc:1270]57288 TaskFailCallBackForFftsPlusTask:fftsplus streamId=50, taskId=21, context_id=18, expandtype=1, rtCode=0x715006c,[fftsplus task exception], psStart=0x0, kernel_name=not found kernel name, binHandle=(nil), binSize=0.
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.132 [task_exception_handler.cc:947] [57288][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[32], taskID[21], tag[AllGather_group_name_0], AlgType(level 0-1-2):[fullmesh-ring-H-D].
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.140 [task_exception_handler.cc:810] [57288][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[group_name_0], user define information[Unspecified], rankSize[8], rankId[0].
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.163 [task_exception_handler.cc:737] [57288][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-05-12-20:54:51.268.778], deviceId[0], index[4], count[3397632], src[0x12c25487ac00], dst[0x12c255000000], dataType[uint8].
```

#### 可能原因

在执行SDMA内存拷贝任务时发生了页表转换失败，也就是内存拷贝的输入或者输出地址未分配内存、分配的内存小于内存拷贝大小或者分配的内存已被释放。

常见的问题根因有以下场景：

- 下发通信算子后，未执行流同步确认通信算子执行完成就析构通信域，由于通信域析构时会释放集合通信的HCCL Buffer地址，因此会导致SDMA内存拷贝时页表转换失败。

    可以在CANN日志的run目录下检索通信域销毁的时间：

    ```bash
    grep -r "Entry-HcclCommDestroy" log/run/plog
    ```

- Atlas A3 训练系列产品/Atlas A3 推理系列产品下，网络链路故障也会导致SDMA ERROR，此时需要检查两端之间的链路状态。
- 调用HCCL通信算子时，传入的输入或输出地址实际分配的内存大小小于传入的数据量Count。

### ERROR CQE报错（EI0013）

ERROR CQE在HCCL中代表RoCE报文的重传超时，出现后必然会伴随集群卡死导致超时。HCCL会定期轮询RoCE驱动以获取其事件，用户可以通过接口**HcclGetCommAsyncError**进行查询是否有发生ERROR CQE报错。

#### 问题现象

在打屏日志中会有EI0013的错误码打印，关键字为"Error ROCE CQE"，如下所示：

```text
[PID: 3448331] 2025-12-04-21:59:08.232.310 Execution Error ROCE CQE(EI0013): An error CQE occurred during operator execution. Local information: server 127.0.0.1, device ID 0, device IP 127.10.0.1. Peer information: server 127.0.0.2, device ID 1, device IP 127.10.0.2.
Possible Cause: 1. The network between two devices is abnormal. For example, the network port is intermittently disconnected.2. The peer process exits abnormally in advance. As a result, the local end cannot receive the response from the peer end.
Solution: 1. Check whether the network devices between the two ends are abnormal.2. Check whether the peer process exits first. If yes, check the cause of the process exit.
```

且在CANN日志中存在关键字"error cqe status"，如下所示：

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

#### 可能原因

发生ERROR CQE的问题根因在于本端给对端发包后在指定的时间段内没有收到对端的确认回复，本端就会有ERROR CQE报错上报，此时表明本端和对端之间的网络通道出现异常或者对端已断开连接或者连接状态差，无法响应，除了网络因素外，对端的进程异常退出也会导致本端收不到回复从而有ERROR CQE报错。

#### 解决方法

首先可根据报错信息确认ERROR CQE远端。

```text
[ERROR] HCCL(2040034,alltoall_test):2025-09-15-08:38:13.607.432 [heartbeat.cc:1229] [2040666][TaskExecStage][HeartbeatAbnormal][ROCE CQE ERROR]cqe error status[12], time:[2025-09-15 08:38:12.776654],localInfo{server[127.10.0.1],deviceId[127.10.0.1],deviceIp[127.11.0.1]}, remoteIP{server[127.10.0.2],deviceId[127.10.0.2],deviceIp[127.11.0.2]}
```

其中，localIP和remoteIP分别代表了本端和远端的device ip，请基于硬件资源信息找到对应的rank所在计算节点或日志。

1. 排查是否有网络问题，可通过hccn\_tool工具查询是否有网口闪断记录，如下结果表示网口在10:13:50 2025时发生了端口断链，此时若有集合通信算子执行，则会有ERROR CQE产生，需要进一步排查网口闪断的原因。

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

2. 排查对端的业务进程在本端发生ERROR CQE时是否先异常退出或已进入资源销毁流程，可以通过观察对端的业务日志或者plog日志确认对端进程的异常退出时间是否在本端发生ERROR CQE前。
3. 若环境变量配置的HCCL_RDMA_TIMEOUT重传超时时间及HCCL_RDMA_RETRY_CNT重传次数比较小，在链路状态不佳时容易出现ERROR CQE错误，将环境变量调大即可。

其中，status\[12\]代表着RoCE报文重传超时，其他状态码极为少见，遇到后请联系技术支持。

### AIV通信算子执行失败

#### 问题现象

针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，在通过export HCCL_OP_EXPANSION_MODE="AIV"启用AIV模式之后，部分场景下HCCL会以kernel的执行方式实现HCCL通信算子的编排及执行，此时若通信算子执行异常，在日志中会有一行如下关键日志打印"fault kernel_name=aiv_all_reduce_***"表明当前为HCCL的aiv算子执行失败：

```text
[ERROR] RUNTIME(699131,python):2025-04-24-21:54:17.707.236 [davinci_kernel_task.cc:1268]699131 PrintErrorInfoForDavinciTask:[INIT][DEFAULT]Aicore kernel execute failed, device_id=0, stream_id=2, report_stream_id=2, task_id=55873, flip_num=2073, fault kernel_name=aiv_all_reduce_***, fault kernel info ext=aiv_all_reduce_910b_bfloat16_t, program id=42, hash=9645272693770703471.
```

此外也会有上述同样的task exception信息打印，仍可以通过[notify wait超时（EI0002）](#notify-wait超时ei0002)排查思路分析任务失败的根因，如是否全量超时、是否集群中存在某个先发生异常的节点等。
