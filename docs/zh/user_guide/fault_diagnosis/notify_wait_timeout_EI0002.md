# 算子执行超时问题背景

一次卡间数据搬运包含**前同步、数据传输、尾同步**三个阶段（以 SDMA 为例）。

![op_execute_sequence](figures/op_execute_sequence.png)

* **前同步（Post / Wait Ack）**：Rank0读取Rank1数据前，需要等待Rank1发送Ack，表示数据已准备完成。
* **数据传输**：Rank0开始读取Rank1数据。
* **尾同步（Post / Wait DataSignal）**：Rank1等待Rank0发送DataSignal，表示数据已读取完成。

如果Rank1未执行Post Ack，Rank0的Wait Ack任务将一直等待直到超时。因此，**算子执行超时的本质是通信双方同步关系失配**。

执行超时常见原因包括：

* 某个Rank未下发通信算子
* 各Rank下发的通信算子不一致
* 数据量、数据类型、算法选择等参数不一致

# 算子执行超时 （EI0002）

当发生算子执行超时时，CANN日志中通常会出现`Task run failed`和`task_exception_handler.cc`等关键字：

```text
[ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.235 [task_exception_handler.cc:908] [2111665][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[2], taskID[21], tag[AllReduce_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
[ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.247 [task_exception_handler.cc:771] [2111665][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[0].
[ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.283 [task_exception_handler.cc:704] [2111665][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.493.816], deviceId[0], index[21], count[256], reduceType[sum], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
```

报错日志中的`group`字段为通信域名称（即`127.10.0.1%enp_60000_0_1761275812718970`），该参数将作用于后续所有排查步骤。

## 排查步骤

### 获取通信域内所有 Rank位置

根据通信域名称检索所有参与Rank，可执行命令`grep -rn "Entry-HcclCommInit" run/plog | grep "<通信域名称>"`检索到该通信域所有rank所在的位置。**示例中展示了通信域大小为`ranks[4]`以及`rank0~3`的run日志位置**，再根据run日志找到对应的debug日志位置。

```text
run/plog/plog-2111667_20251024111652406.log:[INFO] HCCL(2111667,all_reduce_test):2025-10-24-11:16:52.725.226 [op_base.cc:1292] [2111668]Entry-HcclCommInitRootInfoInner:ranks[4], rank[3], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[3]
run/plog/plog-2111665_20251024111652405.log:[INFO] HCCL(2111665,all_reduce_test):2025-10-24-11:16:52.724.374 [op_base.cc:1292] [2111667]Entry-HcclCommInitRootInfoInner:ranks[4], rank[2], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[2]
run/plog/plog-2111668_20251024111652406.log:[INFO] HCCL(2111668,all_reduce_test):2025-10-24-11:16:52.719.213 [op_base.cc:1292] [2111665]Entry-HcclCommInitRootInfoInner:ranks[4], rank[0], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[0]
run/plog/plog-2111666_20251024111652405.log:[INFO] HCCL(2111666,all_reduce_test):2025-10-24-11:16:52.719.502 [op_base.cc:1292] [2111666]Entry-HcclCommInitRootInfoInner:ranks[4], rank[1], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1761275812718970], deviceLogicId[1]
```

### 判断是否为全量超时

检查通信域内所有rank的debug日志，所有rank都报通信算子执行超时为全量超时，仅部分rank报通信算子超时时为非全量超时。超时问题定位思路如下图所示：

![op_execute_timeout_debug](figures/op_execute_timeout_debug.png)

#### 非全量超时

* **部分rank无异常日志**
  
  如果该通信域内的某个rank不存在debug日志或者debug日志中没有任务ERROR信息，说明该rank未下发通信算子。
  **该场景非HCCL问题，需要业务侧排查没有下发通信算子的原因。**
* **部分rank存在其他报错**

  如果该通信域内某个rank的debug日志有ERROR报错，但是不是通信算子执行超时的报错。 **优先分析该rank的首报错**，通常首报错为根因，其余rank的超时属于连带现象，该场景非HCCL问题。

#### 全量超时

如果该通信域内所有rank都报了通信算子超时，此时需进一步分析。

1. 检查通信参数是否一致。

   检查所有rank报错日志中的算子参数是否一致 ：通信算子、count、 dataType、reduceType、 通信域名称是否完全一致。**如果不一致，该场景非HCCL问题，需要业务侧进一步分析算子下发不一致的原因。**
   如下案例中，同一个通信域下的rank0报错在AllReduce算子，而rank1报错在Allgather算子，则需要从业务上进一步排查同一个通信域不同rank之间下发算子不一致的根因。

   ```text
   # rank0报错日志:
   tag[AllReduce_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
   [ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.247 [task_exception_handler.cc:771] [2111665][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[0].
   [ERROR] HCCL(2111665,all_reduce_test):2025-10-24-11:18:29.499.283 [task_exception_handler.cc:704] [2111665][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.493.816], deviceId[0], index[21], count[256], reduceType[sum], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].

   # rank1报错日志:
   tag[AllGather_127.10.0.1%enp_60000_0_1761275812718970], AlgType(level 0-1-2):[fullmesh-ring-NHR].
   [ERROR] HCCL(2111666,all_reduce_test):2025-10-24-11:18:29.513.764 [task_exception_handler.cc:771] [2111666][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[127.10.0.1%enp_60000_0_1761275812718970], user define information[], rankSize[4], rankId[1].
   [ERROR] HCCL(2111666,all_reduce_test):2025-10-24-11:18:29.513.787 [task_exception_handler.cc:704] [2111666][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-10-24-11:16:55.489.331], deviceId[1], index[21], count[256], src[0x12c0c0013000], dst[0x12c0c0014000], dataType[float32].
   ```

2. 检查算子下发时间。

    如果所有rank的算子参数一致，可排查通信域内rank之间算子的下发时间差，可比较各rank报错日志中的**timeStamp**。
    若不同rank的算子下发时间间隔超过 **HCCL_EXEC_TIMEOUT**（默认 1836 秒），所有rank都会等待超时，需从业务上排查rank之间的算子下发时间间隔超过超时时间是否符合预期；若符合预期则可通过HCCL_EXEC_TIMEOUT环境变量指定合适的超时时间。可在log日志中检索当前配置的超时时间：

    ```bash
    grep -r "HCCL_EXEC_TIMEOUT" run/plog
   ```

## 疑难场景

若遇到较难排查的算子执行报错问题，上述排查都满足符合预期时，可开启"HCCL_ENTRY_LOG_ENABLE"环境变量，再复现一次用例。该环境变量使用后会在每次通信算子下发后，在log/run/plog目录下的日志文件中打印一次日志记录通信算子下发的入参信息，用例执行失败后便可排查每个rank上下发的通信算子是否存在异常，开启方式：

```bash
export HCCL_ENTRY_LOG_ENABLE=1
```

开启后，每次通信算子下发都会打印详细入参：

```text
[INFO] HCCL(3015875,python):2025-03-07-11:43:32.305.623 [hccl_opbase_atrace_info.cc:56][3017221]Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth_60000_0_1741318944927847], sendBuf[0x1241d3dcdc00], recvBuf[0x124702f40200], count[10746295], dataType[float32], op[sum], localRank[0], streamId[7],comm[0xfffe380078d0], deviceLogicId[0]
[INFO] HCCL(3015875,python):2025-03-07-11:43:32.306.413 [hccl_opbase_atrace_info.cc:56][3017183]Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth_60000_0_1741318944927847], sendBuf[0x1244bfffe000], recvBuf[0x1244bfffb400], count[1024], dataType[float32], op[sum], localRank[0], streamId[2],comm[0xfffe380078d0], deviceLogicId[0]
```

可据此检查：

* 所有Rank是否均已下发通信算子
* 算子顺序是否一致
* count、dataType、reduceType是否一致
* 是否存在异常Stream

如上日志表明业务在127.10.0.1%eth_60000_0_1741318944927847通信域中下发两个AllReduce算子，但是下发在了两条不同的stream上，streamId[7]和streamId[2]，NPU上多流并发执行，若业务上没有正确的实现流执行的同步机制，这两个处于同一通信域下的AllReduce算子会并发执行，由于HCCL在同一个通信域下的通信算子资源复用，两个AllReduce算子并发执行会导致notify等资源被错误的消耗，因此会有无法预期的报错产生，如执行超时报错或者精度异常等。
