# notify wait超时（EI0002）

由于集合通信是一个通信域内的全局协同行为，若通信域内rank之间下发的通信算子、数据量等不一致，则会由于rank之间的任务不匹配导致执行超时，或者其中一个rank发生了其他报错，则其他rank则会等待报错rank超时从而失败。整体的定位思路如下：

![notify-wait超时报错定位思路](figures/notify_wait_timeout_debug.png)

## 确认通信域内全部rank节点所在位置

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

## 确认通信域内其他rank的行为，是否为全量超时

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

## 确认通信算子下发行为是否异常

若遇到较难排查的算子执行报错问题，可开启"HCCL_ENTRY_LOG_ENABLE"环境变量后，再复现一次用例，该环境变量使用后会在每次通信算子下发后，在log/run/plog目录下的日志文件中打印一次日志记录通信算子下发的入参信息，用例执行失败后便可排查每个rank上下发的通信算子是否存在异常。

```text
[INFO] HCCL(3015875,python):2025-03-07-11:43:32.305.623 [hccl_opbase_atrace_info.cc:56][3017221]Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth_60000_0_1741318944927847], sendBuf[0x1241d3dcdc00], recvBuf[0x124702f40200], count[10746295], dataType[float32], op[sum], localRank[0], streamId[7],comm[0xfffe380078d0], deviceLogicId[0]
[INFO] HCCL(3015875,python):2025-03-07-11:43:32.306.413 [hccl_opbase_atrace_info.cc:56][3017183]Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth_60000_0_1741318944927847], sendBuf[0x1244bfffe000], recvBuf[0x1244bfffb400], count[1024], dataType[float32], op[sum], localRank[0], streamId[2],comm[0xfffe380078d0], deviceLogicId[0]
```

如上日志表明业务在127.10.0.1%eth_60000_0_1741318944927847通信域中下发两个AllReduce算子，但是下发在了两条不同的stream上，streamId\[7\]和streamId\[2\]，npu上多流并发执行，若业务上没有正确的实现流执行的同步机制，这两个同一个通信域下的AllReduce算子会并发执行，由于HCCL在同一个通信域下的通信算子资源复用，两个AllReduce算子并发执行会导致notify等资源被错误的消耗，因此会有无法预期的报错产生，如执行超时报错或者精度异常等。
