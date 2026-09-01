# 定位思路

HCCL通信算子的任务编排完成后会下发到Device侧上异步执行，若此时HCCL下发的任务执行失败，会通过调用回调函数通知HCCL异常task信息（stream和taskId），HCCL会以此检索下发时的task信息，打印失败task的详细信息及其所在的算子信息。

针对如下产品，如果要跟踪task级信息，需要通过[HCCL_DIAGNOSE_ENABLE](../hccl_env/HCCL_DIAGNOSE_ENABLE.md)手动开启。

<!-- npu="A3" id4 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id4 -->
<!-- npu="910b" id1 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id1 -->

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
