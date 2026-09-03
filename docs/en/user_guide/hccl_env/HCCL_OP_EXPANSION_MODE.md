# HCCL_OP_EXPANSION_MODE

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-01T03:51:18.034Z pushedAt=2026-09-01T07:06:58.738Z -->

## Function

Configures the expansion mode of communication operators.

- **For Ascend 950PR/Ascend 950DT**: The supported configurations are listed below. Setting an unsupported value will trigger error reporting.
  - **AI_CPU**: Communication operators are expanded on the AI CPU, and the device automatically selects the corresponding scheduler based on the hardware model.

    This config supports Broadcast, Reduce, AllReduce, Scatter, ReduceScatter, ReduceScatterV, AllGather, AllGatherV, AlltoAll, AlltoAllV, AlltoAllVC, Send, Recv, and BatchSendRecv operators.

    **Note**

    - `AI_CPU` will be deprecated in a later version. Use `AICPU_TS` instead. In the current version, AICPU_TS is functionally identical to AI_CPU.
    - In graph (Ascend IR) or graph capture (aclgraph) use cases, when the communication algorithm uses AI CPU, the number of concurrent graphs on a single device cannot exceed 6. Otherwise, communication may be blocked due to AI CPU cores being fully occupied.

  - **AICPU_TS (default)**: Communication operators are expanded on the AI CPU and scheduled using the STARS scheduler.

    This config supports Broadcast, Reduce, AllReduce, Scatter, ReduceScatter, ReduceScatterV, AllGather, AllGatherV, AlltoAll, AlltoAllV, AlltoAllVC, Send, Recv, and BatchSendRecv operators.

    **Note**

    In graph (Ascend IR) or graph capture (aclgraph) use cases, when the communication algorithm uses AI CPU, the number of concurrent graphs on a single device cannot exceed 6. Otherwise, communication may be blocked due to AI CPU cores being fully occupied.

  - **AIV**: Communication operators are expanded and executed on the Vector Core.
    - This config supports only symmetric networking and inference features.
    - This config does not support parallel communicators (because multiple communicators cannot be simultaneously configured in `AIV` mode). Otherwise, unpredictable behavior may occur. You can set the operator expansion mode of a specific communicator to `AIV` through `HcclCommConfig` when initializing a communicator with specific configurations.
    - This configuration item supports only the Broadcast, Reduce, AllReduce, ReduceScatter, Scatter, AllGather, AlltoAll, AlltoAllV, Send, and Recv operators.
      - For Broadcast, Scatter, AllGather, AlltoAll, and AlltoAllV operators, the supported data types are int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, and bfp16.
      - For Reduce, AllReduce, and ReduceScatter operators, the supported data types are int8, int16, int32, float16, float32, and bfp16.

    - Under this config, AllReduce, ReduceScatter, AllGather, and AlltoAll operators support core control. Set the number of Vector Cores based on the concurrency of computation operators and communication operators in actual use.

  - **CCU_MS**: Communication operators are expanded on the CCU, using CcuBuffer for memory read/write. Ascend 950PR does not support this config.

    In this mode, when the CCU communicates with multiple remote ends, CcuBuffer serves as a relay to save memory read/write bandwidth. CcuBuffer features a small size but high speed.

    When CCU resources are insufficient, the system automatically switches to AI_CPU mode.
    This config supports only the Broadcast, Reduce, AllReduce, ReduceScatter, and AllGather operators, and only in single-device use cases.
    For the Reduce, AllReduce, and ReduceScatter operators, the supported data types are int16, int32, float16, float32, and bfp16. For the data types supported by other communication operators, see the corresponding collective communication API reference.

  - **CCU_SCHED**: Communication operators are expanded on the CCU with scheduling enabled.

    This config uses the CCU as a scheduler to dispatch UB WQE tasks to the UB engine. CcuBuffer is not used, and data is transferred directly between two ranks from on-chip memory to on-chip memory.

    For AllReduce, ReduceScatter, and Reduce operators in single-node communication, when the data volume exceeds a specified threshold, the system automatically switches to AI_CPU mode to prevent performance degradation. (This threshold is not fixed and may vary depending on factors such as the operator running mode and network scale.)

    In this mode, the ReduceScatterV and AllGatherV operators support only single-server use cases.

    When CCU resources are insufficient, the system automatically switches to AI_CPU mode.

- **For Atlas A3 training products/Atlas A3 inference products**: The following configurations are supported. Setting an unsupported value leads to using the default value.
  - **AI_CPU (default)**: Communication operators are expanded on the AI CPU. The device automatically selects the corresponding scheduler based on the hardware model.

    All communication operators are supported within and across SuperPoDs. For Reduce, ReduceScatter, ReduceScatterV, and AllReduce operators, only int8, int16, int32, float16, float32, and bfp16 data types are supported, and the Reduce type supports only sum, max, and min. For data types supported by other communication operators, see the corresponding collective communication API reference.

    **Note**

    - In graph (Ascend IR) or graph capture (aclgraph) use cases, when the communication algorithm uses the default AI CPU mode, the number of concurrent graphs on a single device cannot exceed 6. Otherwise, AI CPU cores may be fully occupied, causing communication blocking.
    - In this mode, communication depends on the open AI CPU to schedule and dispatch tasks in user mode, which poses certain security risks. You must ensure the security and reliability of custom operators to prevent malicious attacks.

  - **AICPU_CacheDisable**: Disables AI CPU cache for HCCL operators.

    AI CPU cache means that when the same communication operator is executed for the second time, HCCL reuses the result of its first execution, thereby saving expansion overhead. Enabling AI CPU cache incurs certain device memory overhead. Therefore, when the communication data volume changes frequently, we recommend disabling caching to reduce device memory overhead.

  - **AIV**: Communication operators are expanded and executed on the Vector Core.

    - This config supports only symmetric networking and inference features.
    - This config does not support parallel communicators (because multiple communicators cannot be simultaneously configured in `AIV` mode). Otherwise, unpredictable behavior may occur. You can set the operator expansion mode of a specific communicator to `AIV` through `HcclCommConfig` when initializing a communicator with specific configurations.
    - This config supports only Broadcast, AllReduce, ReduceScatter, AllGather, AlltoAll, AlltoAllV, and AlltoAllVC operators.
      - For the Broadcast operator, supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, bfp16, int64, uint64, and float64. Only single-node communication within a SuperPoD is supported. Only single-operator mode and Ascend IR graph mode are supported. Multi-node and cross-SuperPoD communication are not supported.
      - For the AllReduce operator, supported data types are int8, int16, int32, float16, float32, and bfp16. The Reduce type supports only sum, max, and min. Only single-node/multi-node communication within a SuperPoD is supported. Cross-SuperPoD communication is not supported.
      - For the ReduceScatter operator, supported data types are int8, int16, int32, float16, float32, and bfp16. The Reduce type supports only sum, max, and min. Only single-node/multi-node communication within a SuperPoD is supported. Cross-SuperPoD communication is not supported.
      - For AllGather, AlltoAll, AlltoAllV, and AlltoAllVC operators, the supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, bfp16, int64, uint64, and float64. Only single-node/multi-node communication within a SuperPoD is supported; cross-SuperPoD communication is not supported.

    - For Broadcast, AllReduce, ReduceScatter, AllGather, and AlltoAll (single-node communication) operators, when the data volume exceeds a specified threshold, the system automatically switches to AI_CPU mode to prevent performance degradation (this threshold is not fixed and may vary depending on factors such as the operator running mode, whether deterministic computation is enabled, and the network scale). For AlltoAllV, AlltoAllVC, and AlltoAll (multi-node communication) operators, when configured in AIV mode, the system does not automatically switch to AI_CPU mode. To avoid performance degradation, use AIV mode when the maximum communication data volume between any two ranks does not exceed 1 MB; otherwise, use AI_CPU mode.
    - Under this config, collective communication supports core control. Set the number of Vector Cores based on the concurrency of computation operators and communication operators in actual use.

      - For the Broadcast operator, you are advised to allocate at least *{ranksize}* Vector Cores.
      - For AllGather and non-deterministic ReduceScatter operators, you are advised to allocate at least *max\(2, ceil\(ranksize/20\)\)* Vector Cores.
      - For AllReduce, deterministic ReduceScatter, AlltoAll, AlltoAllV, and AlltoAllVC operators, you are advised to allocate at least *max\(2, ceil\(ranksize/20\)\)* Vector Cores, and the number of cores must be an even number (if the calculation result is odd, round it up to the next even number).

        If the number of allocated Vector Cores by cannot meet the requirements of algorithm orchestration, HCCL reports an error and prompts the minimum number of Vector Cores required.

    **Note**

    When the algorithm orchestration expansion is set to `AIV`, if [HCCL_DETERMINISTIC](HCCL_DETERMINISTIC.md) is also set to `true` or `strict` to enable deterministic computation, deterministic computation takes higher priority, and AIV expansion may not take effect in certain cases.

- **For Atlas A2 training products/Atlas A2 inference products**: The supported configurations are listed below. Setting an unsupported value leads to using the default value.
  - **HOST (default)**: Communication operators are expanded on the host-side CPU, and the device automatically selects the corresponding scheduler based on the hardware model.
  - **HOST_TS**: Communication operators are expanded on the host-side CPU. The host dispatches tasks to the device's task scheduler which schedules the tasks for execution.
  - **AI_CPU**: Communication operators are expanded on the AI CPU, and the device automatically selects the corresponding scheduler based on the hardware model.

    This config supports only AllGather, AlltoAll, AlltoAllV, and AlltoAllVC operators.

    **Note**

    In graph (Ascend IR) or graph capture (aclgraph) use cases, when the communication algorithm uses AI CPU, the number of concurrent graphs on a single device cannot exceed 6. Otherwise, communication may be blocked due to AI CPU cores being fully occupied.

  - **AIV**: Communication operators are expanded and executed on the Vector Core.

    - This config supports only symmetric networking and inference features.
    - This config does not support parallel communicators (because multiple communicator cannot be simultaneously configured in `AIV` mode). Otherwise, unexpected behavior may occur. You can set the operator expansion mode of a specific communicator to `AIV` through `HcclCommConfig` during the initialization of a communicator with specific configurations.
    - This config supports only Broadcast, AllReduce, AlltoAll, AlltoAllV, AlltoAllVC, AllGather, ReduceScatter, AllGatherV, and ReduceScatterV operators.
      - For the Broadcast operator, supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, bfp16, int64, uint64, and float64. Only the single-operator mode with no more than 8 devices on a single node is supported.
      - For the AllReduce operator, supported data types are int8, int16, int32, float16, float32, and bfp16. The Reduce type supports only sum, max, and min.
      - For AlltoAll, AlltoAllV, and AlltoAllVC operators, supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, bfp16, int64, uint64, and float64. For AlltoAllV and AlltoAllVC operators, only single-node use cases are supported. For the AlltoAll operator in graph mode, only single-node use cases are supported.
      - For the AllGather operator, supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, bfp16, int64, uint64, and float64. For this operator in graph mode, only single-node use cases are supported.
      - For the ReduceScatter operator, supported data types are int8, int16, int32, float16, float32, and bfp16. The Reduce type supports only sum, max, and min. For this operator in graph mode, only single-node use cases are supported.
      - For the AllGatherV operator, supported data types are int8, uint8, int16, uint16, int32, uint32, float16, float32, bfp16, int64, uint64, and float64. Only the single-operator mode is supported.
      - For the ReduceScatterV operator, supported data types are int8, int16, int32, float16, float32, and bfp16. The Reduce type supports only sum, max, and min.

    - Under this config, collective communication supports core control. Set the number of Vector Cores based on the concurrency of computation operators and communication operators in actual use.

      - For AllReduce, ReduceScatter, and ReduceScatterV operators, you are advised to allocate at least 24 cores.
      - For Broadcast, AlltoAll, AlltoAllV, AlltoAllVC, AllGather, and AllGatherV operators, you are advised to allocate at least 16 cores.

        If the number of allocated Vector Cores by cannot meet the requirements of algorithm orchestration, HCCL reports an error and prompts the minimum number of Vector Cores required.

    **Note**

    - When the algorithm orchestration expansion is set to `AIV`, if [HCCL_DETERMINISTIC](HCCL_DETERMINISTIC.md) is also set to `true` or `strict` to enable deterministic computation, deterministic computation takes higher priority, and AIV expansion may not take effect in certain cases.
    - For Atlas 200T A2 Box16, cross-subrack communication use cases are not supported.

<!-- npu="310p" id1 -->
- **For Atlas 300I Duo**: The following configurations are supported. Setting an unsupported value leads to using the default value.
  - **HOST (default)**: Communication operators are expanded on the host-side CPU, and the device automatically selects the corresponding scheduler based on the hardware model.
  - **AI_CPU**: Communication operators are expanded on the AI CPU, and the device automatically selects the corresponding scheduler based on the hardware model.
    - Only single-node single-communicator use cases are supported.
    - Only the AllReduce operator is supported. For the data types supported by the AllReduce operator, see the HcclAllReduce API.
    - When set to `AI_CPU`, communication operators no longer support profiling.
    - For static shape graphs, this config is not supported, meaning that the communication operator expansion mode can't be AI_CPU.
<!-- end id1 -->

## Configuration Example

```bash
export HCCL_OP_EXPANSION_MODE="AI_CPU"
```

## Constraints

- If you call the HCCL C API to initialize a communicator with specific configurations and configure the communication operator expansion mode through `hcclOpExpansionMode` of `HcclCommConfig`, the configuration at the communicator granularity takes precedence.
- For the **inference feature** of Atlas A2 training products/Atlas A2 inference products:

    **When AIV is in use**, if the process is forcibly terminated by pressing CTRL+C, errors indicating that the device accessed an illegal address may appear in the device-side log file exported by msnpureport. The log keywords are `devmm_page_fault_d2h_query_flag`, `devmm_svm_device_fault`, or `ipc_fault_msg_para_check`, as shown below. This case does not affect the NPU status or the execution of subsequent new tasks.

    ```text
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:22.646.254 [klogd.c:247][257382.266115] [ascend] [ERROR] [devmm] [devmm_page_fault_d2h_query_flag 810] <kworker/u16:2:14887,14887> Host page fault send message fail.(hostpid=2131021; devid=0; vfid=0; ret=-22; va=0x12c700300000; hostpid=2131021; devid=0; vfid=0)
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:22.646.284 [klogd.c:247][257382.266124] [ascend] [ERROR] [devmm] [devmm_svm_device_fault 468] <kworker/u16:2:14887,14887> Vm fault failed. (hostpid=2131021; devid=0; vfid=0; ret=64; fault_addr=0x12c700300000; start=0x12c700300000)
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:22.659.429 [klogd.c:247][257382.282181] [ascend] [ERROR] [tsdrv] [ipc_fault_msg_para_check 309] <swapper/3:0> Invalid node id. (devid=0; node_type=100; node_id=40; node_num=25)
    ................
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:24.874.211 [klogd.c:247][257384.473533] [ascend] [ERROR] [tsdrv] [tsdrv_hb_cq_callback 332] <kworker/0:0:20353> receive ts exception msg, call excep_code=0xb4060006, time=1722249204.850014098s, devid=0 tsid=0
    ```
