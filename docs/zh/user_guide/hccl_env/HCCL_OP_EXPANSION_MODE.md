# HCCL_OP_EXPANSION_MODE

## 功能描述

该环境变量用于配置通信算子的展开模式。

- **针对Ascend 950PR/Ascend 950DT** ：支持的配置如下，若设置了不支持的环境变量，系统报错。
  - **AI_CPU**：代表通信算子在AI CPU展开，Device侧根据硬件型号自动选择相应的调度器。

    该配置项仅支持Broadcast、Reduce、AllReduce、Scatter、ReduceScatter、ReduceScatterV、AllGather、AllGatherV、AlltoAll、AlltoAllV、AlltoAllVC算子。

    **注意**：

    图模式（Ascend IR）或者图捕获（aclgraph）场景，当通信算法采用AI CPU模式时，单卡上的并发图数量不能超过6个，否则可能会因AI CPU核被占满而导致通信阻塞。

  - **AIV**：代表通信算子在Vector Core展开，执行也在Vector Core。Ascend 950PR不支持此配置。
    - 该配置项仅支持对称组网、推理特性。
    - 该配置项不支持多通信域并行的场景（因为不支持多个通信域同时配置为“AIV”模式），否则可能会导致不可预期行为。您可以在初始化具有特定配置的通信域时，通过“HcclCommConfig”将某个通信域的算子展开模式设置为“AIV”。
    - 该配置项仅支持Broadcast、Reduce、AllReduce、ReduceScatter、Scatter、AllGather、AlltoAll、AlltoAllV算子，当前仅支持单机场景。
      - 针对Broadcast、Scatter、AllGather、AlltoAll、AlltoAllV算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、bfp16。
      - 针对Reduce、AllReduce、ReduceScatter算子，数据类型支持int8、int16、int32、float16、float32、bfp16。

    - 该配置项下，AllReduce、ReduceScatter、AllGather、AlltoAll算子支持控核能力，建议业务根据实际使用场景中计算算子与通信算子的并发情况进行Vector Core核数的配置。

      若业务编译分配的Vector Core核数无法满足算法编排的要求，HCCL会报错并提示所需要的最低Vector Core核数。

  - **CCU_MS**：代表通信算子在CCU展开，使用MS（Memory Slice）模式。Ascend 950PR不支持此配置。

    MS模式为与多个远端通信时，使用CCU片上Memory Slice作为中转，用于节省内存读写带宽，Memory Slice的特点是大小较小，但速度较快。

    当CCU资源不足时，系统会自动切换为AI_CPU模式。

  - **CCU_SCHED（默认值）**：代表通信算子在CCU展开，使用调度模式。

    调度模式指使用CCU作为调度器，向UB引擎调度UB WQE任务。调度模式下不使用CCU的片上MS，直接在两个rank间进行HBM到HBM的数据传输。

    针对单机通信场景的AllReduce、ReduceScatter、Reduce算子，当数据量超过一定值时，为防止性能下降，系统会自动切换为AI_CPU模式（该阈值并非固定，会根据算子运行模式及网络规模等因素有所调整）。

    当CCU资源不足时，系统会自动切换为AI_CPU模式。

- **针对Atlas A3 训练系列产品/Atlas A3 推理系列产品**：支持的配置如下，若设置了不支持的环境变量，使用默认值。
  - **AI_CPU（默认值）**：代表通信算子在AI CPU展开，Device侧根据硬件型号自动选择相应的调度器。

    在超节点内与超节点间支持全量通信算子。针对Reduce、ReduceScatter、ReduceScatterV、AllReduce算子，数据类型仅支持int8、int16、int32、float16、float32、bfp16，且reduce的操作类型仅支持sum、max、min。其他通信算子支持的数据类型可参见对应的集合通信接口参考。

    **注意：**

    - 图模式（Ascend IR）或者图捕获（aclgraph）场景，当通信算法采用默认的AI CPU模式时，单卡上的并发图数量不能超过6个，否则可能会因AI CPU核被占满而导致通信阻塞。
    - 此模式下通信功能依赖开放AI CPU用户态下发调度任务，存在一定的安全风险，需要用户自行确保自定义算子的安全可靠，防止恶意攻击行为。

  - **AICPU_CacheDisable**：关闭HCCL算子的AI CPU cache特性。

    AI CPU cache是指当同一个通信算子第二次执行时，HCCL会复用该算子首次执行的结果，从而节省展开开销。开启AI CPU cache特性会有一定的显存开销，因此，在通信数据量频繁变化的服务场景中，建议配置关闭cache机制，以减小显存开销。

  - **AIV**：代表通信算子在Vector Core展开，执行也在Vector Core。

    - 该配置项仅支持对称组网、推理特性。
    - 该配置项不支持多通信域并行的场景（因为不支持多个通信域同时配置为“AIV”模式），否则可能会导致不可预期行为。您可以在初始化具有特定配置的通信域时，通过“HcclCommConfig”将某个通信域的算子展开模式设置为“AIV”。
    - 该配置项仅支持Broadcast、AllReduce、ReduceScatter、AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子。
      - 针对Broadcast算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16，仅支持超节点内的单机通信，仅支持单算子模式和Ascend IR图模式，不支持多机和跨超节点间通信。
      - 针对AllReduce算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min，仅支持超节点内的单机/多机通信，不支持跨超节点间通信。
      - 针对ReduceScatter算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min，仅支持超节点内的单机/多机通信，不支持跨超节点间通信。
      - 针对AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16，仅支持超节点内的单机/多机通信，不支持跨超节点间通信。

    - 针对Broadcast、AllReduce、ReduceScatter、AllGather、AlltoAll（单机通信场景）算子，当数据量超过一定值时，为防止性能下降，系统会自动切换为AI_CPU模式（该阈值并非固定，会根据算子运行模式、是否启动确定性计算及网络规模等因素有所调整）；针对AlltoAllV、AlltoAllVC、AlltoAll（多机通信场景）算子，当配置为AIV模式时，系统不会自动切换为AI_CPU模式，为避免性能劣化，当任意两个rank之间的最大通信数据量不超过1MB时，建议配置为AIV模式，否则请采用AI_CPU模式。
    - 该配置项下，集合通信支持控核能力，建议业务根据实际使用场景中计算算子与通信算子的并发情况进行Vector Core核数的配置。

      - 针对Broadcast算子，建议至少分配ranksize个vector核。
      - 针对AllReduce、ReduceScatter、AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子，建议最少分配max\(2, ranksize/20 + 1\)个vector核。

        若业务编译分配的Vector Core核数无法满足算法编排的要求，HCCL会报错并提示所需要的最低Vector Core核数。

    **注意：**

    算法编排展开位置设置为“AIV”时，若同时设置了[HCCL_DETERMINISTIC](HCCL_DETERMINISTIC.md)环境变量为“true”或“strict”开启了确定性计算，确定性计算的优先级更高，某些场景下“AIV”展开可能不生效。

- **针对Atlas A2 训练系列产品/Atlas A2 推理系列产品**：支持的配置如下，若设置了不支持的环境变量，使用默认值。
  - **HOST（默认值）**：代表通信算在Host侧CPU展开，Device侧根据硬件型号自动选择相应的调度器。
  - **HOST_TS**：代表通信算子在Host侧CPU展开，Host向Device的Task Scheduler下发任务，Device的Task Scheduler进行任务调度执行。
  - **AI_CPU**：代表通信算子在AI CPU展开，Device侧根据硬件型号自动选择相应的调度器。

    该配置项仅支持AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子。

    **注意：**

    图模式（Ascend IR）或者图捕获（aclgraph）场景，当通信算法采用AI CPU模式时，单卡上的并发图数量不能超过6个，否则可能会因AI CPU核被占满而导致通信阻塞。

  - **AIV**：代表通信算子在Vector Core展开，执行也在Vector Core。

    - 该配置项仅支持对称组网、推理特性。
    - 该配置项不支持多通信域并行的场景（因为不支持多个通信域同时配置为“AIV”模式），否则可能导致不可预期行为。您可以在初始化具有特定配置的通信域时，通过“HcclCommConfig”将某个通信域的算子展开模式设置为“AIV”。
    - 该配置项仅支持Broadcast、AllReduce、AlltoAll、AlltoAllV、AlltoAllVC、AllGather、ReduceScatter、AllGatherV、ReduceScatterV算子。
      - 针对Broadcast算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16，仅支持单机场景8卡以内的单算子模式。
      - 针对AllReduce算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min。
      - 针对AlltoAll、AlltoAllV、AlltoAllVC算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16。针对AlltoAllV、AlltoAllVC算子，仅支持单机场景；针对AlltoAll算子的图模式运行方式，仅支持单机场景。
      - 针对AllGather算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16。针对该算子的图模式运行方式，仅支持单机场景。
      - 针对ReduceScatter算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min。针对该算子的图模式运行方式，仅支持单机场景。
      - 针对AllGatherV算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16，仅支持单算子模式。
      - 针对ReduceScatterV算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min。

    - 该配置项下，集合通信支持控核能力，建议业务根据实际使用场景中计算算子与通信算子的并发情况进行Vector Core核数的配置。

      - 针对AllReduce、ReduceScatter、ReduceScatterV算子，建议最少分配24个核。
      - 针对Broadcast、AlltoAll、AlltoAllV、AlltoAllVC、AllGather、AllGatherV算子，建议最少分配16个核。

        若业务编译分配的Vector Core核数无法满足算法编排的要求，HCCL会报错并提示所需要的最低Vector Core核数。

    **注意：**

    - 算法编排展开位置设置为“AIV”时，若同时设置了[HCCL_DETERMINISTIC](HCCL_DETERMINISTIC.md)环境变量为“true”或“strict”开启了确定性计算，确定性计算的优先级更高，某些场景下“AIV”展开可能不生效。
    - 对于Atlas 200T A2 Box16异构子框，不支持跨框通信场景。

<cann-filter npu-type="310p">

- **针对Atlas 300I Duo 推理卡**：支持的配置如下，若设置了不支持的环境变量，使用默认值。
  - **HOST（默认值）**：代表通信算子在Host侧CPU展开，Device侧根据硬件型号自动选择相应的调度器。
  - **AI_CPU**：代表通信算子在AI CPU展开，Device侧根据硬件型号自动选择相应的调度器。
    - 仅支持单机单通信域场景。
    - 仅支持AllReduce算子，AllReduce算子支持的数据类型可参见HcclAllReduce接口。
    - 配置为“AI_CPU”后，通信算子不再支持profiling性能数据采集与分析功能。
    - 对于静态shape图，不支持此配置项，即不支持指定通信算子的展开模式为AI CPU。
</cann-filter>

## 配置示例

```bash
export HCCL_OP_EXPANSION_MODE="AI_CPU"
```

## 使用约束

- 若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclOpExpansionMode”参数配置了通信算子的展开模式，则以通信域粒度的配置优先。
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品的**推理特性：**

    **配置为“AIV”的场景下**，若通过“CTRL+C”方式强制结束进程，在msnpureport工具导出的Device侧日志文件中可能会出现Device访问非法地址的错误，日志关键词为“devmm_page_fault_d2h_query_flag”、“devmm_svm_device_fault”或“ipc_fault_msg_para_check”，如下所示，此种场景不会影响Device上卡的状态，不会影响后续新起任务的执行。

    ```text
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:22.646.254 [klogd.c:247][257382.266115] [ascend] [ERROR] [devmm] [devmm_page_fault_d2h_query_flag 810] <kworker/u16:2:14887,14887> Host page fault send message fail.(hostpid=2131021; devid=0; vfid=0; ret=-22; va=0x12c700300000; hostpid=2131021; devid=0; vfid=0)
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:22.646.284 [klogd.c:247][257382.266124] [ascend] [ERROR] [devmm] [devmm_svm_device_fault 468] <kworker/u16:2:14887,14887> Vm fault failed. (hostpid=2131021; devid=0; vfid=0; ret=64; fault_addr=0x12c700300000; start=0x12c700300000)
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:22.659.429 [klogd.c:247][257382.282181] [ascend] [ERROR] [tsdrv] [ipc_fault_msg_para_check 309] <swapper/3:0> Invalid node id. (devid=0; node_type=100; node_id=40; node_num=25)
    ................
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:24.874.211 [klogd.c:247][257384.473533] [ascend] [ERROR] [tsdrv] [tsdrv_hb_cq_callback 332] <kworker/0:0:20353> receive ts exception msg, call excep_code=0xb4060006, time=1722249204.850014098s, devid=0 tsid=0
    ```
