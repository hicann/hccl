# HCCL_OP_EXPANSION_MODE

## 功能描述

该环境变量用于配置通信算子的展开模式。

  <!-- npu="950" id2 -->
- **针对Ascend 950PR&950DT系列产品** ：支持的配置如下，若设置了不支持的环境变量，系统报错。

  - **AI_CPU**：代表通信算子在AI CPU展开，Device侧根据硬件型号自动选择相应的调度器。

    该配置项支持Broadcast、Reduce、AllReduce、Scatter、ReduceScatter、ReduceScatterV、AllGather、AllGatherV、AlltoAll、AlltoAllV、AlltoAllVC、Send、Recv、BatchSendRecv算子。

    **注意**：

    - "AI_CPU"配置将在后续版本废弃，使用“AICPU_TS”代替。当前版本中，“AICPU_TS”配置的功能与“AI_CPU”完全一致。
    - 图模式（Ascend IR）或者图捕获（aclgraph）场景，当通信算法采用AI CPU模式时，单卡上的并发图数量不能超过系统为集合通信预留的AI CPU上限（该上限默认为6，但实际可用量可能因其他业务占用而减少），否则可能会因AI CPU核被占满而导致通信阻塞。

  - **AICPU_TS（默认值）**：代表通信算子在AI CPU展开，使用STARS调度器调度运行。

    该配置项支持Broadcast、Reduce、AllReduce、Scatter、ReduceScatter、ReduceScatterV、AllGather、AllGatherV、AlltoAll、AlltoAllV、AlltoAllVC、Send、Recv、BatchSendRecv算子。Barrier算子在单算子模式下支持该配置项。

    **注意**：

    图模式（Ascend IR）或者图捕获（aclgraph）场景，当通信算法采用AI CPU模式时，单卡上的并发图数量不能超过系统为集合通信预留的AI CPU上限（该上限默认为6，但实际可用量可能因其他业务占用而减少），否则可能会因AI CPU核被占满而导致通信阻塞。

  - **AICPU_CacheDisable**：关闭HCCL算子的AI CPU cache特性。

    **警告**：`AICPU_CacheDisable`配置项在CANN 9.3.0版本标记为废弃，将在2028年3月30日之后发布的版本中删除。请改用`HCOMM_AICPU_TASK_CACHE_ENABLE=0`关闭cache。

    AI CPU cache是指当同一个通信算子第二次执行时，HCCL会复用该算子首次执行的结果，从而节省展开开销。开启AI CPU cache特性会有一定的显存开销，因此，在通信数据量频繁变化的服务场景中，建议配置关闭cache机制，以减小显存开销。

    **`AICPU_CacheDisable`与`HCOMM_AICPU_TASK_CACHE_ENABLE`的差异及优先级**：`AICPU_CacheDisable`是本环境变量的配置项，用于选择AI CPU展开并关闭cache；独立环境变量[HCOMM_AICPU_TASK_CACHE_ENABLE](HCOMM_AICPU_TASK_CACHE_ENABLE.md)仅控制cache开关，不改变算子的展开模式。

    显式配置`HCOMM_AICPU_TASK_CACHE_ENABLE`的有效值时，其值优先于本环境变量得出的cache开关结果；未配置时，沿用本环境变量的结果。下表中，`1`也可写为`true`，`0`也可写为`false`：

    | `HCCL_OP_EXPANSION_MODE` | `HCOMM_AICPU_TASK_CACHE_ENABLE` | cache开关 |
    | --- | --- | --- |
    | `AICPU_TS` | 未配置 | 开启 |
    | `AICPU_TS` | `1` | 开启 |
    | `AICPU_TS` | `0` | 关闭 |
    | `AICPU_CacheDisable` | 未配置 | 关闭 |
    | `AICPU_CacheDisable` | `1` | 开启 |
    | `AICPU_CacheDisable` | `0` | 关闭 |

  - **AIV**：代表通信算子在Vector Core展开，执行也在Vector Core。
    - 该配置项仅支持推理特性。
    - 该配置项不支持多通信域并行的场景（因为不支持多个通信域同时配置为“AIV”模式），否则可能会导致不可预期行为。您可以在初始化具有特定配置的通信域时，通过“HcclCommConfig”将某个通信域的算子展开模式设置为“AIV”。
    - 该配置项仅支持Broadcast、Reduce、AllReduce、ReduceScatter、Scatter、AllGather、AlltoAll、AlltoAllV、AlltoAllVC、Send、Recv算子。
      - 针对Reduce、AllReduce算子，数据类型支持int8、int16、int32、int64、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min，不支持prod。
      - 针对ReduceScatter算子，数据类型支持int8、int16、int32、int64、float16、float32、bfp16、hifloat8，reduce的操作类型仅支持sum、max、min，不支持prod。

    - 该配置项下，支持控核能力，建议业务根据实际使用场景中计算算子与通信算子的并发情况进行Vector Core核数的配置。

  - **CCU_MS**：代表通信算子在CCU展开，使用CcuBuffer进行内存读写。Ascend 950PR不支持此配置。

    此模式下，当CCU与多个远端通信时，使用CcuBuffer作为中转，用于节省内存读写带宽，CcuBuffer的特点是大小较小，但速度较快。

    此模式不支持inplace场景。

    HCCL算子在CCU资源不足时，会先尝试从CCU_MS模式回退到CCU_SCHED模式下发；若CCU_SCHED模式仍无法满足运行条件，再回退到AI CPU展开模式下发。

    该配置项仅支持Broadcast、Reduce、AllReduce、ReduceScatter、AllGather算子，当前仅支持单机场景。
    针对Reduce、AllReduce、ReduceScatter算子，数据类型仅支持int16、int32、float16、float32、bfp16。其他通信算子支持的数据类型可参见对应的集合通信接口参考。

    CCU资源扩容及额外实例的复用规则参见下文CCU_SCHED小节的说明。

  - **CCU_SCHED**：代表通信算子在CCU展开，使用调度模式。

    调度模式指使用CCU作为调度器，向UB引擎调度UB WQE任务。调度模式下不使用CcuBuffer，直接在两个rank间进行片上内存到片上内存的数据传输。

    此模式不支持inplace场景。

    该配置项支持Broadcast、AllGather、ReduceScatter、AllReduce、Reduce、Scatter、AlltoAll、AlltoAllV、AlltoAllVC、AllGatherV、ReduceScatterV算子。针对AllReduce、ReduceScatter、Reduce算子，数据类型仅支持int8、int16、int32、float16、float32、bfp16。针对ReduceScatterV算子，数据类型仅支持int16、int32、float16、float32、bfp16。其他通信算子支持的数据类型可参见对应的集合通信接口参考。

    HCCL的AllReduce、ReduceScatter和Reduce算子在单机通信场景中，当数据量超过一定阈值时，会自动回退到AI CPU展开模式下发，以避免性能下降。该阈值并非固定，会根据算子运行模式及网络规模等因素调整。

    此模式下，ReduceScatterV、AllGatherV算子仅支持单Server场景。

    AlltoAllV使用CCU调度算法执行时，若通信域已有CCU实例的容量不能满足其资源需求，HCCL会在设备剩余资源允许的情况下自动申请一份额外CCU实例，并使用该实例执行算子，无需额外配置。

    - 每个通信域最多创建一份额外CCU实例。原实例和额外实例分别承载完整的算子资源需求，容量不叠加使用。
    - 后续HCCL算子在同一通信域内申请CCU资源时，会优先复用已有的额外实例；若额外实例容量不足，则不再扩容，也不再检查原实例的容量，而是按既有流程回退。
    - 支持单算子流程和图模式的资源申请，图回放不重复创建额外实例。
    - 额外实例由HCCL管理，在调用[HcclCommDestroy](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/HcclCommDestroy.md)销毁通信域时通过回调释放，业务无需单独释放。
    - 扩容需使用配套版本的HCOMM库。需要创建额外实例但HCOMM不支持所需销毁回调能力时，不创建额外实例，沿用回退流程。

    HCCL算子无法获得满足需求的CCU资源时，会自动回退到AI CPU展开模式下发。
  <!-- end id2 -->

  <!-- npu="A3" id3 -->
- **针对Atlas A3系列产品**：支持的配置如下，若设置了不支持的环境变量，使用默认值。

  - **AI_CPU（默认值）**：代表通信算子在AI CPU展开，Device侧根据硬件型号自动选择相应的调度器。

    在超节点内与超节点间支持全量通信算子。针对Reduce、ReduceScatter、ReduceScatterV、AllReduce算子，数据类型仅支持int8、int16、int32、float16、float32、bfp16，且reduce的操作类型仅支持sum、max、min。其他通信算子支持的数据类型可参见对应的集合通信接口参考。

    **注意：**

    - 图模式（Ascend IR）或者图捕获（aclgraph）场景，当通信算法采用默认的AI CPU模式时，单卡上的并发图数量不能超过6个，否则可能会因AI CPU核被占满而导致通信阻塞。
    - 此模式下通信功能依赖开放AI CPU用户态下发调度任务，存在一定的安全风险，需要用户自行确保自定义算子的安全可靠，防止恶意攻击行为。

  - **AICPU_CacheDisable**：关闭HCCL算子的AI CPU cache特性。

    **警告**：`AICPU_CacheDisable`配置项在CANN 9.3.0版本标记为废弃，将在2028年3月30日之后发布的版本中删除。

    AI CPU cache是指当同一个通信算子第二次执行时，HCCL会复用该算子首次执行的结果，从而节省展开开销。开启AI CPU cache特性会有一定的显存开销，因此，在通信数据量频繁变化的服务场景中，建议配置关闭cache机制，以减小显存开销。

  - **AIV**：代表通信算子在Vector Core展开，执行也在Vector Core。

    - 该配置项仅支持对称组网、推理特性。
    - 该配置项不支持多通信域并行的场景（因为不支持多个通信域同时配置为“AIV”模式），否则可能会导致不可预期行为。您可以在初始化具有特定配置的通信域时，通过“HcclCommConfig”将某个通信域的算子展开模式设置为“AIV”。
    - 该配置项仅支持Broadcast、AllReduce、ReduceScatter、AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子。
      - 针对Broadcast算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16、int64、uint64、float64, 仅支持超节点内的单机/多机通信，仅支持单算子模式和Ascend IR图模式，不支持跨超节点间通信。
      - 针对AllReduce算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min，仅支持超节点内的单机/多机通信，不支持跨超节点间通信。
      - 针对ReduceScatter算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min，仅支持超节点内的单机/多机通信，不支持跨超节点间通信。
      - 针对AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16、int64、uint64、float64, 仅支持超节点内的单机/多机通信，不支持跨超节点间通信。

    - HCCL的Broadcast、AllReduce、ReduceScatter、AllGather和AlltoAll（单机通信场景）算子在数据量超过一定阈值时，会自动回退到AI CPU展开模式下发，以避免性能下降。该阈值并非固定，会根据算子运行模式、是否开启确定性计算及网络规模等因素调整。
    - HCCL的AlltoAllV、AlltoAllVC和AlltoAll（多机通信场景）算子在配置为AIV模式时，不会自动回退到AI CPU展开模式下发。为避免性能劣化，当任意两个rank之间的最大通信数据量不超过1MB时，建议配置为AIV模式，否则请采用AI_CPU模式。
    - 该配置项下，集合通信支持控核能力，建议业务根据实际使用场景中计算算子与通信算子的并发情况进行Vector Core核数的配置。

      - 针对Broadcast算子，建议至少分配ranksize个vector核。
      - 针对AllGather、非确定性ReduceScatter算子，建议最少分配max\(2, ceil\(ranksize/20\)\)个vector核。
      - 针对AllReduce、确定性ReduceScatter、AlltoAll、AlltoAllV、AlltoAllVC算子，建议最少分配max\(2, ceil\(ranksize/20\)\)个vector核，且核数需为偶数（若计算结果为奇数则向上取整至下一个偶数）。

        若业务编译分配的Vector Core核数无法满足算法编排的要求，HCCL会报错并提示所需要的最低Vector Core核数。

    **注意：**

    算法编排展开位置设置为“AIV”时，若同时设置了[HCCL_DETERMINISTIC](HCCL_DETERMINISTIC.md)环境变量为“true”或“strict”开启了确定性计算，确定性计算的优先级更高，某些场景下“AIV”展开可能不生效。
  <!-- end id3 -->

  <!-- npu="910b" id4 -->
- **针对Atlas A2系列产品**：支持的配置如下，若设置了不支持的环境变量，使用默认值。

  - **HOST（默认值）**：代表通信算子在Host侧CPU展开，Device侧根据硬件型号自动选择相应的调度器。
  - **HOST_TS**：代表通信算子在Host侧CPU展开，Host向Device的Task Scheduler下发任务，Device的Task Scheduler进行任务调度执行。
  - **AI_CPU**：代表通信算子在AI CPU展开，Device侧根据硬件型号自动选择相应的调度器。

    该配置项仅支持AllGather、AlltoAll、AlltoAllV、AlltoAllVC算子。

    **注意：**

    图模式（Ascend IR）场景，当通信算法采用AI CPU模式时，单卡上的并发图数量不能超过6个，否则可能会因AI CPU核被占满而导致通信阻塞。

  - **AIV**：代表通信算子在Vector Core展开，执行也在Vector Core。

    - 该配置项仅支持对称组网、推理特性。
    - 该配置项不支持多通信域并行的场景（因为不支持多个通信域同时配置为“AIV”模式），否则可能导致不可预期行为。您可以在初始化具有特定配置的通信域时，通过“HcclCommConfig”将某个通信域的算子展开模式设置为“AIV”。
    - 该配置项仅支持Broadcast、AllReduce、AlltoAll、AlltoAllV、AlltoAllVC、AllGather、ReduceScatter、AllGatherV、ReduceScatterV算子。
      - 针对Broadcast算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16、int64、uint64、float64, 仅支持单机场景8卡以内的单算子模式。
      - 针对AllReduce算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min。
      - 针对AlltoAll、AlltoAllV、AlltoAllVC算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16、int64、uint64、float64。针对AlltoAllV、AlltoAllVC算子，仅支持单机场景；针对AlltoAll算子的图模式运行方式，仅支持单机场景。
      - 针对AllGather算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16、int64、uint64、float64。针对该算子的图模式运行方式，仅支持单机场景。
      - 针对ReduceScatter算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min。针对该算子的图模式运行方式，仅支持单机场景。
      - 针对AllGatherV算子，数据类型支持int8、uint8、int16、uint16、int32、uint32、float16、float32、bfp16、int64、uint64、float64, 仅支持单算子模式。
      - 针对ReduceScatterV算子，数据类型支持int8、int16、int32、float16、float32、bfp16，reduce的操作类型仅支持sum、max、min。

    - 该配置项下，集合通信支持控核能力，建议业务根据实际使用场景中计算算子与通信算子的并发情况进行Vector Core核数的配置。

      - 针对AllReduce、ReduceScatter、ReduceScatterV算子，建议最少分配24个核。
      - 针对Broadcast、AlltoAll、AlltoAllV、AlltoAllVC、AllGather、AllGatherV算子，建议最少分配16个核。

        若业务编译分配的Vector Core核数无法满足算法编排的要求，HCCL会报错并提示所需要的最低Vector Core核数。

    **注意：**

    - 算法编排展开位置设置为“AIV”时，若同时设置了[HCCL_DETERMINISTIC](HCCL_DETERMINISTIC.md)环境变量为“true”或“strict”开启了确定性计算，确定性计算的优先级更高，某些场景下“AIV”展开可能不生效。
    - 对于Atlas 200T A2 Box16异构子框，不支持跨框通信场景。
  <!-- end id4 -->

  <!-- npu="310p" id1 -->
- **针对Atlas推理系列产品**：支持的配置如下，若设置了不支持的环境变量，使用默认值。
  - **HOST（默认值）**：代表通信算子在Host侧CPU展开，Device侧根据硬件型号自动选择相应的调度器。
  - **AI_CPU**：代表通信算子在AI CPU展开，Device侧根据硬件型号自动选择相应的调度器。
    - 仅支持单机单通信域场景。
    - 仅支持AllReduce算子，AllReduce算子支持的数据类型可参见HcclAllReduce接口。
    - 配置为“AI_CPU”后，通信算子不再支持profiling性能数据采集与分析功能。
    - 对于静态shape图，不支持此配置项，即不支持指定通信算子的展开模式为AI CPU。
  <!-- end id1 -->

## 配置示例

```bash
export HCCL_OP_EXPANSION_MODE="AI_CPU"
```

## 使用约束

- 若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclOpExpansionMode”参数配置了通信算子的展开模式，则以通信域粒度的配置优先。
- 针对Atlas A2系列产品的**推理特性：**

    **配置为“AIV”的场景下**，若通过“CTRL+C”方式强制结束进程，在msnpureport工具导出的Device侧日志文件中可能会出现Device访问非法地址的错误，日志关键词为“devmm_page_fault_d2h_query_flag”、“devmm_svm_device_fault”或“ipc_fault_msg_para_check”，如下所示，此种场景不会影响Device上卡的状态，不会影响后续新起任务的执行。

    ```text
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:22.646.254 [klogd.c:247][257382.266115] [ascend] [ERROR] [devmm] [devmm_page_fault_d2h_query_flag 810] <kworker/u16:2:14887,14887> Host page fault send message fail.(hostpid=2131021; devid=0; vfid=0; ret=-22; va=0x12c700300000; hostpid=2131021; devid=0; vfid=0)
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:22.646.284 [klogd.c:247][257382.266124] [ascend] [ERROR] [devmm] [devmm_svm_device_fault 468] <kworker/u16:2:14887,14887> Vm fault failed. (hostpid=2131021; devid=0; vfid=0; ret=64; fault_addr=0x12c700300000; start=0x12c700300000)
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:22.659.429 [klogd.c:247][257382.282181] [ascend] [ERROR] [tsdrv] [ipc_fault_msg_para_check 309] <swapper/3:0> Invalid node id. (devid=0; node_type=100; node_id=40; node_num=25)
    ................
    [ERROR] KERNEL(5044,sklogd):2024-07-29-10:33:24.874.211 [klogd.c:247][257384.473533] [ascend] [ERROR] [tsdrv] [tsdrv_hb_cq_callback 332] <kworker/0:0:20353> receive ts exception msg, call excep_code=0xb4060006, time=1722249204.850014098s, devid=0 tsid=0
    ```

## 产品支持情况

<!-- npu="950" id5 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id5 -->
<!-- npu="A3" id6 -->
- Atlas A3系列产品：支持
<!-- end id6 -->
<!-- npu="910b" id7 -->
- Atlas A2系列产品：支持
<!-- end id7 -->
<!-- npu="910" id8 -->
- Atlas训练系列产品：不支持
<!-- end id8 -->
<!-- npu="310p" id9 -->
- Atlas推理系列产品：支持
<!-- end id9 -->
