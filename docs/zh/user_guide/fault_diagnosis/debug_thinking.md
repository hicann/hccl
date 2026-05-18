# 定位思路

## 应知应会

在故障定位之前，请确保您已熟悉HCCL相关基本概念及故障定位辅助功能。

对于HCCL来说，故障码会涵盖大部分常见问题，如果报错中未包含故障码信息，或故障码信息为EI9999，可能为较为少见的故障场景或HCCL内部问题，请基于实际的CANN日志和代码进行分析，如果无法解决请联系技术支持。

对于没有清晰首报错的问题，大集群故障定位时，需要梳理每个rank的行为，通过rank之间的依赖关系找到根节点。面对这个难题，HCCL提供了建链根节点定位能力和集群心跳能力，并会在常见问题中给出诊断结果，相关原理请参见[建链失败定位思路](./param_link_stage.md#建链失败定位思路)、[集群心跳机制](./task_exec_stage.md#集群心跳机制)  。

本文档适用场景如下：

- 本文档对HCCL的实现机制的描述，仅用于解释各类故障模式机理，辅助分析故障现象和定位原因。如果在运行机制方面的内容和该机制相关介绍文档不符，请优先参考运行机制对应文档。
- 本文档中的部分CANN日志示例随着版本更新，内容会有所调整，用户可重点关注日志中的关键信息，如有较大的差异，请以实际的日志信息为准。
- 当业务发生HCCL异常时，在CANN日志中会有HCCL组件的报错日志信息，若在CANN日志中没有发现HCCL组件的报错日志，需排查是否有其他组件的报错信息，若无报错，请注意训练脚本本身有无异常、是否存在core dump或进程卡住等异常情况。

### 故障诊断相关环境变量

- [HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md)、[HCCL_EXEC_TIMEOUT](../hccl_env/HCCL_EXEC_TIMEOUT.md)

  HCCL在建链阶段和执行阶段的超时时间，建议HCCL_CONNECT_TIMEOUT配置的时间小于HCCL_EXEC_TIMEOUT配置的时间，以保证复杂场景下能够正确的上报首报错信息，以区分异常业务进程被阻塞的原因是本端还是远端。

- [HCCL_ENTRY_LOG_ENABLE](../hccl_env/HCCL_ENTRY_LOG_ENABLE.md)

  HCCL算子级入参记录开关，如果集群行为一致性问题无法通过其他手段锁定异常原因时，可以使能此环境变量，记录不同rank上的集合通信行为，通过卡间横向比对辅助找到行为差异引入点。

- [HCCL_DEBUG_CONFIG](../hccl_env/HCCL_DEBUG_CONFIG.md)

    HCCL模块级日志开关，进行算子开发调试时可以通过此配置分析算子内部的算法选择、任务编排等日志信息。

    该环境变量仅支持以下产品：

    Atlas A3 训练系列产品/Atlas A3 推理系列产品

    Atlas A2 训练系列产品/Atlas A2 推理系列产品

- [HCCL_DFS_CONFIG](../hccl_env/HCCL_DFS_CONFIG.md)

    HCCL高级故障探测配置能力，详见环境变量说明，建议保持默认值。

    该环境变量仅支持以下产品：

    Atlas A3 训练系列产品/Atlas A3 推理系列产品

    Atlas A2 训练系列产品/Atlas A2 推理系列产品

### HCCL相关日志说明

HCCL的日志信息会记录在CANN日志中，CANN的相关日志说明请参考《[日志参考](https://hiascend.com/document/redirect/CannCommunitylogref)》。

- 当HCCL报错时会在CANN日志的debug目录下打印关键的故障信息；同时在使用部分训练框架的业务场景下，HCCL也会在业务的日志中打印关键的报错信息；
- HCCL在CANN日志的run目录下会默认记录一些关键运行日志，如通信域的初始化与析构（默认打印）、通信算子的下发（需开启HCCL_ENTRY_LOG_ENABLE环境变量）等，关键日志示例如下：
  - 通信域初始化：

    ```text
    Entry-HcclGetRootInfo:rootInfo[0x7fffcd65f130], deviceLogicId[0]
    Entry-HcclCommInitRootInfoConfigInner:ranks[16], rank[0], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[group_name_0], deviceLogicId[0]
    ```

    - ranks：通信域大小。
    - rank：当前rank在通信域内的rank编号。
    - rootinfo：root节点的信息。
    - identifier：通信域名。

  - 通信域析构：

    ```text
    Entry-HcclCommDestroy: op_base comm destroy begin
    ```

  - 通信算子下发（需开启HCCL_ENTRY_LOG_ENABLE环境变量）：

    ```text
    Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth1_30000_0_1736576907435382], sendBuf[0x12e7bf550000], recvBuf[0x12e7bf550000], count[531260224], dataType[float32], op[sum], localRank[0], streamId[5],comm[0x331c9c00], deviceLogicId[0]
    ```

    - tag：通信算子标识符。
    - sendBuf：输入数据地址指针。
    - recvBuf：输出数据地址指针。
    - count：数据量。
    - dataType：数据类型。
    - op：reduce计算类型。
    - localRank：本端rank号。
    - streamId：通信算子执行流。
    - comm：通信域指针。
    - deviceLogicid：通信算子下发的设备逻辑ID。

  - 为了方便快速检索和识别通信域及本端的相关信息，HCCL提供了快速检索关键字：`Communicator Key Info`和`LocalRank Key Info`。
    - 例如执行`grep -r "Communicator Key Info"`得到以下信息：

      ```text
      run/plog/plog-858941_20251210195327204.log:[INFO] HCCL(858941,all_reduce_test):2025-12-10-19:53:28.131.350 [hccl_communicator_attrs.cc:327] [858941][Communicator Key Info]identifier[127.0.0.1%enp_60000_0_1765367607599032] rankSize[8] serverNum[1] moduleNum[1] superPodNum[0] multiModuleDiffDeviceNumMode[0] multiSuperPodDiffServerNumMode[0]
      ```

      通信域关键信息：`identifier[通信域名]`、`rankSize[通信域大小]`、`serverNum[通信域内节点数]`、`moduleNum[通信域内模组个数]`、`superPodNum[通信域内超节点个数]`、`multiModuleDiffDeviceNumMode[是否模组间卡数不一致]`、`multiSuperPodDiffServerNumMode[是否超节点间节点数不一致]`，信息中，“1”表示是，“0”表示否。

    - 例如执行`grep -r "LocalRank Key Info"`得到以下信息：

      ```text
      run/plog/plog-858941_20251210195327204.log:[INFO] HCCL(858941,all_reduce_test):2025-12-10-19:53:28.131.357 [hccl_communicator_attrs.cc:330] [858941][LocalRank Key Info]userRank[6] hostIp[127.0.0.1] devicePhyId[6] server[127.0.0.1] deviceIp[0.0.0.0] superPodId[0] useSuperPodMode[0] isStandardCard[0]
      ```

      本端关键信息：`userRank[通信域内的Rank号]`、`hostIp[host侧Ip]`、`devicePhyId[物理Id]`、`server[节点信息]`、`deviceIp[device侧Ip]`、`superPodId[超节点Id]`、`useSuperPodMode[是否为超节点模式]`、`isStandardCard[是否为标卡场景]`，信息中，“1”表示是，“0”表示否。

  - 如果想要查询已经配置成功的环境变量，其配置及实际生效值会被打印在CANN日志的run/plog目录下。

    针对Atlas 推理系列产品、Atlas 训练系列产品、Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品，可以通过检索`HCCL_ENV`的关键字查询每个进程的环境变量实际生效值，例如执行：`grep -r "HCCL_ENV" run/plog/plog-_xxx_.log`，得到以下信息：

    ```text
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.877 [externalinput.cc:598] [1595259][HCCL_ENV] HCCL_CONNECT_TIMEOUT set by default to [120]s
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.882 [externalinput.cc:558] [1595259][HCCL_ENV] HCCL_EXEC_TIMEOUT set by default to [1836]s
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.886 [externalinput.cc:663] [1595259][HCCL_ENV] HCCL_INTRA_PCIE_ENABLE set by default to [1], HCCL_INTRA_ROCE_ENABLE set by default to [0]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.890 [externalinput.cc:742] [1595259][HCCL_ENV] environmental variable PROFILING_MODE and GE profiling option is not set, default: false
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.895 [externalinput.cc:833] [1595259][HCCL_ENV] HCCL_WHITELIST_DISABLE set by environment to [0]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.912 [externalinput.cc:880] [1595259][HCCL_ENV] HCCL_IF_IP is not set
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.915 [externalinput.cc:936] [1595259][HCCL_ENV] HCCL_SOCKET_IFNAME set by default to [EmptyString]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.917 [externalinput.cc:903] [1595259][HCCL_ENV] HCCL_SOCKET_FAMILY is not set and is used by default [AF_INET]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.920 [externalinput.cc:865] [1595259][HCCL_ENV] HCCL_IF_BASE_PORT set by default to [60000]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.988 [externalinput.cc:1170] [1595259][HCCL_ENV] HCCL_RDMA_TC set by default to [132]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.881.991 [externalinput.cc:1205] [1595259][HCCL_ENV] HCCL_RDMA_SL set by default to [4]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.058 [externalinput.cc:1250] [1595259][HCCL_ENV] HCCL_RDMA_TIMEOUT set by default to [20]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.064 [externalinput.cc:1284] [1595259][HCCL_ENV] HCCL_RDMA_RETRY_CNT set by default to [7]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.069 [externalinput.cc:1370] [1595259][HCCL_ENV] HCCL_BUFFSIZE set by environment to [1]M
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.072 [externalinput.cc:621] [1595259][HCCL_ENV] HCCL_DETERMINISTIC set by default to [false]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.074 [externalinput.cc:1395] [1595259][HCCL_ENV] HCCL_DIAGNOSE_ENABLE set by default to [0]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.077 [externalinput.cc:1484] [1595259][HCCL_ENV] HCCL_ENTRY_LOG_ENABLE set by default to [0]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.081 [externalinput.cc:1505] [1595259][HCCL_ENV] HCCL_INTER_HCCS_DISABLE is not set, default value is FALSE.
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.090 [externalinput.cc:1569] [1595259][HCCL_ENV] environmental variable HCCL_OP_EXPANSION_MODE is [HOST], aicpuUnfold[0], aivMode[0], enableFfts[1]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.096 [externalinput.cc:1420] [1595259][HCCL_ENV] HCCL_RDMA_QPS_PER_CONNECTION is set to default value [1]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.099 [externalinput.cc:1454] [1595259][HCCL_ENV] HCCL_MULTI_QP_THRESHOLD is set to default value [512]KB
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.116 [externalinput.cc:1724] [1595259][HCCL_ENV][ParseRetryEnable] HCCL_OP_RETRY_ENABLE set by environment variable to [L0:0,L1:0,L2:0].
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.120 [externalinput.cc:1736] [1595259][HCCL_ENV] HCCL_OP_RETRY_PARAMS is not set, default value MaxCnt is [1], HoldTime is [5000]ms, IntervalTime is [1000]ms
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.123 [externalinput.cc:1778] [1595259][HCCL_ENV] HCCL_LOGIC_SUPERPOD_ID set by environment to [0]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.125 [externalinput.cc:525] [1595259][HCCL_ENV] HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT set by default to [EmptyString], rdmaFastPost is [0]
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.128 [externalinput.cc:791] [1595259][HCCL_ENV][Parse][MultiQpSrcPortConfigPath]environmental variable HCCL_RDMA_QP_PORT_CONFIG_PATH is empty
    [INFO] HCCL(1595259,alltoall_test):2026-01-06-15:38:29.882.131 [externalinput.cc:1800] [1595259][HCCL_ENV] HCCL_DEBUG_CONFIG is not set, debugConfig set by default to 0x0
    ```

    针对Ascend 950PR/Ascend 950DT，可通过检索关键字“base\_config”查询当前已设置的环境变量。

    ```text
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.170[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_IF_IP" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.176[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_IF_BASE_PORT" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.181[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_SOCKET_IFNAME" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.187[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_WHITELIST_DISABLE" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.192[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_HOST_SOCKET_PORT_RANGE" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.197[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_SOCKET_FAMILY" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.206[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_CONNECT_TIMEOUT" is parsed. 
    ```

## 快速定位定界思路

1. 确认是否为HCCL相关的异常报错。
    - HCCL针对常见的报错场景，会在业务打屏日志中上报错误信息及故障信息，若在业务日志中存在`EI****`或`EJ****`的故障码，则可根据对应的故障信息排查故障，或结合CANN日志中的报错信息对相关章节进行排查，故障码列表可见[HCCL相关故障码](#hccl相关故障码)。
    - 除了打屏的故障码信息，HCCL在CANN日志中会打印HCCL组件的ERROR级别日志，因此若在CANN日志中没有发现HCCL组件的报错日志，需排查是否有其他组件的报错信息，若无报错，请注意训练脚本本身有无异常、是否存在core dump或进程卡住等其他异常

2. 收集全量CANN日志。

    由于HCCL集合通信是一个通信域下全局的协同行为，某个节点上有HCCL的异常报错往往是因为在等待某个对端超时，此时需要结合对端的日志信息一起排查问题的根因。对于HCCL问题的定位定界需要收集集群下所有节点的CANN日志，包括debug目录和run目录的日志。

3. 确认当前报错阶段，并根据不同阶段进行排查。

    HCCL业务存在三个阶段，分别是通信域初始化、参数面建链和通信算子执行，由于不同阶段使用的硬件资源、通信拓扑和同步方式有明显差异，因此可先确认当前HCCL报错所在的阶段，再根据不同的阶段找到对应的章节做进一步排查。

    - HCCL在常见的报错场景增加了多级检索关键字，可以根据报错日志中的关键字快速识别当前报错阶段，并根据报错信息做进一步排查和定位。多级检索关键字详见[HCCL多级检索关键字](#hccl多级检索关键字)，如下日志表明在算子执行阶段发生了超时报错，且当前的算子展开方式为HOST模式：

      ```text
      [ERROR] HCCL(858209,all_reduce_test):2025-12-10-19:52:32.589.097 [task_exception_handler.cc:27] [858274][TaskExecStage][Timeout][HOST]Task run failed, base information is streamID:[1740], taskID[23], tag[AllReduce_127.0.0.1%enp_60000_0_1765367469951573], AlgType(level 0-1-2):[fullmesh-ring-NHR].
      ```

      **注意**：多级检索关键字功能仅在CANN 8.5.0版本及后续版本支持，对于不支持的版本或没有检索到关键字的场景，可根据其他方法判断当前的报错阶段。

    - HCCL提供了通信域创建接口和通信算子接口，且接口均为同步下发，异步执行。因此可分为以下几个场景：
        - 若业务在调用通信域创建接口失败时，或在报错日志中有`topoinfo`、`ranktable`关键字打印，可参考[通信域初始化阶段](comm_domain_init_stage.md)章节进一步排查。
        - 若业务在调用通信算子接口失败时，或在报错日志中有`transport`关键字打印，可参考[参数面建链阶段](param_link_stage.md)章节进一步排查。
        - 若业务创建通信域接口和通信算子下发均成功，而是在触发流同步时有HCCL的算子执行失败，或在报错日志中有"TaskExceptionHandler"、"FFTS+ run failed"、"Task run failed"关键字打印，可参考[任务下发执行阶段](task_exec_stage.md)章节做进一步排查。

            除此三个阶段的关键信息外，若在业务的打屏日志中有明确的错误码信息，如`EI0001`，可直接根据错误码在后续内容中找到对应的故障码，并进一步排查。

### HCCL多级检索关键字

| 一级关键字 | 二级检索关键字 | 故障场景 |
| --- | --- | --- |
| InitGroupStage | EnvConfig | [通信域初始化阶段环境变量配置异常](env_config_error_EI0001.md) |
|                |RanktableConfig | [通信域初始化阶段rankTable文件读取失败](rank_table_load_fail.md) |
|                |RanktableCheck | [通信域初始化阶段rankTable集群信息校验失败](cluster_info_verify_fail.md) |
|                |RanktableDetect | [通信域初始化阶段集群信息探测失败](cluster_info_nego.md) |
|                |Resource | 通信域初始化节点资源初始化失败 |
| InitChannelStage | ParameterConflict | [参数面建链阶段参数一致性校验失败](./param_link_stage.md#参数一致性校验ei0005) |
|                |VersionConflict | 参数面建链阶段HCCL版本不一致校验失败 |
|                |Timeout | [参数面建链阶段超时报错](./param_link_stage.md#建链超时ei0006) |
| TaskExecStage | InvalidArgument | 算子执行阶段入参校验失败 |
|               |Not Supported | 算子执行阶段不支持场景 |
|               |Timeout | [算子执行阶段执行超时](./task_exec_stage.md#定位思路) |
|               |RunFailed | [算子执行阶段执行失败](./task_exec_stage.md#task-exception机制) |
|               |HeartbeatAbnormal | [算子执行阶段发现心跳异常事件](./task_exec_stage.md#集群心跳机制) |  |

### HCCL相关故障码

| 故障码 | 故障码说明 |
| --- | --- |
| EI0001 | [环境变量配置异常](env_config_error_EI0001.md) |
| EI0002 | [通信算子执行超时](./task_exec_stage.md#定位思路) |
| EI0003 | 集合通信算子入参校验失败 |
| EI0004 | [rankTable文件加载失败](rank_table_load_fail.md) |
| EI0005 | [参数一致性校验失败](./param_link_stage.md#参数一致性校验ei0005) |
| EI0006 | [通信算子参数面建链超时](./param_link_stage.md#建链超时ei0006) |
| EI0007 | 资源初始化失败 |
| EI0008 | HCCL版本不一致，校验失败 |
| EI0011 | [QP内存资源申请失败](./param_link_stage.md#qp内存资源申请相关ei0011) |
| EI0012 | [算子执行时发生SDMA任务异常](./task_exec_stage.md#sdma-errorei0012) |
| EI0013 | [算子执行时发生ROCE CQE ERROR异常](./task_exec_stage.md#error-cqe报错ei0013) |
| EI0014 | [集群信息校验失败](cluster_info_verify_fail.md) |
| EI0015 | [通信域集群信息协商阶段超时](cluster_info_nego.md) |
| EI0019 | [通信域创建阶段server节点端口绑定失败](./cluster_info_nego.md#server节点端口绑定失败ei0019)或[参数面建链阶段端口绑定失败](./param_link_stage.md#参数面端口绑定失败ei0019) |
