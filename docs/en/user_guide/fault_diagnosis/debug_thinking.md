# Troubleshooting Approach

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:58:50.226Z pushedAt=2026-08-13T07:01:01.072Z -->

## Prerequisites

Before performing fault localization, ensure that you are familiar with basic HCCL concepts and auxiliary fault localization functions.

For HCCL, fault codes cover most common issues. If the error message does not contain fault code information, or if the fault code is EI9999, the issue may be a relatively rare fault scenario or an internal HCCL problem. In such cases, analyze the issue based on the actual CANN logs and code. If the issue cannot be resolved, contact technical support.

For issues without a clear first reported error, when performing fault localization in a large cluster, it is necessary to sort out the behavior of each rank and identify the root node through the dependencies between ranks. To address this challenge, HCCL provides the link establishment root node localization capability and the cluster heartbeat capability, and it outputs diagnostic results for common issues. For related principles, see [Link Establishment Failure Troubleshooting Approach](./param_link_stage.md#troubleshooting-approach-for-link-setup-failures) and [Cluster Heartbeat Mechanism](./task_exec_stage.md#cluster-heartbeat-mechanism).

This document is applicable to the following scenarios:

The descriptions of the HCCL implementation mechanisms in this document are intended solely to explain the principles of various failure modes and to assist in analyzing fault symptoms and identifying root causes. If any content related to operational mechanisms in this document conflicts with the corresponding documentation for those mechanisms, the latter shall prevail.

- Some CANN log examples in this document may be adjusted as versions are updated. Users are advised to focus on the key information in the logs. In case of significant discrepancies, the actual log information shall prevail.

- When an HCCL exception occurs during service execution, error log information from the HCCL component will appear in the CANN log. If no error log from the HCCL component is found in the CANN log, it is necessary to check whether error information exists from other components. If no errors are found, attention should be paid to whether the training script itself has anomalies, or whether there are abnormal conditions such as core dumps or process hangs.

### Fault Diagnosis-Related Environment Variables

- [HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md), [HCCL_EXEC_TIMEOUT](../hccl_env/HCCL_EXEC_TIMEOUT.md)

  Timeout durations for HCCL during the link establishment phase and the execution phase. It is recommended that the value configured for HCCL_CONNECT_TIMEOUT be smaller than that for HCCL_EXEC_TIMEOUT, so that the first error information can be correctly reported in complex scenarios, thereby distinguishing whether the blocking of an abnormal service process is caused by the local end or the remote end.

- [HCCL_ENTRY_LOG_ENABLE](../hccl_env/HCCL_ENTRY_LOG_ENABLE.md)

  A switch for recording HCCL operator-level input parameters. When the root cause of a cluster behavior consistency issue cannot be identified through other means, this environment variable can be enabled to record the collective communication behavior on different ranks, and cross-card horizontal comparison can be used to assist in locating the point where the behavior divergence is introduced.

- [HCCL_DEBUG_CONFIG](../hccl_env/HCCL_DEBUG_CONFIG.md)

    HCCL module-level log switch. During operator development and debugging, this configuration can be used to analyze log information such as algorithm selection and task orchestration within the operator.

    This environment variable is supported only on the following products:

    Atlas A3 training products/Atlas A3 inference products

    Atlas A2 training products/Atlas A2 inference products

- [HCCL_DFS_CONFIG](../hccl_env/HCCL_DFS_CONFIG.md)

    Advanced HCCL fault detection configuration capability. For details, see the environment variable description. It is recommended to keep the default value.

    This environment variable is supported only on the following products:

    Atlas A3 training products/Atlas A3 inference products

    Atlas A2 training products/Atlas A2 inference products

### HCCL-Related Log Description

HCCL log information is recorded in CANN logs. For details about CANN logs, see *[Log Reference](https://hiascend.com/en/document/redirect/CannCommunitylogref)*.

- When HCCL reports an error, key fault information is printed in the debug directory of CANN logs. In addition, in some service scenarios where training frameworks are used, HCCL also prints key error information in the service logs.

- In the run directory of CANN logs, HCCL records some key runtime logs by default, such as communicator initialization and destruction (printed by default) and communication operator dispatch (requires the `HCCL_ENTRY_LOG_ENABLE` environment variable to be enabled). Examples of key logs are as follows:

  - Communicator initialization:

    ```text
    Entry-HcclGetRootInfo:rootInfo[0x7fffcd65f130], deviceLogicId[0]
    Entry-HcclCommInitRootInfoConfigInner:ranks[16], rank[0], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[group_name_0], deviceLogicId[0]
    ```

    - ranks: Size of the communicator.

    - rank: Rank number of the current rank within the communicator.

    - rootinfo: Information about the root node.

    - identifier: Name of the communicator.

  - Communicator destruction:

    ```text
    Entry-HcclCommDestroy: op_base comm destroy begin
    ```

  - Communication operator dispatch (requires enabling the HCCL_ENTRY_LOG_ENABLE environment variable):

    ```text
    Entry-HcclAllReduce: tag[AllReduce_127.10.0.1%eth1_30000_0_1736576907435382], sendBuf[0x12e7bf550000], recvBuf[0x12e7bf550000], count[531260224], dataType[float32], op[sum], localRank[0], streamId[5],comm[0x331c9c00], deviceLogicId[0]
    ```

    - tag: Communication operator identifier.

    - sendBuf: Input data address pointer.

    - recvBuf: Output data address pointer.

    - count: Data volume.

    - dataType: Data type.

    - op: Reduce computation type.

    - localRank: Local rank number.

    - streamId: Stream on which the communication operator is executed.

    - comm: Communicator pointer.

    - deviceLogicid: Device logical ID delivered by the communication operator.

  - To facilitate quick retrieval and identification of communicator and local information, HCCL provides quick retrieval keywords: `Communicator Key Info` and `LocalRank Key Info`.

    - For example, executing `grep -r "Communicator Key Info"` yields the following information:

      ```text
      run/plog/plog-858941_20251210195327204.log:[INFO] HCCL(858941,all_reduce_test):2025-12-10-19:53:28.131.350 [hccl_communicator_attrs.cc:327] [858941][Communicator Key Info]identifier[127.0.0.1%enp_60000_0_1765367607599032] rankSize[8] serverNum[1] moduleNum[1] superPodNum[0] multiModuleDiffDeviceNumMode[0] multiSuperPodDiffServerNumMode[0]
      ```

      Communicator key information: `identifier[communicator name]`, `rankSize[communicator size]`, `serverNum[number of servers within communicator]`, `moduleNum[number of modules within communicator]`, `superPodNum[number of super pods within communicator]`, `multiModuleDiffDeviceNumMode[Whether the device count differs between modules]`, `multiSuperPodDiffServerNumMode[Whether the server count differs between super pods]`. In the information, "1" indicates yes and "0" indicates no.

    - For example, executing `grep -r "LocalRank Key Info"` yields the following information:

      ```text
      run/plog/plog-858941_20251210195327204.log:[INFO] HCCL(858941,all_reduce_test):2025-12-10-19:53:28.131.357 [hccl_communicator_attrs.cc:330] [858941][LocalRank Key Info]userRank[6] hostIp[127.0.0.1] devicePhyId[6] server[127.0.0.1] deviceIp[0.0.0.0] superPodId[0] useSuperPodMode[0] isStandardCard[0]
      ```

      Local key information: `userRank[Rank number within communicator]`, `hostIp[host-side IP]`, `devicePhyId[physical ID]`, `server[node information]`, `deviceIp[device-side IP]`, `superPodId[superPod ID]`, `useSuperPodMode[Whether it is superPod mode]`, `isStandardCard[Whether it is a standard card scenario]`. In the information, "1" indicates yes, and "0" indicates no.

  - To query the environment variables that have been successfully configured, their configured and effective values are printed in the `run/plog` directory of CANN logs.

    For Atlas inference products, Atlas training products, Atlas A2 training products/Atlas A2 inference products, and Atlas A3 training products/Atlas A3 inference products, the actually effective values of environment variables for each process can be queried by searching for the `HCCL_ENV` keyword. For example, run `grep -r "HCCL_ENV" run/plog/plog-_xxx_.log` to obtain the following information:

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

    For Ascend 950PR/Ascend 950DT, the currently configured environment variables can be queried by searching for the keyword "base_config".

    ```text
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.170[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_IF_IP" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.176[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_IF_BASE_PORT" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.181[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_SOCKET_IFNAME" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.187[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_WHITELIST_DISABLE" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.192[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_HOST_SOCKET_PORT_RANGE" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.197[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_SOCKET_FAMILY" is not set. Default value is used. 
    [INFO] HCCL(229424,python3.8):2025-12-23-22:31:40.239.206[base_config.cc:33][229424][Init][EnvVarParam]Env config "HCCL_CONNECT_TIMEOUT" is parsed. 
    ```

## Rapid Fault Localization and Delimitation Approach

1. Determine whether the error is HCCL-related.

    - For common error scenarios, HCCL reports error information and fault information in the service console logs. If a fault code such as `EI****` or `EJ****` is found in the service logs, the fault can be diagnosed based on the corresponding fault information, or the relevant sections can be checked in combination with the error information in the CANN logs. For the fault code list, see [HCCL-Related Fault Codes](#hccl-related-fault-codes).

    - In addition to the fault code information printed on the console, HCCL prints ERROR-level logs of the HCCL component in the CANN logs. Therefore, if no error log of the HCCL component is found in the CANN logs, it is necessary to check whether there are error messages from other components. If no error is found, check whether the training script itself is abnormal, or whether there are other anomalies such as core dumps or process hangs.

2. Collect all CANN logs.

    Since HCCL collective communication is a globally coordinated behavior within a communicator, an HCCL error on a single node is often caused by waiting for a peer timeout. In such cases, the root cause must be investigated in conjunction with the log information of the peer. For HCCL fault localization and demarcation, CANN logs from all nodes in the cluster must be collected, including logs from both the debug directory and the run directory.

3. Confirm the current error phase and perform troubleshooting based on the phase.

    HCCL services involve three phases: communicator initialization, parameter plane establishment, and communication operator execution. Since the hardware resources, communication topology, and synchronization methods used in different phases differ significantly, first confirm the phase in which the current HCCL error occurs, and then refer to the corresponding section for further troubleshooting based on the phase.

    - HCCL has added multi-level retrieval keywords for common fault scenarios. You can quickly identify the current error phase based on the keywords in the error log, and perform further troubleshooting and localization based on the error information. For details about multi-level retrieval keywords, see [HCCL Multi-Level Retrieval Keywords](#hccl-multi-level-retrieval-keywords). The following log indicates that a timeout error occurred during the operator execution phase, and the current operator expansion mode is HOST mode:

      ```text
      [ERROR] HCCL(858209,all_reduce_test):2025-12-10-19:52:32.589.097 [task_exception_handler.cc:27] [858274][TaskExecStage][Timeout][HOST]Task run failed, base information is streamID:[1740], taskID[23], tag[AllReduce_127.0.0.1%enp_60000_0_1765367469951573], AlgType(level 0-1-2):[fullmesh-ring-NHR].
      ```

      **NOTE**: The multi-level retrieval keyword feature is supported only in CANN 8.5.0 and later versions. For unsupported versions or scenarios where no keywords are retrieved, the current error phase can be determined using other methods.

    - HCCL provides communicator creation interfaces and communication operator interfaces. Both types of interfaces are synchronously submitted and asynchronously executed. Therefore, the following scenarios can be distinguished:

        - If the service fails when calling the communicator creation interface, or if keywords such as `topoinfo` and `ranktable` appear in the error logs, refer to the [Communicator Initialization Phase](comm_domain_init_stage.md) chapter for further troubleshooting.

        - If the service fails when calling the communication operator interface, or if the keyword `transport` appears in the error logs, refer to the [Parameter Plane Establishment Phase](param_link_stage.md) chapter for further troubleshooting.

        - If both the communicator creation interface and the communication operator submission succeed, but an HCCL operator execution failure occurs during stream synchronization, or if keywords such as "TaskExceptionHandler", "FFTS+ run failed", or "Task run failed" appear in the error logs, refer to the [Task Dispatch and Execution Phase](task_exec_stage.md) chapter for further troubleshooting.

            In addition to the key information from these three phases, if explicit error code information such as `EI0001` appears in the service console logs, you can directly locate the corresponding fault code in the subsequent content based on the error code and proceed with further troubleshooting.

### HCCL Multi-Level Retrieval Keywords

| Primary Keyword | Secondary Retrieval Keyword | Fault Scenario |
| --- | --- | --- |
| InitGroupStage | EnvConfig | [Environment Variable Configuration Exception During Communicator Initialization](env_config_error_EI0001.md) |
|                |RanktableConfig | [rankTable File Read Failure During Communicator Initialization](rank_table_load_fail.md) |
|                |RanktableCheck | [Cluster Information Verification Failure During Communicator Initialization](cluster_info_verify_fail.md) |
|                |RanktableDetect | [Cluster Information Detection Failure During Communicator Initialization](cluster_info_nego.md) |
|                |Resource | Node resource initialization failure during communicator initialization |
| InitChannelStage | ParameterConflict | [Parameter Consistency Verification Failure During Parameter Plane Link Establishment](./param_link_stage.md#parameter-consistency-check-ei0005) |
|                |VersionConflict | HCCL version inconsistency verification failure during parameter plane link establishment |
|                |Timeout | [Timeout Error During Parameter Plane Link Establishment](./param_link_stage.md#link-setup-timeout-ei0006) |
| TaskExecStage | InvalidArgument | Input parameter verification failure during operator execution |
|               |Not Supported | Unsupported scenario during operator execution |
|               |Timeout | [Execution Timeout During Operator Execution](./task_exec_stage.md#troubleshooting-approach) |
|               |RunFailed | [Execution Failure During Operator Execution](./task_exec_stage.md#task-exception-mechanism) |
|               |HeartbeatAbnormal | [Heartbeat Abnormal Event Detected During Operator Execution](./task_exec_stage.md#cluster-heartbeat-mechanism) |  |

### HCCL-Related Fault Codes

| Fault Code | Fault Code Description |
| --- | --- |
| EI0001 | [Environment Variable Configuration Error](env_config_error_EI0001.md) |
| EI0002 | [Communication Operator Execution Timeout](./task_exec_stage.md#troubleshooting-approach) |
| EI0003 | Collective communication operator input parameter verification failed |
| EI0004 | [rankTable File Loading Failure](rank_table_load_fail.md) |
| EI0005 | [Parameter Consistency Verification Failure](./param_link_stage.md#parameter-consistency-check-ei0005) |
| EI0006 | [Communication Operator Parameter Plane Link Establishment Timeout](./param_link_stage.md#link-setup-timeout-ei0006) |
| EI0007 | Resource initialization failed |
| EI0008 | HCCL version mismatch, verification failed |
| EI0011 | [QP Memory Resource Application Failure](./param_link_stage.md#qp-memory-resource-allocation-ei0011) |
| EI0012 | [SDMA Task Exception During Operator Execution](./task_exec_stage.md#sdma-error-ei0012) |
| EI0013 | [ROCE CQE ERROR Exception During Operator Execution](./task_exec_stage.md#error-cqe-error-report-ei0013) |
| EI0014 | [Cluster Information Verification Failure](cluster_info_verify_fail.md) |
| EI0015 | [Communicator Cluster Information Negotiation Phase Timeout](cluster_info_nego.md) |
| EI0019 | [Server Node Port Binding Failure During Communicator Creation Phase](./cluster_info_nego.md#server-node-port-binding-failure-ei0019) or [Port Binding Failure During Parameter Plane Link Establishment Phase](./param_link_stage.md#parameter-plane-port-binding-failure-ei0019) |
