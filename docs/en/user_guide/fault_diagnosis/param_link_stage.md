# Parameter Plane Link Setup Stage

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:59:12.894Z pushedAt=2026-08-14T07:32:53.946Z -->

## Troubleshooting Approach for Link Setup Failures

When a communication operator is invoked, HCCL creates socket connections over the Parameter Plane network based on the TCP protocol to exchange information such as addresses according to service requirements. If a fault prevents some ranks from invoking the expected communication operator, they will be unable to initiate link setup requests. Alternatively, if network connectivity issues or behavior consistency problems prevent ranks from responding to each other's link setup requests, socket connection timeout errors will occur on other ranks.

Due to the design of HCCL algorithms and the order in which operators are invoked, link setup timeouts exhibit cascading propagation among ranks. Therefore, when a link setup timeout is detected, the fault point must be located first.

During the parameter plane link setup stage, HCCL provides the mechanisms described below to facilitate rapid problem locating.

### Root Node Location Mechanism for Link Setup

Considering the cascading propagation of link setup issues, for example, rank0 times out while waiting for link setup with rank1, and rank1 times out while waiting for link setup with rank2. If the link setup between rank1 and rank2 fails due to network or other reasons, rank0 will eventually report a link setup timeout error with rank1 as well, but the root cause lies between rank1 and rank2. Therefore, locating the root node of a link setup failure in a cluster is relatively difficult and tedious. HCCL initiates a fault detection link immediately after a service link setup failure. The main implementation principle is as follows:

The schematic diagram of root node location for link setup failure is shown below:

![Schematic Diagram of Root Node Location for Link Setup Failure](figures/link_fail_root_debug.png)

1. After a link setup failure, each rank starts a server side that listens to and can respond to the fault detection links of all ranks.

2. Initiate a fault probe link connection request to the remote end that cannot respond to its own service link setup request.

3. If the remote end cannot respond to its own probe link setup request, the link to the remote end or the service process on the remote end is considered faulty, and a probe failure event is generated. This event is then propagated to other links that have been successfully established on the server side.

4. If the remote end has established a probe link, receive the probe failure event sent by the peer and forward it.

In this way, if a link setup failure occurs due to any single-point issue, the node location of the fault point can be quickly identified through logs, enabling further problem diagnosis. For the detailed diagnosis process, see [Link Setup Timeout (EI0006)](#link-setup-timeout-ei0006).

If no event is detected after probing, the issue is most likely a behavioral consistency problem. That is, each rank has entered the link setup stage and responds to fault probe requests from other ranks, but the links experience mutual wait timeout due to inconsistent communication operators invoked by each rank. This is generally caused by a cluster behavioral consistency issue. Check factors such as scripts, environment, versions, and datasets. If the behavior of communication operators needs to be referenced, the operator behavior can be inferred from the tag information corresponding to the keyword "Alloc transports failed" in the link setup failure error log. For example, traverse the tag information of each rank. If, within a 16-rank communication domain, 15 ranks are performing allgather while one rank is performing AllReduce, focus on analyzing the differences in the invocation logic of the two operators.

For link setup timeout scenarios, it is possible to quickly determine whether it is a full-scale link setup timeout. If it is not a full-scale link setup timeout, priority should be given to troubleshooting nodes that have not reported link setup timeout errors. The reference command is as follows:

```bash
for i in *;do cd $i;pwd;grep -rnc "connection fail" | grep -v ":0" | wc -l; cd ..;done
```

### Consistency Check Mechanism

After HCCL successfully creates a socket connection with the peer, it exchanges information such as operator input parameters and the CANN version and verifies them against the local information. If any inconsistency is detected during the verification, an error is reported in the CANN log and the console log, and an error code is returned. For the detailed fault locating process, see [Parameter Consistency Check (EI0005)](#parameter-consistency-check-ei0005).

In single-operator mode, to ensure performance, HCCL triggers link setup only when an operator of a new type or algorithm is called for the first time in each communication domain. Since the consistency check is performed only after the link setup succeeds, this feature cannot intercept all dispatch inconsistency issues.

### Error Stage Analysis

HCCL has the following common error stage scenarios during the Communication Operator Parameter Plane link setup stage:

- For Atlas A3 training products/Atlas A3 inference products and Atlas A2 training products/Atlas A2 inference products, device NIC port binding failure may occur. The following command can be used to check whether there is a port binding failure issue. For details, see [Parameter Plane Port Binding Failure (EI0019)](#parameter-plane-port-binding-failure-ei0019).

    ```bash
    grep -rE "socket type\[(0|1)\].*Please check the port status and whether the port is being used by other process"
    ```

- Parameter plane socket link setup timeout may occur. The following command can be used to check whether there is a parameter plane link setup failure issue. For details, see [Link Setup Timeout (EI0006)](#link-setup-timeout-ei0006).

    ```bash
    grep -r "wait socket establish timeout"
    ```

- Communication operator consistency check failure may occur. The following command can be used to check whether there is a consistency check failure issue. For details, see [Parameter Consistency Check (EI0005)](#parameter-consistency-check-ei0005).

    ```bash
    grep -r "CMD information .* check fail"
    ```

## Parameter Plane Port Binding Failure (EI0019)

### Symptom

In the CANN log, the keyword "Please check the port status and whether the port is being used by other process." appears, as shown below. **In addition, note that port binding failures can also occur during the communication domain cluster negotiation stage. This can be determined based on "socket type" in the error log.** If the type is 0 or 1, it indicates a parameter plane port binding failure. If the type is 2, it indicates a host-side NIC port binding failure during communication domain cluster information negotiation. For details, see [Server Node Port Binding Failure (EI0019)](./cluster_info_nego.md#server-node-port-binding-failure-ei0019).

```text
[ERROR] HCCL(1009464,all_reduce_test):2025-03-15-00:41:48.470.172 [hccl_socket.cc:110] [1009464][InitGroupStage][RanktableDetect] socket type[0], listen on ip[192.168.2.199] and specific port[16666] fail. Please check the port status and whether the port is being used by other process.
```

### Possible Cause

During the Parameter Plane Setup of a communication operator, the current rank or process needs to bind a port on a device-side NIC, but the port is found to be already occupied by another process.

### Solution

When using device-side NIC ports, HCCL binds port 16666 by default. Therefore, if multiple processes run on the same device and all call the HCCL Communication Operator interface, the port will already be bound by another process, resulting in a failure.

In this case, first check whether running multiple processes on the same device aligns with the task expectations. If it does, the multi-process scenario can be enabled by configuring the `HCCL_NPU_SOCKET_PORT_RANGE` environment variable, for example:

```bash
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
```

## QP Memory Resource Allocation (EI0011)

During the parameter plane link setup stage, HCCL creates QPs. If the device-side memory is insufficient, an OOM error is reported. To resolve this issue, adjust the service configuration, reduce the number of ROCE links in use, or release some memory.

### Symptom

The keyword "EI0011" or "Resource_Error_Insufficient_Device_Memory" appears in the console log, as shown below:

```text
[PID: 2103452] 2025-11-03-20:18:46.447.213 Resource_Error_Insufficient_Device_Memory(EI0011): Failed to allocate [size: [0.25MB, 3MB], Affected by QP depth configuration.] bytes of NPU memory.
        Possible Cause: Allocation failure due to insufficient NPU memory.
        Solution: Stop unnecessary processes and ensure the required memory is available.
```

### Solution

Adjust the service configuration (such as batchSize), reduce the number of ROCE links in use, or release some memory to resolve the issue.

**NOTE**

If an OOM error occurs during other HCCL memory allocations, such as cclBuffer memory allocation, the drv component reports the error code and prints error information. Whether the failure is caused by HCCL memory allocation can be determined based on the error information or the stack in CANN logs. If the failure is caused by HCCL memory allocation, the requested memory size can be adjusted by configuring the HCCL_BUFFSIZE environment variable.

## Link Setup Timeout (EI0006)

The HCCL link setup timeout is affected by the environment variable [HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md). If the peer end fails to respond to the service link setup request within the timeout period, "socket timeout" is reported. Meanwhile, if the remote end exits due to a timeout or other faults, the already established links may also report "recv fail" errors while waiting for data exchange.

### Symptom

The CANN log contains the keyword `wait socket establish timeout` or `[InitChannelStage][Timeout]`, as shown below:

```text
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.403 [hccl_socket_manager.cc:797] [18744][Wait][LinkEstablish]wait socket establish timeout, role[1] rank[1] timeout[120 s]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.454 [hccl_socket_manager.cc:861] [18744][Wait][LinksEstablishCompleted] is failed. ret[9].
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.2.199(0)   |  16666  |   192.0.3.198(1)   |  3234403008  |  client  | time out |   DISABLE  | LinkInfo
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.942 [hccl_socket_manager.cc:836] [18744][Create][Sockets]Wait links establish completed failed, local role is client. ret[9][ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.964 [transport_manager.cc:1402] [18744][SetMachinePara]call trace: hcclRet -> 9
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.114.027 [transport_manager.cc:1252] [18744][CreateLink][InitChannelStage][Timeout]SetMachinePara error.
[ERROR] HCCL(17528,python3):2026-03-18-10:34:34.224.286 [detect_connect_anomalies.cc:494] [20039][CreateClientConnect]GetStatus fail, ret[9]
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.949 [detect_connect_anomalies.cc:127] [18744]-------------------CONNECT TIMEOUT DETECT RESULT-----------------------
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.966 [detect_connect_anomalies.cc:132] [18744]This node (server 192.168.200.100, device ID 1) detects that srcRank (server 192.168.200.100, device ID 1) fails to connect to dstRank (server 192.168.200.100, device ID 0). Continue to analyze the fault based on the logs of srcRank and dstRank.
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.970 [detect_connect_anomalies.cc:135] [18744]1. If the link setup timeout is reported on both ends, check the network connectivity between the two ends.2. If dstRank reports other exceptions, locate the cause based on the exception information of dstRank.3. If dstRank does not report any error, the possible cause is that the service process is suspended or exits in advance
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.977 [detect_connect_anomalies.cc:143] [18744]----------------------------------------------------------------------
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.013 [transport_manager.cc:1325] [18744][InitChannelStage][Timeout]Transport init error! createLink para:rank[1]-localUserrank[1]-localIpAddr[192.168.200.100/1], remoteRank[0]-remoteUserrank[0]-remoteIpAddr[192.168.200.100/0], machineType[1], linkMode[1], isUsedRdma[0], tag[HcomAllReduce_6629421139219749105_0]
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.040 [transport_manager.cc:1214] [18744][TransportManager][PrintErrorInfo]local rank information: nicType[VNIC_TYPE], logicSuperPodId is not set, phySuperPodId[287454020].
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.095 [transport_manager.cc:256] [18111][checkSubCommLinkThreadsStatus]call trace: hcclRet -> 9
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.101 [transport_manager.cc:363] [18111][AllocSubCommLinks]call trace: hcclRet -> 9
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.105 [transport_manager.cc:672] [18111][Alloc]call trace: hcclRet -> 9
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.108 [hccl_communicator_host.cc:6370] [18111][AllocAlgResource]Alloc transports failed, tag[HcomAllReduce_6629421139219749105_0_device]
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.120 [hccl_communicator_host.cc:4325] [18111][HcclCommunicator][ExecOp] AllocAlgResource failed, algName=[AllReduceRingFor91093Executor]
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.145 [hccl_communicator_host.cc:2858] [18111][AllReduce]call trace: hcclRet -> 9
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.152 [hccl_comm.cc:306] [18111][HcclComm][HcomAllReduce_6629421139219749105_0]errNo[0x0000000000000009] index[0]
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.156 [hcom.cc:515] [18111][AllReduce][Result]errNo[0x0000000005010009] hcclComm AllReduce error, tag[HcomAllReduce_6629421139219749105_0], input_ptr[0x12e083e00200], output_ptr[0x12e086600400], count[10485888], data_type[float32], op[sum]
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.164 [hcom_ops_kernel_info_store.cc:807] [18111][HcomAllReduceOpKernel]call trace: hcclRet -> 9
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.142.169 [hcom_ops_kernel_info_store.cc:358] [18111][HCCLOpsKernel]call trace: hcclRet -> 9

```

### Identifying the Link Setup Peer to Troubleshoot Based on Logs

- If "DETECT EVENT LIST" is printed in the error log, focus first on the failed link setup pair in the log. For example, in the log sample above, troubleshoot the root cause of the link setup failure between device7 of node 127.10.0.1 and device6 of node 127.10.0.1, as indicated by the "DETECT EVENT\[1\]" abnormal event.

- If "DETECT EVENT LIST" is not printed in the error log, obtain the device IPs of both ends of the link from the "LINK_ERROR_INFO" table in the error log. Additionally, retrieve the node information of the local and peer ends from the "**Transport init error! createLink para:**" key log entry, which is in the format \[hostIp/deviceId\], as shown below:

    Run **grep -r "Transport init error! createLink para:" debug/plog/plog-\*.log**. The following information is returned:

    ```text
    [ERROR] HCCL(3215542,all_reduce_test):2025-11-20-18:18:03.114.306 [transport_manager.cc:886] [3215599][InitChannelStage][Timeout]Transport init error! createLink para:rank[2]-localUserrank[2]-localIpAddr[127.10.0.1/2], remoteRank[1]-remoteUserrank[1]-remoteIpAddr[127.10.0.1/1], machineType[1], linkMode[1], isUsedRdma[0], tag[AllReduce_127.10.0.1%enp_60000_0_1763633852475745
    ```

  - localUserrank: the rank number of the local end.

  - localIpAddr: IP information of the local node.

  - remoteUserrank: Rank number of the remote end.

  - remoteIpAddr: IP information of the remote node.

  - tag: Communication operator identifier.

After obtaining the information of the peer end where the link setup failure occurred, **you can then perform further analysis by combining the CANN logs from both ends.**

### Confirming Peer Behavior to Troubleshoot Inter-Card Behavior Inconsistency

Parameter plane link setup is a two-way interactive process. Both ends must initiate a link setup request within the timeout period for the link to be established successfully. Otherwise, an error is reported due to wait timeout. Therefore, the peer node information can be identified from the error information on the local end, and the peer logs can be examined for further diagnosis:

**Figure 1** Troubleshooting approach  
![](figures/debug_thinking.png "Troubleshooting approach")

**Troubleshooting Point 1:**

If no error log is found on the peer end, it indicates that the peer end may not have synchronously delivered the corresponding communication operator. As a result, the local end cannot wait for the link setup request feedback from the peer end, eventually leading to a wait timeout.

It is necessary to check from the service side whether the communication operator delivery behavior is consistent between the two ends.

**Troubleshooting Point 2:**

If the peer end has reported errors other than the parameter plane link setup timeout, the cause of the peer end error must be troubleshot first.

**Troubleshooting Point 3:**

If the peer end has also reported a parameter plane link setup timeout error, but the error information indicates that the peer end is setting up a link with another node rather than the local end, the cause of the parameter plane link setup timeout on the peer end must be troubleshot first according to the procedure.

**Troubleshooting Point 4:**

If the peer is also experiencing a parameter plane link setup timeout with the local endpoint, first check whether the error times on both ends have exceeded the link setup wait time. If the link setup timeout has been exceeded, the root cause of the communication operator timeout on both ends must be investigated at the service level.

The link setup wait time can be specified by using HCCL_CONNECT_TIMEOUT, which defaults to 120 seconds. The timeout configured for the current service can be queried by running `grep -r "HCCL_CONNECT_TIMEOUT" run/plog/` in the run directory of the CANN log.

**Troubleshooting Point 5:**

If the parameter plane link setup timeout on both the peer and the local endpoint occurs within the link setup timeout period, further investigation into the network connectivity between the two ends is required:

1. Check whether the TLS switches on both ends are consistent. If the TLS switches on the two ends are inconsistent, the socket creation will fail verification, causing link setup timeout on both ends. The TLS switch status on both ends can be confirmed using the following methods:

    - In the LINK_ERROR_INFO table of the error log, the status field indicates the TLS state of the current device: UNKNOWN indicates that the state is not obtained, DISABLE indicates that TLS is disabled, and ENABLE indicates that TLS is enabled.

    - Run `grep -r "TLS SWITCH" log/run/device-*` in the node's log to obtain the TLS status:

        ```text
        run/device-0/device-2849330_20251024153927364.log:[INFO] HCCP(2988,hccp_service.bin):2025-10-24-15:39:26.133.826 [rs_ssl.c:1529]tid:2988,rs_ssl_init(1529) : TLS SWITCH (1)
        run/device-1/device-2849331_20251024153928174.log:[INFO] HCCP(30877,hccp_service.bin):2025-10-24-15:39:25.142.466 [rs_ssl.c:1529]tid:30877,rs_ssl_init(1529) : TLS SWITCH (0)
        ```

    - Use the hccn_tool to check the TLS configuration of the node: `for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch`

        ```bash
        # for i in {0..1}; do hccn_tool -i $i -tls -g ; done | grep switch
        dev_id:0, tls switch[0](0:disable, 1:enable), tls preconfigured[1](0:non-preset, 1:preset), tls alarm time threshold[60]days
        dev_id:1, tls switch[1](0:disable, 1:enable), tls preconfigured[1](0:non-preset, 1:preset), tls alarm time threshold[60]days
        ```

2. If the two ends of the link setup are on different nodes, check the network connectivity between the device network ports of the local and peer ends. Use the hccn_tool command on one node to ping the device IP of the other node:

    ```bash
    hccn_tool -i {node} -ping -g address {peer ip}
    ```

    If the two ranks cannot ping each other or a network port is down, contact the lab administrator to check the configuration of the corresponding NIC and switch.

3. When using the SuperPoD in Atlas A3 training products/Atlas A3 inference products, check whether nodes under different physical SuperPoDs are incorrectly configured as a single logical SuperPoD. In this case, HCCL incorrectly assumes that the two nodes can communicate through the vnic within the SuperPoD, resulting in a mutual wait timeout.

    The link type and physical SuperPoD information of both ends can be confirmed through the following logs: the link type is vnic, and the physical SuperPoD IDs of the two ends are different (0 and 1, respectively). However, because the same logical SuperPoD ID (logic_1) is configured, the vnic link is selected for communication, causing a timeout. This can be fixed by modifying or canceling the HCCL_LOGIC_SUPERPOD_ID configuration.

    Local end log:

    ```text
    debug/plog/plog-3003627_20260205184335411.log:14:[ERROR] HCCL(3003627,scatter_test):2026-02-05-18:44:26.379.547 [transport_manager.cc:885] [3003959][TransportManager][PrintErrorInfo]local rank information: nicType[VNIC_TYPE], logicSuperPodId[logic_1], phySuperPodId[0]. Note: Do not configure ranks belonging to different physical superpod ID info a single logical superpod ID
    ```

    Remote end log:

    ```text
    debug/plog/plog-3003628_20260205184354321.log:14:[ERROR] HCCL(3003628,scatter_test):2026-02-05-18:44:26.379.542 [transport_manager.cc:885] [3003959][TransportManager][PrintErrorInfo]local rank information: nicType[VNIC_TYPE], logicSuperPodId[logic_1], phySuperPodId[1]. Note: Do not configure ranks belonging to different physical superpod ID info a single logical superpod ID
    ```

Note:

1. The default threshold for the current faulty link to generate a probe failure event is 20s. Users can adjust it through the `connection_fault_detection_time` field in the HCCL_DFS_CONFIG environment variable. Setting it to 0 disables this feature. In large-scale clusters or when severe inter-card desynchronization occurs, this configuration may need to be increased to ensure the correctness of probe results.

2. In some complex service scenarios, link setup timeout and execution timeout may occur simultaneously in a single service execution. Multiple jumps based on probe results may be required to locate the fault point. Therefore, check the logs of the probe node to confirm whether the root node has been reached. The fault root node typically has other errors, no abnormal logs at all, or a mutual wait timeout with other ranks.

## Parameter Consistency Check (EI0005)

### Symptom

The console log contains the keyword "The arguments for collective communication are inconsistent between ranks", as shown below:

```text
EI0005: 2024-04-24-06:32:27.781.599 The arguments for collective communication are inconsistent between ranks:parameter count, local end 16512, remote end 8320
        TraceBack (most recent call last):
        Transport init error. Reason: [Create] [DestLink]Create Dest error! createLink para:rank[5]-localUserrank[4]-localIpAddr[127.10.0.1], dst_rank[6]-remoteUserrank[7]-remote_ip_addr[127.10.0.1]
        Transport init error. Reason: [Create] [DestLink]Create Dest error! createLink para:rank[5]-localUserrank[4]-localIpAddr[127.10.0.1], dst_rank[4]-remoteUserrank[5]-remote_ip_addr[127.10.0.1]
        call hccl op:HcomAllReduce(HcomAllReduce) load task fail[FUNC:Distribute][FILE:hccl_task_info.cc] [LINE:329]
        [[{[node Ge0p3_0]}]]
```

Or the CANN log contains the keyword "CMD information *** check fail", as shown below:

```text
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.640 [rank_consistentcy_checker.cc:429] [3743951][InitChannelStage][ParameterConflict]CMD information tag check fail. local[AllGather_127.10.0.1%enp_60000_0_1761379874757928], remote[AllReduce_127.10.0.1%enp_60000_0_1761379874757928]
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.666 [rank_consistentcy_checker.cc:439] [3743951][InitChannelStage][ParameterConflict]CMD information cmdType check fail. local[6], remote[2]
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.679 [rank_consistentcy_checker.cc:439] [3743951][InitChannelStage][ParameterConflict]CMD information op check fail. local[255], remote[0]
```

### Possible Cause

During parameter plane link setup, after the socket is established, a parameter consistency check is performed between the two endpoints. The check covers the operator identifier tag, operator type cmdType, reduction type op, data volume count, HCCL buffer size cclbufferSize, data type dataType, and other parameters. The inconsistent data can be identified based on the information in the error message. For example, in the following case, the operator identifier tags on the two endpoints are inconsistent, causing the communication operator to fail the consistency check during link setup. The data in local and remote represent the inconsistent values on the two endpoints.

The node information of the two endpoints with inconsistent parameters can be confirmed through the "Transport init error! createLink para:" error log. For example, execute `grep -r "Transport init error! createLink para:"` and the result is as follows:

```text
[ERROR] HCCL(3215542,all_reduce_test):2025-11-20-18:18:03.114.306 [transport_manager.cc:886] [3215599][InitChannelStage][Timeout]Transport init error! createLink para:rank[2]-localUserrank[2]-localIpAddr[127.10.0.1/2], remoteRank[1]-remoteUserrank[1]-remoteIpAddr[127.10.0.1/1], machineType[1], linkMode[1], isUsedRdma[0], tag[AllReduce_127.10.0.1%enp_60000_0_1763633852475745
```

- localUserrank: the rank number of the local endpoint.

- localIpAddr: the node IP information of the local endpoint.

- remoteUserrank: Rank number of the peer.

- remoteIpAddr: Node IP information of the peer.

- tag: Communication operator identifier.

### Solution

1. If the functionality works properly when SuperKernel is not enabled but an initialization inconsistency occurs after SuperKernel is enabled, it is recommended to move the HCCL operator out of the SuperKernel calibration scope. For specific instructions, see the "max-autotune Mode Features > Calibrating the SuperKernel Scope Within a Graph" section in [*PyTorch Graph Mode Usage Guide*](https://hiascend.com/en/document/detail/en/Pytorch/latest/index/index.html).

2. Based on the error information, troubleshoot the root cause of the inconsistency between the operators delivered by the two ends that failed the parameter consistency check from the service perspective.

    **NOTE** Some values printed in the log are enumeration values, where cmdType indicates the operator type and op indicates the reduction type. The mapping of enumeration values is shown in the following table:

    | cmdType Enumeration Value | Operator Type |
    | --- | --- |
    | 1 | BroadCast |
    | 2 | AllReduce |
    | 3 | Reduce |
    | 4 | Send |
    | 5 | Receive |
    | 6 | AllGather |
    | 7 | ReduceScatter |
    | 8 | AlltoAllV |
    | 9 | AlltoAllVC |
    | 10 | AlltoAll |
    | 11 | Gather |
    | 12 | Scatter |
    | 13 | BatchSendRecv |
    | 16 | AllGatherV |
    | 17 | ReduceScatterV |

The reduction types corresponding to the op enumeration values are shown in the following table:

    | op Enumeration Value | Reduction Type |
    | --- | --- |
    | 0 | SUM |
    | 1 | PROD |
    | 2 | MAX |
    | 3 | MIN |
    | 255 | Non-Reduce Operator |
