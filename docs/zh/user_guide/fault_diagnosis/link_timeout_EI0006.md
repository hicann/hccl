# 建链超时（EI0006）

HCCL建链超时受环境变量[HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md)的影响，若在超时时间内对端无法响应业务建链请求，则会上报“socket timeout”，同时如果远端由于超时等故障退出，已经建好的链路在等待数据交换的过程中也可能会出现“recv fail”的报错。

## 问题现象

在CANN日志中存在关键字“wait socket establish timeout”或“\[InitChannelStage\][Timeout\]”，如下所示：

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

## 根据日志确认需排查的建链对端

- 若报错日志中打印了“DETECT EVENT LIST”，可先重点关注日志中失败的建链对，如上日志示例中，需先排查“DETECT EVENT\[1\]”异常事件显示的127.10.0.1节点的device7和127.10.0.1节点的device6之间的建链失败根因。

- 若报错日志中没有打印“DETECT EVENT LIST”，可从报错日志的"LINK_ERROR_INFO"表格中获取建链两端的device ip，同时可从“**Transport init error! createLink para:**”关键日志信息中获取本端和对端所在的节点信息，格式为\[hostIp/deviceId\]，如下所示：

    执行**grep -r "Transport init error! createLink para:" debug/plog/plog-\*.log**，得到如下信息：

    ```text
    [ERROR] HCCL(3215542,all_reduce_test):2025-11-20-18:18:03.114.306 [transport_manager.cc:886] [3215599][InitChannelStage][Timeout]Transport init error! createLink para:rank[2]-localUserrank[2]-localIpAddr[127.10.0.1/2], remoteRank[1]-remoteUserrank[1]-remoteIpAddr[127.10.0.1/1], machineType[1], linkMode[1], isUsedRdma[0], tag[AllReduce_127.10.0.1%enp_60000_0_1763633852475745
    ```

  - localUserrank：本端rank编号。
  - localIpAddr：本端的节点Ip信息。
  - remoteUserrank：对端rank编号。
  - remoteIpAddr：对端的节点Ip信息。
  - tag：通信算子标识符。

获取到需要排查的建链失败对端信息之后，**便可结合两端的CANN日志做进一步分析。**

### 确认对端行为排查是否有卡间行为不一致

由于参数面建链是一个两端的互动流程，需要两端在超时时间内均发起建链请求才能创建成功，否则因为等待超时而报错，因此可以根据本端的报错信息中找到对端的节点信息，查看对端的日志做进一步的判断：

**图1**  排查思路  
![](figures/debug_thinking.png "排查思路")

**排查点1：**

若对端没有任何报错日志，说明对端可能没有同步下发对应的通信算子，因此本端无法等待到对端的建链请求反馈，最终等待超时。

需从业务上排查两端的通信算子下发行为是否一致。

**排查点2：**

若对端发生了除了参数面建链超时外的其他报错，则需要先排查对端的报错原因。

**排查点3：**

若对端也发生了参数面建链超时报错，但对端的报错信息中并不在和本端建链，而是和其他节点建链，则需要按照流程先排查对端的参数面建链超时原因。

**排查点4：**

若对端也在和本端参数面建链超时，可先排查两端的报错时间是否超过了建链等待时间，如超过了建链超时时间，需要业务上排查两端通信算子下发超时时间的根因。

建链等待时间可通过HCCL_CONNECT_TIMEOUT指定，默认为120秒，可在CANN日志的run目录下通过`grep -r "HCCL_CONNECT_TIMEOUT" run/plog/`查询当前业务配置的超时时间。

**排查点5：**

若对端和本端的参数面建链超时在建链超时时间内，则需要进一步排查两端的网络连通性：

1. 排查两端的tls开关是否一致，若两端的tls开关不一致，则socket创建时会校验失败导致两端均建链超时，可以通过以下方法确认两端的tls开关：
    - 报错日志的LINK_ERROR_INFO表格中的status表示的是当前卡的tls状态，UNKNOWN表示未获取到，DISABLE表示未开启，ENABLE表示开启。
    - 在节点的log日志中执行`grep -r "TLS SWITCH" log/run/device-*`获取tls状态：

        ```text
        run/device-0/device-2849330_20251024153927364.log:[INFO] HCCP(2988,hccp_service.bin):2025-10-24-15:39:26.133.826 [rs_ssl.c:1529]tid:2988,rs_ssl_init(1529) : TLS SWITCH (1)
        run/device-1/device-2849331_20251024153928174.log:[INFO] HCCP(30877,hccp_service.bin):2025-10-24-15:39:25.142.466 [rs_ssl.c:1529]tid:30877,rs_ssl_init(1529) : TLS SWITCH (0)
        ```

    - 通过hccn_tool工具查看节点的tls配置`for i in {0..7}; do hccn_tool -i $i -tls -g ; done | grep switch`：

        ```bash
        # for i in {0..1}; do hccn_tool -i $i -tls -g ; done | grep switch
        dev_id:0, tls switch[0](0:disable, 1:enable), tls preconfigured[1](0:non-preset, 1:preset), tls alarm time threshold[60]days
        dev_id:1, tls switch[1](0:disable, 1:enable), tls preconfigured[1](0:non-preset, 1:preset), tls alarm time threshold[60]days
        ```

2. 若建链的两端在不同的节点上，则需要检查本端和对端的device网口之间的网络连通性，使用hccn_tool命令在其中一个节点ping另外一个节点的device ip：

    ```bash
    hccn_tool -i {node} -ping -g address {对端ip}
    ```

    若两个rank之间ping不通或者有网口是down的，请联系实验室管理员排查对应网卡及交换机的配置。

    <!-- npu="A3" id4 -->
3. 若使用Atlas A3 训练系列产品/Atlas A3 推理系列产品中的超节点，请注意检查是否错误地将不同物理超节点下的节点配置成为一个逻辑超节点，这种情况下HCCL会错误地认为两个节点能够通过超节点内的vnic进行通信，从而导致互等超时。

    可以通过如下日志确认两端的链路类型和物理超节点信息：链路类型为vnic，且两端的物理超节点ID不相同（分别是0和1），但由于配置了相同的逻辑超节点ID（logic_1），因此选择vnic链路进行通信导致超时，可以通过修改或者取消HCCL_LOGIC_SUPERPOD_ID配置进行修复。

    本端日志：

    ```text
    debug/plog/plog-3003627_20260205184335411.log:14:[ERROR] HCCL(3003627,scatter_test):2026-02-05-18:44:26.379.547 [transport_manager.cc:885] [3003959][TransportManager][PrintErrorInfo]local rank information: nicType[VNIC_TYPE], logicSuperPodId[logic_1], phySuperPodId[0]. Note: Do not configure ranks belonging to different physical superpod ID info a single logical superpod ID
    ```

    远端日志：

    ```text
    debug/plog/plog-3003628_20260205184354321.log:14:[ERROR] HCCL(3003628,scatter_test):2026-02-05-18:44:26.379.542 [transport_manager.cc:885] [3003959][TransportManager][PrintErrorInfo]local rank information: nicType[VNIC_TYPE], logicSuperPodId[logic_1], phySuperPodId[1]. Note: Do not configure ranks belonging to different physical superpod ID info a single logical superpod ID
    ```
    <!-- end id4 -->

需注意：

1. 当前故障链路产生探测失败事件的阈值默认为20s，用户可以通过HCCL_DFS_CONFIG环境变量中`connection_fault_detection_time`的字段进行调整，配置为0则关闭此功能。在集群规模较大或伴随严重的卡间不同步现象时，可能需要增大此配置以确保探测结果正确性。
2. 在部分复杂业务场景下，建链超时、执行超时可能同时出现在单次业务中，需要基于探测结果进行多次跳转才能定位到故障点。因此请以探测节点的日志确认是否已经到达根节点。故障根节点通常会有其他报错、或无任何异常日志，或和其他rank互等超时。
