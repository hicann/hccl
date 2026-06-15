# 参数面建链阶段

## 建链失败定位思路

在调用通信算子时，HCCL会通过参数面网络基于TCP协议进行socket连接创建，以此来基于业务需要进行地址等信息交换，此时如果出现某种故障导致部分rank未调用到预期的通信算子，导致无法发起建链请求，或由于网络连通性、行为一致性问题导致无法响应彼此之间的建链请求，就会导致其他rank出现socket连接超时报错。

由于HCCL的算法和算子调用顺序的原因，建链超时会呈现在rank间有级联传递的情况，因此如果发现建链超时，需要优先找到故障点位置。

HCCL在参数面建链阶段提供了下文所述机制来辅助问题快速定位。

### 建链根节点定位机制

考虑到建链问题的级联传播问题，如rank0在和rank1建链等待超时，而rank1在和rank2建链等待超时，若此时rank1和rank2的建链由于网络或其他原因导致失败，在rank0上最终也会上报和rank1的建链超时报错，但失败的根因却在rank1和rank2之间。因此在集群中找到建链失败根节点位置是较为困难和繁琐的，HCCL会在业务上建链失败后立即启动故障探测链路，其主要的实现原理为：

建链失败根节点定位原理图如下所示：

![建链失败根节点定位原理图](figures/link_fail_root_debug.png)

1. 每个rank在建链失败后会启动监听能够响应所有rank故障探测链路的server端。
2. 向无法响应自己业务建链请求的远端发起故障探测链路连接请求。
3. 如果远端无法响应自己的探测建链请求，则认为和远端的链路，或远端的业务进程存在问题，产生探测失败事件。并向已经在server端建立成功的其他链路发送扩散该事件。
4. 如果远端建立起了探测链路，则接收对端发送的探测失败事件并进行转发。

这样，如果出现任何单点问题导致的建链失败，可以通过日志快速定位故障点的节点位置，并进行下一步的问题定位。详细的定位流程可参考[建链超时（EI0006）](#建链超时ei0006)。

如果经过探测无任何事件，则很有可能是行为一致性问题，也就是每个rank均已进入建链阶段并响应其他rank的故障探测请求，但由于彼此调用的通信算子不一致导致链路互等超时，一般是由于集群行为一致性问题，请检查脚本、环境、版本、数据集等因素。如果需要参考通信算子的行为，可以通过建链失败报错日志中关键字“Alloc transports failed”中对应的tag信息推测算子行为，比如遍历每个rank的tag信息，如果16rank通信域内，15rank均为allgather，1个rank为AllReduce，则重点分析两个算子的调用逻辑差异。

针对建链超时场景，可快速判断是否为全量建链超时，若非全量建链超时，可先重点排查未上报建链超时报错的节点，可参考的命令为：

```bash
for i in *;do cd $i;pwd;grep -rnc "connection fail" | grep -v ":0" | wc -l; cd ..;done
```

### 一致性校验机制

HCCL在与对端成功创建socket链接后，会互相交换算子入参、CANN版本等信息并与本端的信息做校验，如果此时校验结果存在不一致的情况，则会在CANN日志及打屏日志中上报错误并返回错误码。详细问题定位流程可参考[参数一致性校验(EI0005)](#参数一致性校验ei0005)。

单算子模式下，为了保证性能HCCL仅在每个通信域新类型或算法的算子被首次调用时才会触发建链，由于建链成功后才会进行一致性校验，因此此特性无法拦截所有的下发不一致问题。

### 报错阶段分析

HCCL在通信算子参数面建链阶段会有以下几个常见的报错阶段场景：

- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品与Atlas A2 训练系列产品/Atlas A2 推理系列产品，device网卡端口绑定失败，可通过以下命令排查是否有端口绑定失败问题，详细信息可参考[参数面端口绑定失败（EI0019）](#参数面端口绑定失败ei0019)。

    ```bash
    grep -rE "socket type\[(0|1)\].*Please check the port status and whether the port is being used by other process"
    ```

- 参数面socket建链超时，可通过以下命令排查是否有参数面建链失败问题，详细信息可参考[建链超时（EI0006）](#建链超时ei0006)。

    ```bash
    grep -r "wait socket establish timeout"
    ```

- 通信算子一致性校验失败，可通过以下命令排查是否有一致性校验失败问题，详细信息可参考[参数一致性校验\(EI0005\)](#参数一致性校验ei0005)。

    ```bash
    grep -r "CMD information .* check fail"
    ```

## 参数面端口绑定失败（EI0019）

### 问题现象

在CANN日志中存在关键字"Please check the port status and whether the port is being used by other process."，如下所示。**此外需注意在通信域集群协商阶段也会有端口绑定失败问题，可以根据报错日志中的"socket type"判断**，若type为0或者1，则为参数面端口绑定失败，若type为2，则为通信域集群信息协商时host侧网卡端口绑定失败，可参考[server节点端口绑定失败（EI0019）](./cluster_info_nego.md#server节点端口绑定失败ei0019)。

```text
[ERROR] HCCL(1009464,all_reduce_test):2025-03-15-00:41:48.470.172 [hccl_socket.cc:110] [1009464][InitGroupStage][RanktableDetect] socket type[0], listen on ip[192.168.2.199] and specific port[16666] fail. Please check the port status and whether the port is being used by other process.
```

### 可能原因

当前rank或进程在通信算子参数面建链时需要绑定一个device侧网卡的端口，但发现端口已被其他进程占用。

### 解决方法

HCCL使用device侧网卡的端口时默认需绑定16666端口，因此若有多个进程执行在同一个device上，且均会调用HCCL的通信算子接口，那么就会出现端口已被其他进程绑定导致失败的问题。

此时可先从业务上排查多个进程跑在同一个device上是否符合任务预期，若符合任务预期结果，可通过配置HCCL_NPU_SOCKET_PORT_RANGE环境变量启用多进程场景，如：

```bash
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
```

## QP内存资源申请相关（EI0011）

在参数面建链阶段HCCL会创建QP，如果device侧内存不足会上报OOM错误。请通过调整业务配置、减少ROCE链路的使用数量，或释放部分内存解决问题。

### 问题现象

在打屏日志中存在关键字"EI0011"或"Resource_Error_Insufficient_Device_Memory"，如下所示：

```text
[PID: 2103452] 2025-11-03-20:18:46.447.213 Resource_Error_Insufficient_Device_Memory(EI0011): Failed to allocate [size: [0.25MB, 3MB], Affected by QP depth configuration.] bytes of NPU memory.
        Possible Cause: Allocation failure due to insufficient NPU memory.
        Solution: Stop unnecessary processes and ensure the required memory is available.
```

### 解决方法

调整业务配置（如batchSize）、减少ROCE链路的使用数量，或释放部分内存解决问题。

**注意：**

HCCL的其他内存申请如cclBuffer内存申请若出现OOM错误，会由drv组件上报错误码并打印错误信息，可根据报错信息或CANN日志中的堆栈判断是否为HCCL内存申请失败，若为HCCL内存申请失败，可通过配置HCCL_BUFFSIZE环境变量调整申请的内存大小。

## 建链超时（EI0006）

HCCL建链超时受环境变量[HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md)的影响，若在超时时间内对端无法响应业务建链请求，则会上报“socket timeout”，同时如果远端由于超时等故障退出，已经建好的链路在等待数据交换的过程中也可能会出现“recv fail”的报错。

### 问题现象

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

### 根据日志确认需排查的建链对端

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

需注意：

1. 当前故障链路产生探测失败事件的阈值默认为20s，用户可以通过HCCL_DFS_CONFIG环境变量中`connection_fault_detction_time`的字段进行调整，配置为0则关闭此功能。在集群规模较大或伴随严重的卡间不同步现象时，可能需要增大此配置以确保探测结果正确性。
2. 在部分复杂业务场景下，建链超时、执行超时可能同时出现在单次业务中，需要基于探测结果进行多次跳转才能定位到故障点。因此请以探测节点的日志确认是否已经到达根节点。故障根节点通常会有其他报错、或无任何异常日志，或和其他rank互等超时。

## 参数一致性校验(EI0005)

### 问题现象

在打屏日志中存在关键字"The arguments for collective communication are inconsistent between ranks"，如下所示：

```text
EI0005: 2024-04-24-06:32:27.781.599 The arguments for collective communication are inconsistent between ranks:parameter count, local end 16512, remote end 8320
        TraceBack (most recent call last):
        Transport init error. Reason: [Create] [DestLink]Create Dest error! createLink para:rank[5]-localUserrank[4]-localIpAddr[127.10.0.1], dst_rank[6]-remoteUserrank[7]-remote_ip_addr[127.10.0.1]
        Transport init error. Reason: [Create] [DestLink]Create Dest error! createLink para:rank[5]-localUserrank[4]-localIpAddr[127.10.0.1], dst_rank[4]-remoteUserrank[5]-remote_ip_addr[127.10.0.1]
        call hccl op:HcomAllReduce(HcomAllReduce) load task fail[FUNC:Distribute][FILE:hccl_task_info.cc] [LINE:329]
        [[{[node Ge0p3_0]}]]
```

或在CANN日志中存在关键字"CMD information *** check fail"，如下所示：

```text
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.640 [rank_consistentcy_checker.cc:429] [3743951][InitChannelStage][ParameterConflict]CMD information tag check fail. local[AllGather_127.10.0.1%enp_60000_0_1761379874757928], remote[AllReduce_127.10.0.1%enp_60000_0_1761379874757928]
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.666 [rank_consistentcy_checker.cc:439] [3743951][InitChannelStage][ParameterConflict]CMD information cmdType check fail. local[6], remote[2]
[ERROR] HCCL(3743927,all_reduce_test):2025-10-25-16:11:16.831.679 [rank_consistentcy_checker.cc:439] [3743951][InitChannelStage][ParameterConflict]CMD information op check fail. local[255], remote[0]
```

### 可能原因

参数面建链时，在socket建立完成后会进行两端的参数一致性校验，校验的范围包括算子标识符tag、算子类型cmdType、规约类型op、数据量count、HCCL Buffer的大小cclbufferSize、数据类型dataType等，可根据报错里的信息确定不一致的数据。例如下述示例中，两端的算子标识符tag不一致，导致通信算子在建链时一致性校验不通过，local和remote中的数据为两端不一致的数据。

其中参数不一致的两端节点信息可以通过"Transport init error! createLink para:"报错日志确认，比如执行`grep -r "Transport init error! createLink para:"`，查看结果如下：

```text
[ERROR] HCCL(3215542,all_reduce_test):2025-11-20-18:18:03.114.306 [transport_manager.cc:886] [3215599][InitChannelStage][Timeout]Transport init error! createLink para:rank[2]-localUserrank[2]-localIpAddr[127.10.0.1/2], remoteRank[1]-remoteUserrank[1]-remoteIpAddr[127.10.0.1/1], machineType[1], linkMode[1], isUsedRdma[0], tag[AllReduce_127.10.0.1%enp_60000_0_1763633852475745
```

- localUserrank：本端rank编号。
- localIpAddr：本端的节点IP信息。
- remoteUserrank：对端rank编号。
- remoteIpAddr：对端的节点IP信息。
- tag：通信算子标识符。

### 解决方法

1. 如果在未启用SuperKernel时功能正常，但启用了SuperKernel后出现初始化不一致的问题，此时建议将HCCL算子移出SuperKernel的标定范围。具体操作方法可参考《[PyTorch图模式使用指南](https://hiascend.com/document/redirect/pttorchairuseguide)》中的“max-autotune模式功能 > 图内标定SuperKernel范围”章节。
2. 根据报错信息从业务上排查参数校验不一致的两端下发的算子不一致的根因。

    **注意**：日志中部分打印为枚举值，其中cmdType为算子类型，op为规约类型，枚举值对应关系表格如下：

    | cmdType枚举值 | 算子类型 |
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

    op枚举值对应的规约类型如下表所示：

    | op枚举值 | 规约类型 |
    | --- | --- |
    | 0 | SUM |
    | 1 | PROD |
    | 2 | MAX |
    | 3 | MIN |
    | 255 | 非Reduce算子 |
