# 建链超时问题背景

在调用通信算子时，HCCL会通过**参数面网络**基于TCP协议创建Socket连接，用于交换地址、通信拓扑等初始化信息。

如果出现以下情况，则其他等待建立连接的Rank会出现Socket建链超时报错。

- 部分Rank未执行到对应的通信算子，无法发起建链请求。
- 网络连通性异常，导致建链请求无法到达对端。
- 两端通信行为不一致（如TLS配置不一致、通信算子执行不同步等），导致对端无法正确响应建链请求。

由于HCCL中的通信算子按照业务执行顺序串行处理，一个通信算子的建链阻塞会导致后续通信算子无法继续执行，因此**建链超时通常会在多个Rank之间形成级联传播现象**。

例如：

- Rank4执行通信算子3，与Rank3建链超时。
- Rank3实际上被通信算子2阻塞，正在等待与Rank2完成建链。
- Rank2又被通信算子1阻塞，正在等待与Rank1建链。
- 最终根因是Rank1与Rank2之间建链失败，导致整个依赖链上的Rank依次超时。

![link_create_multi_rank_debug](figures/link_create_multi_rank_debug.png)

因此，在定位建链超时问题时，不应仅关注当前报错Rank，而应结合建链关系持续向前追踪，找到**最早发生建链失败的Rank对**，优先分析其失败原因。

HCCL提供了建链失败根节点自动定位机制来辅助快速定位：自动探测故障根节点并打印探测结果，其中打印的(srcRank, dstRank)即定位到的**失败Rank对**（详见[建链失败根节点定位结果](#建链失败根节点定位结果)，机制参见[建链失败定位思路](./param_link_stage_troubleshooting.md#建链根节点定位机制)）。

# 建链超时 （EI0006）

当发生参数面建链超时时，CANN日志中通常会出现如下关键字：`wait socket establish timeout`。

```text
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.403 [hccl_socket_manager.cc:797] [18744][Wait][LinkEstablish]wait socket establish timeout, role[1] rank[1] timeout[120 s]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.454 [hccl_socket_manager.cc:861] [18744][Wait][LinksEstablishCompleted] is failed. ret[9].
```

## 建链失败根节点定位结果

建链失败后，HCCL会启动建链失败根节点定位（详细机制参见[建链失败定位思路](./param_link_stage_troubleshooting.md#建链根节点定位机制)）；探测完成后打印`CONNECT TIMEOUT DETECT RESULT`探测结果，其中打印的(srcRank, dstRank)即定位到的失败Rank对，探测结果同时作为EI0006的报错原因上报。探测结果的主要内容包括：

```text
This node (server 192.168.200.100, device ID 1) detects that srcRank (server 192.168.200.100, device ID 1) fails to connect to dstRank (server 192.168.200.100, device ID 0). Continue to analyze the fault based on the logs of srcRank and dstRank.
```

其中：

- `This node`：本端节点。
- `srcRank`：探测发起节点。
- `dstRank`：探测不通的对端节点，即故障根节点，优先分析该对端。
- `server`：机器标识。Atlas A3系列产品和Atlas A2系列产品为Server IP地址；Ascend 950PR&950DT系列产品为netInstId，即[rank table配置](../cluster_info_config/rank_table_config_a5.md)中net_layer为0的net_instance_id。
- `device ID`：设备物理ID（`devicePhyId`）。
- 探测结果末尾会给出排查建议：(srcRank, dstRank)两端均上报建链超时时检查网络连通性；dstRank存在其他报错时优先分析其报错；dstRank无报错时可能为业务进程挂起或提前退出。
- 未探测到异常事件时，打印`This node detects no exception event`，说明各rank之间的探测链路均正常，建链超时可能是算子下发不一致，请参考下文[排查点1](#排查点1对端没有任何异常日志)从业务侧排查。

此机制仅适用于如下产品：
<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品
<!-- end id3 -->

<!-- npu="950" id4 -->
说明
> Ascend 950PR&950DT系列产品上，建链成功后的一致性校验数据交换发生超时时，同样会触发建链失败根节点定位。
<!-- end id4 -->

不同产品的探测结果日志示例如下：

<!-- npu="A3,910b" id5 -->
Atlas A3系列产品和Atlas A2系列产品：

```text
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.949 [detect_connect_anomalies.cc:169] [18744]-------------------CONNECT TIMEOUT DETECT RESULT-----------------------
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.966 [detect_connect_anomalies.cc:172] [18744]This node (server 192.168.200.100, device ID 1) detects that srcRank (server 192.168.200.100, device ID 1) fails to connect to dstRank (server 192.168.200.100, device ID 0). Continue to analyze the fault based on the logs of srcRank and dstRank.
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.970 [detect_connect_anomalies.cc:173] [18744]1. If the link setup timeout is reported on both ends, check the network connectivity between the two ends.
2. If dstRank reports other exceptions, locate the cause based on the exception information of dstRank.
3. If dstRank does not report any error, the possible cause is that the service process is suspended or exits in advance
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.977 [detect_connect_anomalies.cc:184] [18744]----------------------------------------------------------------------
```
<!-- end id5 -->

<!-- npu="950" id6 -->
Ascend 950PR&950DT系列产品：

```text
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.949 [detect_connection_anomalies.cc:78] [18744]-------------------CONNECT TIMEOUT DETECT RESULT-----------------------
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.966 [detect_connection_anomalies.cc:88] [18744]This node (server az0-rack0-pod0, device ID 1) detects that srcRank (server az0-rack0-pod0, device ID 1) fails to connect to dstRank (server az0-rack0-pod0, device ID [0]). Continue to analyze the fault based on the logs of srcRank and dstRank.
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.970 [detect_connection_anomalies.cc:91] [18744]1. If the link setup timeout is reported on both ends, check the network connectivity between the two ends.
2. If dstRank reports other exceptions, locate the cause based on the exception information of dstRank.
3. If dstRank does not report any error, the possible cause is that the service process is suspended or exits in advance
[ERROR] HCCL(17528,python3):2026-03-18-10:34:43.141.977 [detect_connection_anomalies.cc:100] [18744]----------------------------------------------------------------------
```
<!-- end id6 -->

## 问题定位流程

### 找到建链对端

建链超时时会打印建链对端的信息：

```text
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.2.199(0)   |  16666  |   192.0.3.198(1)   |  3234403008  |  client  | time out |   DISABLE  | LinkInfo
```

从`LINK_ERROR_INFO`中获取建链两端的device ip，`dest_ip`为对端IP，`src_ip`为本端IP，示例中建链对端ip为`192.0.2.199`
通过对端ip地址找到对端日志路径，执行指令`grep -rni "localIp\[192.xx.xx.xxx\]"`全局搜索找到对端run日志路径(其中`192.xx.xx.xxx`为建链对端的ip地址)，再根据run日志位置找到debug日志，**排查对端debug日志中的报错**。

```text
run日志和debug日志对应：
├── debug
│   ├── device-1
│   │   ├── device-2849436_20260722090447711.log
│   └── plog
│       ├── plog-2849436_20260722090443026.log
├── run
│   ├── device-1
│   │   ├── device-2849436_20260722090449878.log
│   └── plog
│       ├── plog-2849436_20260722090443227.log
```

### 确认对端行为排查是否有卡间行为不一致

结合对端debug日志，按照以下流程逐步定位：

![debug_thinking](figures/debug_thinking.png)

#### 排查点1：对端没有任何异常日志

若对端不存在debug日志或者debug日志中没有任何ERROR信息，说明对端没有下发通信算子，因此未发起建链请求。
该场景非HCCL问题，**需从业务侧排查两端的通信算子下发行为是否一致**。

#### 排查点2：对端存在其他类型报错

若对端debug日志首先出现的是其他错误（而非参数面建链超时），则建链超时通常只是后续现象。
该场景非HCCL问题，**应优先分析对端首报错原因，再继续定位建链问题**。

#### 排查点3：对端也发生建链超时，但对象不是本端

如示例所示本端`192.0.3.198`在跟对端`192.0.2.199`建链时发生超时，而对端在跟`192.0.2.196`建链发生超时，此时需继续递归找与`192.0.2.196`建链的对端行为。**该报错非第一现场，需再按照排查流程继续排查**。
这是建链超时最典型的**级联传播**现象。

```text
# 本端建链超时日志
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.2.199(0)   |  16666  |   192.0.3.198(1)   |  3234403008  |  client  | time out |   DISABLE  | LinkInfo

# 对端建链超时日志
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.2.196(3)   |  16666  |   192.0.2.199(0)   |  3234403008  |  client  | time out |   DISABLE  | LinkInfo
```

#### 排查点4：双方互等超时

若本端和对端均在等待彼此建立连接，`192.0.2.199`和`192.0.3.198`互相建链超时。**先排查两端的报错时间是否超过了建链等待时间**。
若两端报错时间相差已经超过配置的建链等待时间（默认120秒），通常说明业务执行不同步，一端长时间未进入通信算子。**需要业务上排查两端通信算子下发超时时间的根因**
建链等待时间可通过环境变量`HCCL_CONNECT_TIMEOUT`配置，默认为120秒，可执行：`grep -r "HCCL_CONNECT_TIMEOUT" run/plog/` 确认当前配置。

```text
# 本端建链超时日志
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.2.199(0)   |  16666  |   192.0.3.198(1)   |  3234403008  |  client  | time out |   DISABLE  | LinkInfo

# 对端建链超时日志
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.646 [hccl_socket_manager.cc:623] [18744]   _________________________LINK_ERROR_INFO___________________________
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.650 [hccl_socket_manager.cc:624] [18744]   |  comm error, device[1]
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.653 [hccl_socket_manager.cc:626] [18744]   |  dest_ip(user_rank)  |   dest_port   |  src_ip(user_rank)   |   src_port   |   MyRole   |   Status   |    TlsStatus   |
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.655 [hccl_socket_manager.cc:628] [18744]   |----------------------|---------------|----------------------|--------------|------------|------------|----------------|
[ERROR] HCCL(17528,python3):2026-03-18-10:33:52.113.706 [hccl_socket_manager.cc:583] [18744]   |  192.0.3.198(1)   |  16666  |   192.0.2.199(0)   |  3234403008  |  server | time out |   DISABLE  | LinkInfo
```

#### 排查点5：双方在超时时间内互等

若双方几乎同时进入建链，但始终无法建立Socket连接，则需要进一步排查网络及配置问题。

1. 检查TLS配置是否一致。

   TLS配置不一致时，Socket握手校验失败，双方都会表现为建链超时。可以通过以下方法确认两端的tls开关：

   - 在节点的log日志中执行`grep -r "TLS SWITCH" log/run/device-*`获取tls状态：

     ```text
     run/device-0/device-2849330_20251024153927364.log:[INFO] HCCP(2988,hccp_service.bin):2025-10-24-15:39:26.133.826 [rs_ssl.c:1529]tid:2988,rs_ssl_init(1529) : TLS SWITCH (1)
     run/device-1/device-2849331_20251024153928174.log:[INFO] HCCP(30877,hccp_service.bin):2025-10-24-15:39:25.142.466 [rs_ssl.c:1529]tid:30877,rs_ssl_init(1529) : TLS SWITCH (0)
     ```

   - 检查`TLS SWITCH`字段（0：关闭，1：开启），确保通信双方TLS配置一致，[参考TLS配置](https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/hcclug/hcclug_000045.html)。

2. 检查Device网络连通性。

   - 先确认LINK_ERROR_INFO中的src_ip是Nic还是Vnic IP，在报错的节点上执行如下命令查看各网卡IP：

     ```text
     for n in {0..15}; do hccn_tool -i $n -ip -g ;done   #参数 -ip    查询NIC IP地址
     for n in {0..15}; do hccn_tool -i $n -vnic -g ;done   # 参数 -vnic 查询Vnic IP地址
     ```

     在查询得到的IP列表中匹配报错的ip地址，以及ip的索引，例如：在如下查询到的ip列表中，`192.168.2.199`是vnic IP，其索引是0（从0开始）。

     ```text
     $for n in {0..15}; do hccn_tool -i $n -ip -g ;done
     vnic link status: UP
     vnic ipaddr: 192.168.2.199
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.198
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.197
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.196
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.195
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.194
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.193
     vnic netmask: 255.255.255.0
     vnic link status: UP
     vnic ipaddr: 192.168.2.192
     vnic netmask: 255.255.255.0
     ```

   - 使用hccn_tool命令在节点ping LINK_ERROR_INFO中的dest_ip。第一步中查到是vnic ip，采用hccs ping；如果是nic ip，采用roce ping，其中`node`就是第一步中的索引。

     ```text
     hccn_tool -i {node} -ping -g address {dest_ip} #ROCE ping
     或者
     hccn_tool -i {node} -hccs_ping -g address {dest_ip}  #HCCS ping
     ```

   若两个rank之间ping不通或者有网口是down的，请联系实验室管理员排查对应网卡及交换机的配置。

3. 检查逻辑超节点配置。

   对于Atlas A3超节点场景，需要确认是否错误配置了逻辑超节点。若不同物理超节点被配置为同一逻辑超节点，HCCL可能错误选择VNIC链路通信，最终导致双方互等超时。

   可以通过如下日志确认两端的链路类型和物理超节点信息：链路类型为vnic，且两端的物理超节点ID不相同（分别是0和1），但由于配置了相同的逻辑超节点ID（logic_1），因此选择vnic链路进行通信导致超时，可以通过修改或者取消`HCCL_LOGIC_SUPERPOD_ID`配置进行修复。

    本端日志：

    ```text
    debug/plog/plog-3003627_20260205184335411.log:14:[ERROR] HCCL(3003627,scatter_test):2026-02-05-18:44:26.379.547 [transport_manager.cc:885] [3003959][TransportManager][PrintErrorInfo]local rank information: nicType[VNIC_TYPE], logicSuperPodId[logic_1], phySuperPodId[0]. Note: Do not configure ranks belonging to different physical superpod ID info a single logical superpod ID
    ```

    远端日志：

    ```text
    debug/plog/plog-3003628_20260205184354321.log:14:[ERROR] HCCL(3003628,scatter_test):2026-02-05-18:44:26.379.542 [transport_manager.cc:885] [3003959][TransportManager][PrintErrorInfo]local rank information: nicType[VNIC_TYPE], logicSuperPodId[logic_1], phySuperPodId[1]. Note: Do not configure ranks belonging to different physical superpod ID info a single logical superpod ID
    ```
