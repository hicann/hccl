# 集群信息协商相关

## 业务流程及定位思路

### 业务流程

HCCL在集群信息协商时（基于root节点信息创建通信域的场景），通过server节点和每个rank节点之间建立socket连接，互相交换本端信息的方式来获取整个通信域集群信息完成通信域初始化。

![集群信息协商业务流程](figures/cluster_info_nego_flow.png)

1. server节点调用HcclGetRootInfo接口拉起监听线程。
    1. 集群中选择一个server节点，通常为通信域内的rank0节点，该节点调用HcclGetRootInfo接口。
    2. 获取host网卡的IP及端口信息生成rootInfo值，其中Host网卡的选择可通过HCCL_SOCKET_IFNAME环境变量指定，端口的选择可通过HCCL_IF_BASE_PORT及HCCL_HOST_SOCKET_PORT_RANGE环境变量指定。
    3. server节点会根据IP和端口进行绑定和监听，并启动一个后台线程等待通信域内的所有agent连接，同时直接返回rootInfo，完成接口调用。

2. rank节点调用HcclCommInitRootInfo接口向server节点建立连接。
    1. 上层业务或框架层将rootInfo广播给通信域内的每个rank，每个rank节点调用HcclCommInitRootInfo接口，同时会将rootInfo作为参数输入。
    2. 通过host网卡与server建立socket连接，并向server发送自己的rankInfo信息，发送完毕后进入接收状态，等待server返回完整的集群信息。该阶段建立socket连接和等待集群信息返回的流程需在一定的超时时间内完成，超时时间可通过HCCL_CONNECT_TIMEOUT环境变量控制。

3. server节点收集全量集群信息并发送给每个rank。
    1. server节点的后台线程在收集到全部rank的rankInfo信息后，会生成完整的集群信息，并发送给rank的每个agent线程。这样，每个rank上就获取到整个通信域的全部rank信息。
    2. 同时server等待所有rank的socket连接也需在一定的超时时间内完成，超时时间可通过HCCL_CONNECT_TIMEOUT环境变量控制。

### 原因分析

由于基于root节点信息创建通信域方式需要在rank节点之间建立socket连接，这里使用的是Host侧的网卡，且需要通信域内全部rank在超时时间内同步执行。因此Host网卡的网络连通问题或部分rank没有正确建立socket连接是常见的导致协商失败原因，问题定位的首要任务是要找到异常rank。

集群信息协商阶段失败的常见原因及关键日志：

- server节点的端口绑定失败，可通过以下命令排查是否有端口绑定失败问题，详细信息可参考[server节点端口绑定失败（EI0019）](#server节点端口绑定失败ei0019)。

    ```bash
    grep -r "socket type\[2\].*Please check the port status and whether the port is being used by other process"
    ```

- server节点未收到通信域下全部rank的socket连接，可通过以下命令快捷查询，详细信息可参考[部分rank未连接到server节点（EI0015）](#部分rank未连接到server节点ei0015)。

    ```bash
    grep -r "topo exchange server get socket timeout"
    ```

- rank节点与server节点建立socket超时，可通过以下命令快捷查询，详细信息可参考[rank与server节点建立socket超时（EI0015）](#rank与server节点建立socket超时ei0015)。

    ```bash
    grep -r "topo exchange agent get socket timeout"
    ```

### 关键日志信息

- 在通信域的创建环节中，HCCL会在调用通信域创建接口时记录关键日志信息，日志记录存储在CANN日志下run目录的plog文件中，可以根据对应的日志判断某个进程是否调用了对应的通信域创建接口，详细日志信息可参考[HCCL相关日志说明](./debug_thinking.md#hccl相关日志说明)。
- 通信域协商阶段如果有rank未成功建立起与root节点的socket连接，root节点会在超时退出前打印已连接的rank信息，同时向已经建链成功的rank广播缺失的rank信息，可根据该信息找到未连接的rank进一步确认问题根因，详细日志信息可参考[部分rank未连接到server节点（EI0015）](#部分rank未连接到server节点ei0015)。

## server节点端口绑定失败（EI0019）

### 问题现象

打印日志中会有EI0019的报错，报错信息如下：

```text
[PID: 2267203] 2025-11-21-11:38:29.575.404 Communication_Error_Bind_IP_Port(EI0019): Failed to enable listening for the host network adapter socket.Reason: The IP address 192.168.1.100 and port 50001 have already been bound.
```

在CANN日志中存在关键字"socket type\[2\], \*\*\* Please check the port status and whether the port is being used by other process."，如下所示。**此外需注意在通信算子下发时参数面建链阶段也会有端口绑定失败问题，可以根据报错日志中的"socket type"判断，type为2，则为通信域集群协商时host侧网卡端口绑定失败。针对Atlas A3 训练系列产品/Atlas A3 推理系列产品与Atlas A2 训练系列产品/Atlas A2 推理系列产品，若type为0或者1，则为参数面端口绑定失败，可参考**[参数面端口绑定失败（EI0019）](./param_link_stage.md#参数面端口绑定失败ei0019)。

```text
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.860 [hccl_socket.cc:110] [3626636][InitChannelStage][RanktableDetect] socket type[2], listen on ip[192.168.1.100%enp53s0f2] and specific port[60000] fail. Please check the port status and whether the port is being used by other process.
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.869 [topoinfo_detect.cc:744] [3626636][InitGroupStage][RanktableDetect]StartRootNetwork failed, ret[7]
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.874 [topoinfo_detect.cc:233] [3626636][InitGroupStage][RanktableDetect]SetupServer failed, hostIP[192.168.1.100%enp53s0f2] and hostPort[60000] ret[7]
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.882 [op_base.cc:1071] [3626636][InitGroupStage][RanktableDetect]HcclGetRootInfo failed, ret[7]
```

### 可能原因

HCCL端口绑定失败，在通信域创建阶段，HCCL需要默认绑定60000-60031端口，若此时该端口已被绑定，则HCCL会绑定端口失败从而导致通信域创建失败。

### 解决方法

可以通过如下方式配置端口范围：

- 通过[HCCL_IF_BASE_PORT](../hccl_env/HCCL_IF_BASE_PORT.md)环境变量指定Host网卡的起始端口号及设置端口预留范围。
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品与Atlas A2 训练系列产品/Atlas A2 推理系列产品，若业务上需要在单个NPU上同时执行多个进程，需通过[HCCL_HOST_SOCKET_PORT_RANGE](../hccl_env/HCCL_HOST_SOCKET_PORT_RANGE.md)设置HCCL在Host侧使用的通信端口范围来避免多进程之间端口使用冲突。

## 部分rank未连接到server节点（EI0015）

### 问题现象

在CANN日志中存在关键字"topo exchange server get socket timeout!"或"Failed to connect agent"。

- server节点问题现象：

    ```text
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.624.966 [topoinfo_exchange_server.cc:314] [1041362][InitGroupStage][RanktableDetect]topo exchange server get socket timeout! timeout[120 s]
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.625.103 [topoinfo_exchange_server.cc:501] [1041362][InitGroupStage][DisplayConnectionedRank]total connected num is [4],line num is [1]
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.625.112 [topoinfo_exchange_server.cc:503] [1041362][InitGroupStage][DisplayConnectionedRank]need connect rankNum is [8]
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.625.145 [topoinfo_exchange_server.cc:517] [1041362][InitGroupStage][DisplayConnectionedRank]connected rankinfo[LINE 0]: [0000000000000000],[0000000000000002],[0000000000000004],[0000000000000006];
    ```

- agent节点问题现象：

    ```text
    [ERROR] HCCL(1041085,all_reduce_test):2025-11-21-01:20:01.630.122 [topoinfo_exchange_base.cc:145] [1041085][InitGroupStage][RanktableDetect] TopoDetect ERROR occur !!! fault_type[1], fault_info["Failed to connect agent[1,3,5,7,]"]
    [ERROR] HCCL(1041085,all_reduce_test):2025-11-21-01:20:01.630.557 [topoinfo_exchange_agent.cc:552] [1041085][InitGroupStage][RanktableDetect]rank num[8]is different with rank list size[4] in total topo rank info.
    ```

### 可能原因

server节点在调用HcclGetRootinfo接口后会拉起一个背景线程等待所有的rank来连接，直到超时时间为止。因此若在超时时间内，通信域的所有rank没有成功连接到server线程，server线程会等待超时报错。同时server线程在超时报错后会打印出当前已连接的rank列表，便可根据该信息找到未连接成功的rank，再进一步排查对应rank未能成功连接的原因。

### 解决方法

部分rank未连接问题定位思路如下图所示：

![部分rank未连接问题定位思路](figures/rank_disconnect_debug.png)

1. 从报错信息中确认未连接的rank。
    - **server节点**：当server节点等到超时后，会打印出已连接的rank信息，由于通信域内的rankId顺序为\[0 \~ rankSize-1\]，可以从已连接的rank信息中计算出未连接的rank，如上面的日志用例中缺失了rank9的连接，因此需要进一步排查rank9未连接的原因。
    - **agent节点**：对于连接成功的agent会在server节点超时后收到server节点扩散的集群未连接rank报错信息，因此可以直接根据报错信息中的“Failed to connect agent”确认未连接的rank，并进一步确认对应rank未能成功连接的原因，如上面的日志用例中表明缺失了rank9的连接，因此需要进一步排查rank9未连接的原因。

2. 确认未连接rank是否有下发通信域创建接口。

    由于HCCL在通信域创建时在CANN日志的run目录有默认的日志记录打印，因此可以在**整个集群的日志**下过滤是否有对应rank的通信域下发，如执行如下命令，过滤是否有rank9的通信域创建下发记录，若有多个通信域可能会多行日志记录，可根据日志中的"identifier"通信域名确认是否为同一个通信域：

    ```bash
    grep -r "Entry-HcclCommInitRootInfoInner" | grep "rank\[9\]"
    ```

    - 若对应rank下发了通信域创建接口，根据检索结果获取缺失rank所在的节点及进程号信息，再进一步根据缺失rank的CANN日志排查相关的报错日志信息。
    - 若在集群日志中没有检索到对应rank的通信域创建接口下发日志，则需要从业务上排查该rank没有下发对应通信域创建接口的原因。

## rank与server节点建立socket超时（EI0015）

### 问题现象

在CANN日志中存在关键字"topo exchange agent get socket timeout!"，如下所示：

```text
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.979 [topoinfo_exchange_agent.cc:190] [7988][InitGroupStage][RanktableDetect]topo exchange agent get socket timeout! timeout[120] 
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.995 [topoinfo_exchange_agent.cc:41] [7988][TopoInfoExchangeAgent][Setup]TopoExchangeAgent: connect server[127.10.0.1 : 60000] failed
```

### 可能原因

在通过集群协商创建通信域阶段，当前rank会根据rootInfo信息与server节点的ip和port创建socket，但由于网络连通性问题导致socket建立超时。

### 解决方法

1. 检查host侧网络和对应端口的连通性。

    由于host侧往往会存在多个网卡，HCCL默认按照字典序选择host网卡进行socket连接，因此可能会选择到不连通的host网卡，可以通过[HCCL_SOCKET_IFNAME](../hccl_env/HCCL_SOCKET_IFNAME.md)环境变量指定要选择的网卡；若网卡选择正确，还需进一步排查指定的端口是否连通，如按以下日志需排查报错节点是否能连通127.10.0.1的60000端口。可以通过以下命令查询当前进程获取的host网卡信息：

    ```bash
    grep -r "get host ip success\|find nic.*success"
    ```

2. 检查通信域内每个agent与server节点的下发通信域创建时间间隔是否超过超时时间。

    可在CANN日志的run目录下执行`grep -r "Entry-"`来确认通信域创建接口的下发时间，或直接根据报错日志的打印时间来计算每个agent与server节点的下发通信域创建时间间隔，若时间间隔超过了超时时间，可通过[HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md)配置建链超时时间，默认为120秒。

    可通过如下命令查询每个通信域创建接口下发的时间：

    ```bash
    grep -r "Entry-HcclGetRootInfo\|Entry-HcclCommInitRootInfoInner" run/plog
    ```

    根据以下查询结果，可以看出rank\[3\]的通信域接口下发时间比其他rank慢了200秒左右，而建链的超时时间为120秒，因此最终通信域创建整个流程等待超时失败。针对该场景，若不同rank上由于业务进程有前置差异，如有的进程需要加载更多的数据则会更慢启动，因此可以通过[HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md)环境变量配置调大超时时间解决该报错。

    ```text
    [INFO] HCCL(3079955,all_reduce_test):2025-11-20-11:59:56.716.583 [op_base.cc:1293] [3079955]Entry-HcclCommInitRootInfoInner:ranks[4], rank[3], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[3]
    [INFO] HCCL(3079952,all_reduce_test):2025-11-20-11:56:36.704.523 [op_base.cc:858] [3079952]Entry-HcclGetRootInfo:rootInfo[0xaaaae85c79a0], deviceLogicId[0]
    [INFO] HCCL(3079952,all_reduce_test):2025-11-20-11:56:36.711.546 [op_base.cc:1293] [3079952]Entry-HcclCommInitRootInfoInner:ranks[4], rank[0], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[9127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[0]
    [INFO] HCCL(3079953,all_reduce_test):2025-11-20-11:56:36.712.024 [op_base.cc:1293] [3079953]Entry-HcclCommInitRootInfoInner:ranks[4], rank[1], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[1]
    [INFO] HCCL(3079954,all_reduce_test):2025-11-20-11:56:36.712.065 [op_base.cc:1293] [3079954]Entry-HcclCommInitRootInfoInner:ranks[4], rank[2], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[2]
    ```

## 典型多机场景通信域初始化失败（EI0015）

### 问题现象

三机场景下通信域创建失败案例如下图所示：

![三机场景下通信域创建失败案例](figures/3node_comm_create_fail.png)

该问题现象为一个典型的三机共24卡的通信域创建协商超时的报错日志，其中节点0为通信域的root节点，分析每个节点的报错现象：

- **节点0**：节点0为root节点，报错信息为server线程等待通信域内所有rank连接超时，可以从报错信息中获取已成功连接的rank，并反向推算出未连接的rank为rank16\~rank23。
- **节点1**：该节点属于和root节点成功创建socket连接的节点，在等待接收root节点超时后收到了root节点扩散的未连接的rank信息，可从报错日志中直接得到未连接的rank为rank16\~rank23。
- **节点2**：节点2的报错日志为与server节点建立socket超时，且问题根因在于节点2与root节点的Host侧网络配置错误导致无法连接，修改配置后问题解决。

### 定位思路

从该典型场景可以看出，当集群发生通信域创建建链超时时，无论是server节点还是已成功连接的节点，都可以从报错日志中快速确认未连接的rank，也就是报错的根节点，此时仅需可以重点排查未连接rank的失败原因即可，如常见的连接超时原因为未配置[HCCL_SOCKET_IFNAME](../hccl_env/HCCL_SOCKET_IFNAME.md)环境变量导致使用未连通的Host网卡。
