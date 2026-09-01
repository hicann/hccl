# 通信域建链超时问题背景

HCCL在集群信息协商时（基于root节点信息创建通信域的场景），会通过server节点和每个rank节点之间建立socket连接，以相互交换本端信息的方式获取整个通信域集群信息，来完成通信域初始化。

![image](figures/topo_detect_theory.png)

1. 通信域内的root节点调用HcclGetRootinfo，会拉起一个server线程，监测一个端口，等待接收通信域内每个rank的消息；同时该接口会返回一个rootInfo变量表示server的Ip和port，交由上层框架层将rootInfo广播给通信域内的每个rank。
2. 通信域内每个rank节点调用HcclCommInitRootInfo，同时会将rootInfo作为入参输入，该接口会拉起一个client线程，通过**host网卡**与server建立socket连接，并向server发送自己的rankInfo信息，发送完毕后进入接收状态，等待server返回完成的ranktable。
3. Server在收集到全部rank的rankInfo信息后，会生成完整的rankTable信息，并发送给每一个client，这样每个rank上就有整个通信域的全部rank信息。

由于基于拓扑探测的方式需要在rank之间建立socket连接，这里使用的是**Host侧的网卡**进行socket连接及通信，且需要通信域全部rank在超时时间内同步执行。**因此，若没有正确的选择互联的Host侧网卡，通信域内下发行为不一致是比较常见的超时失败原因。**

# 通信域初始化topo探测超时（EI0015）

## 超时问题速查表<a id="超时问题速查表"></a>

当报错日志中出现`topoinfo_exchange_server.cc`或者`topoinfo_exchange_agent.cc`字段时，当前处于通信域创建阶段，根据报错日志可以跳转至对应章节进行排查：

| 常见关键词 / 线索                                | 排查章节                                   | 说明                          |
| ----------------------------------------- | -------------------------------------- | --------------------------- |
| `topo exchange server get socket timeout` | [部分rank未连接到server节点](#部分rank未连接到server节点)        | server等待所有rank socket连接     |
| `Wait timeout for sockets recv`           | [client接收ranktable超时](#client接收ranktable超时)       | client等待server发送ranktable超时 |
| `topo exchange agent get socket timeout`  | [client与server节点建立socket超时](#rank与server节点建立socket超时) | client连接server socket超时     |

**报错示例：**

```text
server节点:
[ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.176.080 [topoinfo_exchange_server.cc:314] [5590][Get][Connection]topo exchange server get socket timeout! timeout[3600 s]

client节点：
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.253.479 [topoinfo_exchange_base.cc:75] [347][Recv][ClusterInfoMsg]receive msg length from fdhandle failed, ret[9]
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.253.490 [topoinfo_exchange_agent.cc:161] [347][DetectClusterTopoInfo]call trace: hcclRet -> 4
```

## 问题排查步骤

首先找到server节点，server节点在调用HcclGetRootinfo接口后会拉起一个背景线程，在配置的超时时间内等待所有的rank来连接，​因此若在超时时间内通信域内的所有rank没有成功连接到server线程，server线程就会出现超时报错。​同时server线程在超时报错后会打印出当前已连接的rank列表，根据该信息找到未连接成功的rank，再进一步排查对应rank未能成功连接的原因。

### 部分rank未连接到server节点<a id="部分rank未连接到server节点"></a>

部分rank未连接到server节点时，server节点会等待agent节点socket连接超时，问题现象如下：

```text
[ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.176.080 [topoinfo_exchange_server.cc:314] [5590][Get][Connection]topo exchange server get socket timeout! timeout[3600 s] 
[ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.176.770 [topoinfo_exchange_server.cc:499] [5590][TopoInfoExchangeServer][DisplayConnectionedRank]total connected num is [2004],line num is [251] [ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.176.782 [topoinfo_exchange_server.cc:512] [5590][TopoInfoExchangeServer][DisplayConnectionedRank]connected rankinfo[LINE 0]: [0000000000000000],[0000000000000001],[0000000000000002],[0000000000000003],[0000000000000004],[0000000000000005],[0000000000000006],[0000000000000007]; 
... ...
[ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.231.208 [op_base.cc:1150] [340][InitCommRootInfo]Init failed, return[0x0000000005000004], rankNum[2048], rank[0], rootInfo identifier[group_name_3], server[172.16.154.221%eth0], logicDevId[-1]
 ... ... 
[ERROR] HCCL(340,python3.9):2026-07-15-19:50:49.450.492 [topoinfo_exchange_server.cc:115] [5590][TopoInfoExchangeServer][Setup]Broadcast Rank Basic Infos failed，connectFailedAgentIdList[10,17,40,79,119,310,358,682,755,812,850,860,865,925,961,989,1025,1054,1077,1164,1170,1178,1203,1224,1341,1375,1422,1469,1562,1563,1583,1629,1634,1637,1641,1650,1738,1775,1829,1840,1866,1951,1975,2007,]
```

#### 确认未连接的rank<a id="确认未连接的rank"></a>

从报错日志中找到未成功连接的rank，以示例日志为例：
connectFailedAgentIdList：表示未连接的rank编号列表
connected rankinfo：详细表示已连接的每个rank编号，示例中：`[0000000000000003]`表示rank3已成功连接
在示例中：`connectFailedAgentIdList[10,17,40,79,119, ... ...]`，选择以rank10作为突破口分析其未连接的原因。

#### 找到未成功连接rank的位置<a id="找到未成功连接rank的位置"></a>

找到未成功连接rank的日志位置，以rank10为例：
先从报错日志中找到报错的通信域，以示例日志为例：`identifier`表示通信域名称，在示例中通信域名称为`group_name_3`
再根据通信域名称和rankID查找对应rank是否有通信域创建下发记录，可通过执行如下命令在**整个集群的日志**下搜索，找到对应的rank日志路径：
`grep -rn "xxx" plog/ | grep -rn "Entry-HcclCommInit" | grep -rn "rank\[10\]"` ，其中`xxx`为通信域名称，`plog/`是集群plog日志目录。
若在集群日志中没有检索到对应rank的通信域创建接口下发日志，**则需要从业务上排查该rank未下发的原因**。

#### 排查失败rank和server节点的通信域初始化时间差

若对应rank下发了通信域创建接口，则在该节点的run/plog日志中会有一条日志记录：

```text
[INFO] HCCL(343,python3.9):2026-07-15-18:50:49.284.233 [op_base.cc:1281] [343]Entry-HcclCommInitRootInfoConfigInner:ranks[2048], rank[10], rootinfo: host ip[172.16.154.221] port[64000] nicDeploy[1] identifier[group_name_3], deviceLogicId[2]
```

得到client节点通信域下发时间点`2026-07-15-18:50:49.284.233`。

1. 确认server节点下发通信域创建的时间。

   - 确认server节点所在日志位置，一般rank0为server节点：`grep -rn "xxx" plog/ | grep -rn "Entry-HcclCommInit" | grep -rn "rank\[0\]"` ，其中`xxx`为通信域名称，`plog/`是集群plog日志目录。

      ```text
      hcclLog/run/plog/plog-340_20260715182300676.log:17282:[INFO] HCCL(340,python3.9):2026-07-15-18:50:49.175.969 [op_base.cc:1281] [340]Entry-HcclCommInitRootInfoConfigInner:ranks[2048], rank[0], rootinfo: host ip[172.16.154.221] port[64000] nicDeploy[1] identifier[group_name_3], deviceLogicId[0]
      ```

   - 在server节点日志路径下查找调用`HcclGetRootinfo`的时间点：执行`grep -rn "Entry-HcclGetRootInfo" xxxxx`，`xxxxx`是上一步找到的日志路径：
     可能会搜到多条记录，取最后一条记录的时间戳`2026-07-15-18:50:49.174.231`（通信域创建接口是串行的，则当前通信域是最后一个通信域）。

      ```text
      # grep -rn "Entry-HcclGetRootInfo" hcclLog/run/plog/plog-340_20260715182300676.log
      8849:[INFO] HCCL(340,python3.9):2026-07-15-18:49:48.236.358 [op_base.cc:806] [340]Entry-HcclGetRootInfo:rootInfo[0xffffe8730d48], deviceLogicId[0]
      8913:[INFO] HCCL(340,python3.9):2026-07-15-18:50:30.271.209 [op_base.cc:806] [340]Entry-HcclGetRootInfo:rootInfo[0xffffe8731048], deviceLogicId[0]
      17271:[INFO] HCCL(340,python3.9):2026-07-15-18:50:49.174.231 [op_base.cc:806] [340]Entry-HcclGetRootInfo:rootInfo[0xffffe8731068], deviceLogicId[0]
      ```

2. 检查该rank与server节点的下发通信域创建时间间隔是否超过超时时间。

   - 获取通信域建链的超时时间，在run日志中搜索：`grep -rn "HCCL_CONNECT_TIMEOUT" run/plog`。在本示例中client节点通信域下发时间点：`2026-07-15-18:50:49.284.233`，server节点通信域下发时间点：`2026-07-15-18:50:49.174.231`，下发时间差小于设置的超时时间3600s。
     如果通信域创建接口的下发时间差大于超时时间，**需要从业务侧排查通信域创建接口时间差过大的原因，或者调整[HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md)环境变量配置调大超时时间解决该报错**。

      ```text
      [INFO] HCCL(353,python3.9):2026-07-15-18:41:23.173.030 [externalinput.cc:382] [353]HCCL_CONNECT_TIMEOUT set by environment to [3600]s
      ```

#### 排查失败rank报错的原因

第三步排查正常时，根据[该rank的run日志位置](#找到未成功连接rank的位置)找到debug日志，debug日志和run日志关系[参考](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/maintenref/logreference/logreference_0002.html)。根据debug日志的报错现象进一步[排查](#超时问题速查表)，其中run日志和debug日志对应关系：

```text
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

### rank与server节点建立socket超时<a id="rank与server节点建立socket超时"></a>

rank与server节点建立socket超时，问题现象如下：

```text
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.979 [topoinfo_exchange_agent.cc:190] [7988][Get][Connection]topo exchange agent get socket timeout! timeout[120]
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.995 [topoinfo_exchange_agent.cc:41] [7988][TopoInfoExchangeAgent][Setup]TopoExchangeAgent: connect server[172.16.0.0 : 1888] failed
```

1. 如果是多机场景，需检查是否有通过`HCCL_SOCKET_IFNAME`环境变量指定使用的host网卡，若环境上存在多个Host网卡时，hccl默认按字典序选择网卡进行socket建链，因此容易选择到不连通的Host网卡，可以通过ifconfig查看互联的host网卡名，并通过配置`HCCL_SOCKET_IFNAME`环境变量指定host网卡重新拉起作业。

2. 若网卡选择正确，需要排查指定的端口是否连通，如按以下日志`TopoExchangeAgent: connect server`排查报错节点是否能连通172.16.0.0的1888端口（如果`ping 172.16.0.0`则默认使用22端口）。

### client接收ranktable超时<a id="client接收ranktable超时"></a>

client接收ranktable超时，问题现象如下：

```text
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.253.079 [adapter_hccp.cc:1389] [347][Recv][RaSocket]errNo[0x0000000005000013] Wait timeout for sockets recv, data[0xffffeb38eb60], size[4 Byte], recvSize[0 Byte]. Peerrank did not send the data in time. Check whether the peerrank is abnormal.
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.253.471 [hccl_socket.cc:397] [347][Recv]call trace: hcclRet -> 9
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.253.479 [topoinfo_exchange_base.cc:75] [347][Recv][ClusterInfoMsg]receive msg length from fdhandle failed, ret[9]
... ...
[ERROR] HCCL(347,python3.9):2026-07-15-19:50:49.304.157 [op_base.cc:1150] [347][InitCommRootInfo]Init failed, return[0x0000000005000004], rankNum[2048], rank[1132], rootInfo identifier[group_name_3], server[0.0.0.0], logicDevId[-1]
```

#### 确认server节点

根据报错日志中失败的通信域`identifier`字段，找到该通信域的server节点，一般rank0就是通信域的server节点。
在全量集群日志中执行以下命令，搜索日志所在位置为server节点的run日志路径，命令中`group_name_3`为本示例中通信域名称：

```text
grep -rn "group_name_3" | grep -rn "Entry-HcclCommInit" | grep -rn "rank\[0\]"
```

根据run日志路径找到debug日志，再根据debug日志中的报错进一步排查。

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

#### 排查server节点报错

1. 根据server节点上的报错信息，[确认未连接的rank](#确认未连接的rank)。
2. 若server未连接的rank中无本rank，说明本端已正常连接，需排查其他未成功连接的rank。
3. 若server未连接的rank中包含了本端，说明本端与server的socket连接存在异常，通常是host侧网络出了异常，比如发生了ARP偏移、端口拦截等行为，需进一步排查host侧网络是否存在异常配置。
   在大规模集群场景下，Master节点能够处理的并发建链数受Linux内核参数“somaxconn”和“tcp\_max\_syn\_backlog”的限制。因此，如果这两个参数的取值过小，可能导致部分客户端在连接建立过程中出现概率性提前异常退出，进而导致集群初始化失败。可以通过以下配置来调整连接数限制（所有机器的OS均需配置，包括裸机和镜像环境等），通常将该值改成`65535`：

   ```text
   sysctl -w net.core.somaxconn=65535
   sysctl -w net.ipv4.tcp_max_syn_backlog=65535
   ```
