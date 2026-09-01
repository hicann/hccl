# 部分rank未连接到server节点（EI0015）

## 问题现象

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

## 可能原因

server节点在调用HcclGetRootinfo接口后会拉起一个背景线程等待所有的rank来连接，直到超时时间为止。因此若在超时时间内，通信域的所有rank没有成功连接到server线程，server线程会等待超时报错。同时server线程在超时报错后会打印出当前已连接的rank列表，便可根据该信息找到未连接成功的rank，再进一步排查对应rank未能成功连接的原因。

## 解决方法

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
