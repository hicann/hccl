# rank与server节点建立socket超时（EI0015）

## 问题现象

在CANN日志中存在关键字"topo exchange agent get socket timeout!"，如下所示：

```text
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.979 [topoinfo_exchange_agent.cc:190] [7988][InitGroupStage][RanktableDetect]topo exchange agent get socket timeout! timeout[120] 
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.995 [topoinfo_exchange_agent.cc:41] [7988][TopoInfoExchangeAgent][Setup]TopoExchangeAgent: connect server[127.10.0.1 : 60000] failed
```

## 可能原因

在通过集群协商创建通信域阶段，当前rank会根据rootInfo信息与server节点的ip和port创建socket，但由于网络连通性问题导致socket建立超时。

## 解决方法

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
