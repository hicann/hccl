# 业务流程及定位思路

## 业务流程

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

## 原因分析

由于基于root节点信息创建通信域方式需要在rank节点之间建立socket连接，这里使用的是Host侧的网卡，且需要通信域内全部rank在超时时间内同步执行。因此Host网卡的网络连通问题或部分rank没有正确建立socket连接是常见的导致协商失败原因，问题定位的首要任务是要找到异常rank。

集群信息协商阶段失败的常见原因及关键日志：

- server节点的端口绑定失败，可通过以下命令排查是否有端口绑定失败问题，详细信息可参考[server节点端口绑定失败（EI0019）](./server_node_port_bind_fail_EI0019.md#server节点端口绑定失败ei0019)。

    ```bash
    grep -r "socket type\[2\].*Please check the port status and whether the port is being used by other process"
    ```

- server节点未收到通信域下全部rank的socket连接，可通过以下命令快捷查询，详细信息可参考[部分rank未连接到server节点（EI0015）](./partial_rank_not_connect_to_server_EI0015.md#部分rank未连接到server节点ei0015)。

    ```bash
    grep -r "topo exchange server get socket timeout"
    ```

- rank节点与server节点建立socket超时，可通过以下命令快捷查询，详细信息可参考[rank与server节点建立socket超时（EI0015）](./rank_server_socket_timeout_EI0015.md#rank与server节点建立socket超时ei0015)。

    ```bash
    grep -r "topo exchange agent get socket timeout"
    ```

## 关键日志信息

- 在通信域的创建环节中，HCCL会在调用通信域创建接口时记录关键日志信息，日志记录存储在CANN日志下run目录的plog文件中，可以根据对应的日志判断某个进程是否调用了对应的通信域创建接口，详细日志信息可参考[HCCL相关日志说明](./debug_thinking.md#hccl相关日志说明)。
- 通信域协商阶段如果有rank未成功建立起与root节点的socket连接，root节点会在超时退出前打印已连接的rank信息，同时向已经建链成功的rank广播缺失的rank信息，可根据该信息找到未连接的rank进一步确认问题根因，详细日志信息可参考[部分rank未连接到server节点（EI0015）](./partial_rank_not_connect_to_server_EI0015.md#部分rank未连接到server节点ei0015)。
