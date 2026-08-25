# 定位思路

HCCL会基于已有的通信域信息与邻近的rank建立起独立的维测链路，以此提供集群的单点故障广播扩散能力（注：HCCL会控制维测链路数量和通信数据量，用户不用担心其对通信链路造成的性能损失），使任意rank的plog日志中均包含故障根节点信息。当前支持的故障探测能力请参见下表。

| 优先级 | 异常类型 | status（运行日志-run/plog） | ExceptionType（调试日志-debug/plog） | 判断标准 |
| --- | --- | --- | --- | --- |
| 1 | 网络问题 | ERROR CQE | Error cqe Occurred | 定期轮询ROCE驱动重传超次事件，通过QPN映射远端IP |
| 2 | 进程卡死 | STUCK | Stuck Occurred | 每隔1/3的HCCL_EXEC_TIMEOUT时间轮询所有算子的入/出次数分析是否卡住 |
| 3 | 进程退出 | LOST | Heartbeat Lost Occurred | 30s时间内未收到远端的心跳报文 |

为了控制打印数量，当前Cluster Exception ERROR日志只打印有效事件的前三个，且优先级为ERROR CQE \> STUCK \> LOST，若需要确认全量的心跳事件，可在run目录日志中检索。

- HCCL在检测到异常事件后，会在集群中进行信息的扩散转发，HCCL在运行过程中会将收到的异常事件打印到run日志。

    **日志格式为：**\[HeartbeatAbnormal\]local rank\[IP/ID\]：crimer rank\[IP/ID\] status\[4异常事件类型\]by informer rank\[IP/ID\]。

  - HeartbeatAbnormal：代表为心跳异常事件。
  - local rank：当前节点的信息。
  - crimer rank：根节点信息。
  - status：异常事件类型。
  - by informer rank：集群故障上报者信息。

    用户可结合关键字“HeartbeatAbnormal”与status状态进行检索，日志示例如下：

    ```text
    [INFO] HCCL(686,python):2025-10-23-07:52:59.191.363 [heartbeat.cc:951] [8970][TaskExecStage][HeartbeatAbnormal]local rank [127.10.0.1/1]: crimer rank [127.10.0.2/2] status[LOST] by informer rank [127.10.0.3/3]
    ```

- 如果后续出现算子执行报错，并且调用了task exception回调函数通知了HCCL，HCCL会根据已经收到的异常事件，结合HCCL_EXEC_TIMEOUT等超时事件配置，推测出最可能的单点故障原因，打印在ERROR日志中。

    **日志格式为**：\[TaskExecStage\]\[HeartbeatAbnormal\]Cluster Exception Location\[IP/ID\], Arrival Time:\[星期 月 日 时:分:秒 年\], Discoverer:\[IP/ID\], ExceptionType:\[异常类型\], Possible Reason:可能原因。

  - \[TaskExecStage\]\[HeartbeatAbnormal\]：代表集群故障发生在算子执行阶段，为心跳异常事件。
  - Cluster Exception Location：集群故障发生位置。
  - Arrival Time：集群故障发生时间。
  - Discoverer：集群故障发现节点。
  - ExceptionType：集群故障的异常类型。
  - Possible Reason：集群故障发生的可能原因。

    用户可检索关键字“HeartbeatAbnormal”，日志示例如下所示：

    ```text
    [ERROR]HCCL(835695,all_reduce_test):2025-10-23-17:28:06.049.385[task_exception_handler.cc:610][835695][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 17:25:58 2025], Discoverer:[127.10.0.1/2], ExceptionType:[Heartbeat Lost Occurred], Possible Reason:1. Process has exited, 2. Network Disconnected
    ```

如果超时后未伴随异常事件，则有可能为集群行为一致性问题，请优先排查脚本、版本、数据集等因素，如果有需要，可以通过开启HCCL_ENTRY_LOG_ENABLE环境变量进行算子级行为跟踪。

> [!NOTE]说明
>
> 1. 如果训练/推理任务被notify超时前提前杀掉，或task exception机制由于某种原因未及时调用callback函数通知HCCL，使HCCL没有打印异常信息。用户依然可以通过run日志中系统运行过程中的异常事件进行根节点定位。此时需要对异常事件进行甄别，一般我们认为，对于系统卡住时间附近的LOST/ERROR CQE事件即为导致系统停止的原因，而STUCK检测时间为（1/3\~2/3 ）\* HCCL_EXEC_TIMEOUT，需要注意。
> 2. 网络异常和进程退出均有可能会同时导致LOST和ERROR CQE事件，请结合心跳事件具体情况来看，比如：如果两端rank互报对端LOST。
