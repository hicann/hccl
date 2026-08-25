# 进程卡死或对端心跳丢失

## 问题现象

在CANN日志中存在关键字"Cluster Exception Location"，如下所示：

对端心跳丢失：

```text
[ERROR]HCCL(835695,all_reduce_test):2025-10-23-17:28:06.049.385[task_exception_handler.cc:610][835695][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 17:25:58 2025], Discoverer:[127.10.0.1/2], ExceptionType:[Heartbeat Lost Occurred], Possible Reason:1. Process has exited, 2. Network Disconnected
```

进程卡死：

```text
[ERROR]HCCL(1219039,all_reduce_test):2025-10-23-21:05:09.859.568[task_exception_handler.cc:610] [1219039][TaskExecStage][HeartbeatAbnormal]Cluster Exception Location[IP/ID]:[127.10.0.1/1], Arrival Time:[Thu Oct 23 21:03:19 2025], ExceptionType:[Stuck Occurred], Possible Reason:1.Host process is stuck, 2.Device task is stuck
```

## 可能原因及定位思路

可从报错日志中识别异常类型及异常所在的节点信息：

- Cluster Exception Location：表示异常所在的节点信息。
- Arrival Time：表示异常广播到本端的时间。
- ExceptionType：异常类型，包括心跳丢失（Heartbeat Lost Occurred）、进程卡死（Stuck Occurred）、网络丢包（Error CQE Occurred）等。
- Possible Reason：异常可能的发生原因及排查思路：
  - Heartbeat Lost Occurred：排查异常所在的节点在异常广播到本端的时间是否已经提前退出或者节点间网络异常无法连接。
  - Stuck Occurred：排查异常所在的节点的业务进程是否卡死在某个节点或者发生了死锁。
  - Error cqe Occurred：排查异常所在的节点是否发生了cqe error。
