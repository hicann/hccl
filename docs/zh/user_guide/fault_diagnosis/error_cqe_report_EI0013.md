# ERROR CQE报错（EI0013）

ERROR CQE在HCCL中代表RoCE报文的重传超时，出现后必然会伴随集群卡死导致超时。HCCL会定期轮询RoCE驱动以获取其事件，用户可以通过接口**HcclGetCommAsyncError**进行查询是否有发生ERROR CQE报错。

## 问题现象

在打屏日志中会有EI0013的错误码打印，关键字为"Error ROCE CQE"，如下所示：

```text
[PID: 3448331] 2025-12-04-21:59:08.232.310 Execution Error ROCE CQE(EI0013): An error CQE occurred during operator execution. Local information: server 127.0.0.1, device ID 0, device IP 127.10.0.1. Peer information: server 127.0.0.2, device ID 1, device IP 127.10.0.2.
Possible Cause: 1. The network between two devices is abnormal. For example, the network port is intermittently disconnected.2. The peer process exits abnormally in advance. As a result, the local end cannot receive the response from the peer end.
Solution: 1. Check whether the network devices between the two ends are abnormal.2. Check whether the peer process exits first. If yes, check the cause of the process exit.
```

且在CANN日志中存在关键字"error cqe status"，如下所示：

```text
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.612 [hns_roce_lite.c:630]hns_roce_lite_handle_error_cqe(630) : error cqe status: 0x15
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.622 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000000): 0x00041580
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.627 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000001): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.630 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000002): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.634 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000003): 0x1500047c
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.637 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000004): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.640 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000005): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.644 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000006): 0x00000000
[ERROR] ROCE(2040034,alltoall_test):2025-09-15-08:38:12.776.647 [hns_roce_lite.c:747]dump_err_cqe(747) : CQ(0x10) CQE(0x5) INDEX(0x00000007): 0x00000000
[ERROR] HCCP(2040034,alltoall_test):2025-09-15-08:38:12.776.650 [ra_hdc_lite.c:794]tid:2040458,ra_hdc_lite_period_poll_cqe : [create][ra_hdc_period_poll]failed CQE status[12], wr[0]
[ERROR] HCCL(2040034,alltoall_test):2025-09-15-08:38:13.607.432 [heartbeat.cc:1229] [2040666][TaskExecStage][HeartbeatAbnormal][ROCE CQE ERROR]cqe error status[12], time:[2025-09-15 08:38:12.776654],localInfo{server[127.10.0.1],deviceId[127.10.0.1],deviceIp[127.11.0.1]}, remoteIP{server[127.10.0.2],deviceId[127.10.0.2],deviceIp[127.11.0.2]}
```

## 可能原因

发生ERROR CQE的问题根因在于本端给对端发包后在指定的时间段内没有收到对端的确认回复，本端就会有ERROR CQE报错上报，此时表明本端和对端之间的网络通道出现异常或者对端已断开连接或者连接状态差，无法响应，除了网络因素外，对端的进程异常退出也会导致本端收不到回复从而有ERROR CQE报错。

## 解决方法

首先可根据报错信息确认ERROR CQE远端。

```text
[ERROR] HCCL(2040034,alltoall_test):2025-09-15-08:38:13.607.432 [heartbeat.cc:1229] [2040666][TaskExecStage][HeartbeatAbnormal][ROCE CQE ERROR]cqe error status[12], time:[2025-09-15 08:38:12.776654],localInfo{server[127.10.0.1],deviceId[127.10.0.1],deviceIp[127.11.0.1]}, remoteIP{server[127.10.0.2],deviceId[127.10.0.2],deviceIp[127.11.0.2]}
```

其中，localIP和remoteIP分别代表了本端和远端的device ip，请基于硬件资源信息找到对应的rank所在计算节点或日志。

1. 排查是否有网络问题，可通过hccn_tool工具查询是否有网口闪断记录，如下结果表示网口在10:13:50 2025时发生了端口断链，此时若有集合通信算子执行，则会有ERROR CQE产生，需要进一步排查网口闪断的原因。

    ```bash
    $ hccn_tool -i 0 -link_stat -g
    [devid 0]current time        : Tue Oct 28 21:46:46 2025
    [devid 0]link up count       : 2
    [devid 0]link down count     : 1
    [devid 0]link change records :
    [devid 0]    Sun Oct  5 10:13:51 2025    LINK UP
    [devid 0]    Sun Oct  5 10:13:50 2025    LINK DOWN
    [devid 0]    Sun Oct  5 10:13:35 2025    LINK UP
    ```

2. 排查对端的业务进程在本端发生ERROR CQE时是否先异常退出或已进入资源销毁流程，可以通过观察对端的业务日志或者plog日志确认对端进程的异常退出时间是否在本端发生ERROR CQE前。
3. 若环境变量配置的HCCL_RDMA_TIMEOUT重传超时时间及HCCL_RDMA_RETRY_CNT重传次数比较小，在链路状态不佳时容易出现ERROR CQE错误，将环境变量调大即可。

其中，status\[12\]代表着RoCE报文重传超时，其他状态码极为少见，遇到后请联系技术支持。
