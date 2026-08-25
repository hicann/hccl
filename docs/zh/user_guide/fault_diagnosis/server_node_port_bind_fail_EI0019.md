# server节点端口绑定失败（EI0019）

## 问题现象

打印日志中会有EI0019的报错，报错信息如下：

```text
[PID: 2267203] 2025-11-21-11:38:29.575.404 Communication_Error_Bind_IP_Port(EI0019): Failed to enable listening for the host network adapter socket.Reason: The IP address 192.168.1.100 and port 50001 have already been bound.
```

在CANN日志中存在关键字"socket type\[2\], \*\*\* Please check the port status and whether the port is being used by other process."，如下所示。**此外需注意在通信算子下发时参数面建链阶段也会有端口绑定失败问题，可以根据报错日志中的"socket type"判断，type为2，则为通信域集群协商时host侧网卡端口绑定失败。针对Atlas A3 训练系列产品/Atlas A3 推理系列产品与Atlas A2 训练系列产品/Atlas A2 推理系列产品，若type为0或者1，则为参数面端口绑定失败，可参考**[参数面端口绑定失败（EI0019）](./param_port_bind_fail_EI0019.md#参数面端口绑定失败ei0019)。

```text
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.860 [hccl_socket.cc:110] [3626636][InitChannelStage][RanktableDetect] socket type[2], listen on ip[192.168.1.100%enp53s0f2] and specific port[60000] fail. Please check the port status and whether the port is being used by other process.
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.869 [topoinfo_detect.cc:744] [3626636][InitGroupStage][RanktableDetect]StartRootNetwork failed, ret[7]
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.874 [topoinfo_detect.cc:233] [3626636][InitGroupStage][RanktableDetect]SetupServer failed, hostIP[192.168.1.100%enp53s0f2] and hostPort[60000] ret[7]
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.882 [op_base.cc:1071] [3626636][InitGroupStage][RanktableDetect]HcclGetRootInfo failed, ret[7]
```

## 可能原因

HCCL端口绑定失败，在通信域创建阶段，HCCL需要默认绑定60000-60031端口，若此时该端口已被绑定，则HCCL会绑定端口失败从而导致通信域创建失败。

## 解决方法

可以通过如下方式配置端口范围：

- 通过[HCCL_IF_BASE_PORT](../hccl_env/HCCL_IF_BASE_PORT.md)环境变量指定Host网卡的起始端口号及设置端口预留范围。
- 若业务上需要在单个NPU上同时执行多个进程，需通过[HCCL_HOST_SOCKET_PORT_RANGE](../hccl_env/HCCL_HOST_SOCKET_PORT_RANGE.md)设置HCCL在Host侧使用的通信端口范围来避免多进程之间端口使用冲突。**该操作仅适用于如下产品**：
    <!-- npu="A3" id2 -->
  - Atlas A3 训练系列产品/Atlas A3 推理系列产品
    <!-- end id2 -->
  <!-- npu="910b" id3 -->
  - Atlas A2 训练系列产品/Atlas A2 推理系列产品
    <!-- end id3 -->
