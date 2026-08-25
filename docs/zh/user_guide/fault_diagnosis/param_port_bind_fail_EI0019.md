# 参数面端口绑定失败（EI0019）

## 问题现象

在CANN日志中存在关键字"Please check the port status and whether the port is being used by other process."，如下所示。**此外需注意在通信域集群协商阶段也会有端口绑定失败问题，可以根据报错日志中的"socket type"判断**，若type为0或者1，则为参数面端口绑定失败，若type为2，则为通信域集群信息协商时host侧网卡端口绑定失败，可参考[server节点端口绑定失败（EI0019）](./server_node_port_bind_fail_EI0019.md#server节点端口绑定失败ei0019)。

```text
[ERROR] HCCL(1009464,all_reduce_test):2025-03-15-00:41:48.470.172 [hccl_socket.cc:110] [1009464][InitGroupStage][RanktableDetect] socket type[0], listen on ip[192.168.2.199] and specific port[16666] fail. Please check the port status and whether the port is being used by other process.
```

## 可能原因

当前rank或进程在通信算子参数面建链时需要绑定一个device侧网卡的端口，但发现端口已被其他进程占用。

## 解决方法

HCCL使用device侧网卡的端口时默认需绑定16666端口，因此若有多个进程执行在同一个device上，且均会调用HCCL的通信算子接口，那么就会出现端口已被其他进程绑定导致失败的问题。

此时可先从业务上排查多个进程跑在同一个device上是否符合任务预期，若符合任务预期结果，可通过配置HCCL_NPU_SOCKET_PORT_RANGE环境变量启用多进程场景，如：

```bash
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
```
