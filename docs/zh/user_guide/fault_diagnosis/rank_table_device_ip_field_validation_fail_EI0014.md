# rank table文件中device_ip字段校验失败（EI0014）

## 问题现象

在CANN日志中存在关键字`the IP address(***) in the ranktable is inconsistent with the IP(***)address of the network adapter`，如下所示：

```text
[ERROR] HCCP(166192,eExecutor):2025-01-21-16:59:39.962.565 [ra_host.c:480]tid:167056,ra_rdev_init_check_ip(480) : [check][ip]fail, ret(129) the IP address(127.10.0.0) in the ranktable is inconsistent with the IP address(127.10.0.1) of the network adapter, please make sure they're consistent. num(2)
```

## 可能原因

HCCL在校验device ip时发现当前device侧获取的device ip与rank table中给当前rank配置的device ip不一致，因此校验失败。

比如在rank0上，绑定的device对应的device ip为127.10.0.1，但是在rank table中给rank0配置的device ip为127.10.0.0，导致HCCL检验失败。

## 解决方法

需检查rank table的配置与通信域中每个rank实际执行的device ip是否一致。
