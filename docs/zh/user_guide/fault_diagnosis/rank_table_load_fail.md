# rank table文件加载失败

## 定位思路

基于rank table创建通信域的方式需要加载rank table文件，若文件路径不存在、无权限或文件的格式、配置错误，HCCL则会加载失败，报错返回。

后续内容为一些常见的rank table文件校验失败报错案例，若未找到对应案例可根据实际的报错信息进行定位排查。

## rank table文件读取失败（EI0004）

### 问题现象

在CANN日志中存在关键字"is not a valid real path"，如下所示：

```text
[ERROR] HCCL(1104629,test_one_side):2025-10-28-17:45:13.679.684 [param_check.cc:66] [1104629][InitGroupStage][RanktableConfig]errNo[0x0000000005010001] path /ranktable.json is not a valid real path
```

### 可能原因

基于rank table文件初始化通信域时，传入的rank table文件路径不存在或者权限不足。

### 解决方法

修改正确的rank table文件路径或者配置正确的可读权限。

## rank table字段配置错误（EI0004）

### 问题现象

针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，在CANN日志中存在关键字“RanktableCheck”，如下所示：

```text
[ERROR] HCCL(1265,):2025-10-21 07:56:47.198.454 [topoinfo_ranktableConcise.cc:727][15326][InitGroupStage][RanktableCheck]errNo[0x0000000005010001] super_device_id[] is invalid
```

### 可能原因

rank table的"version"字段为"1.2"，但rank table里"super_device_id"字段填写为空，导致rank table校验失败。

### 解决方法

在rank table文件中补充"super\_device\_id"字段，配置说明可参考[rank table配置资源信息（Atlas A3 训练系列产品/Atlas A3 推理系列产品）](../cluster_info_config/rank_table_config_a3.md)。

## rank table文件中device\_ip字段校验失败（EI0014）

### 问题现象

在CANN日志中存在关键字`the IP address(***) in the ranktable is inconsistent with the IP(***)address of the network adapter`，如下所示：

```text
[ERROR] HCCP(166192,eExecutor):2025-01-21-16:59:39.962.565 [ra_host.c:480]tid:167056,ra_rdev_init_check_ip(480) : [check][ip]fail, ret(129) the IP address(127.10.0.0) in the ranktable is inconsistent with the IP address(127.10.0.1) of the network adapter, please make sure they're consistent. num(2)
```

### 可能原因

HCCL在校验device ip时发现当前device侧获取的device ip与rank table中给当前rank配置的device ip不一致，因此校验失败。

比如在rank0上，绑定的device对应的device ip为127.10.0.1，但是在rank table中给rank0配置的device ip为127.10.0.0，导致HCCL检验失败。

### 解决方法

需检查rank table的配置与通信域中每个rank实际执行的device ip是否一致。
