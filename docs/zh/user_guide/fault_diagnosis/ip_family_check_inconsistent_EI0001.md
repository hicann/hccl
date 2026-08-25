# IP Family校验不一致 （EI0001）

## 问题现象

在CANN日志中存在关键字"rank\[\*\] device ip family\[2\] is not same as others\[\*\]."，如下所示：

```text
[ERROR] HCCL(144905,python):2025-04-20-00:26:54.435.048 [config.cc:413] [145735][InitGroupStage][RanktableCheck]rank[0] device ip family[2] is not same as others[10].
```

## 可能原因

两个rank获取到的IP Family不同，比如一边是IPv4，而另一边是IPv6。

## 解决方法

查询是否配置了IPv4：

```bash
hccn_tool -i {deviceId} -ip -g
```

查询是否配置了IPv6：

```bash
hccn_tool -i {deviceId} -ip -inet6 -g
```

同一次作业的所有rank的IP Family应保持一致。HCCL默认先使用IPv4协议，若Device侧没有配置IPv4协议的IP，则会使用IPv6协议对应的ip。可以使用HCCL_SOCKET_FAMILY环境变量指定需要使用的网卡IP协议。

**注意**：family打印为枚举值，枚举值及对应关系如下表所示。

| IP Family枚举值 | IP协议 |
| --- | --- |
| 2 | IPv4 |
| 10 | IPv6 |
