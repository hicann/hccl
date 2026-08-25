# HCCL_SOCKET_IFNAME配置错误 (EI0001)

## 问题现象

在CANN日志中存在关键字"get host ip fail by socket Ifname"，如下所示：

```text
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.432 [sal.cc:501] [925892][InitGroupStage][EnvConfig]set ifname to [abc] by HCCL_SOCKET_IFNAME, but not found in the environment, ifnames in the environment is as follows
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.437 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[lo] ip[127.10.0.1%lo]
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.441 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[enp] ip[127.10.0.2%enp]
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.447 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[docker0] ip[172.17.0.1%docker0]
```

## 问题根因

通过HCCL_SOCKET_IFNAME环境变量指定了Host网卡，但在当前的环境上没有找到对应的网卡（若为容器场景需指定容器内可用的Host网卡），报错日志打印列举了当前环境上查询到的Host网卡。

## 解决方法

修改HCCL_SOCKET_IFNAME环境变量，指定为环境上存在的Host网卡。
