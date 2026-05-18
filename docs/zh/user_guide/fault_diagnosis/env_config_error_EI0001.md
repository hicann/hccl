# 环境变量配置异常（EI0001）

## 定位思路

在业务的日志中出现**"EI0001"**故障码意味着HCCL的环境变量配置异常，一般情况下打印日志的ERROR MESSAGE和CANN日志中会显示配置异常的环境变量名称及错误原因，以及合理的配置范围，如果有疑问请参照[环境变量参考](../hccl_env/README.md)。

## HCCL_RDMA_SL配置错误 (EI0001)

### 问题现象

在打印日志中存在关键字`EI0001`或`Value *** for environment variable *** is invalid`，如下所示：

```text
[PID:3729526]2025-10-23-17:30:40.098.984Config_Error_Invalid_Environment_Variable(EI0001): Value 1000 for environment variable HCCL_RDMA_SL is invalid. Expected value : range[0, 7].
```

针对Atlas 推理系列产品、Atlas 训练系列产品、Atlas A2 训练系列产品/Atlas A2 推理系列产品、Atlas A3 训练系列产品/Atlas A3 推理系列产品，CANN日志的ERROR日志中存在关键字"externalinput.cc"，表示是在读取环境变量配置时报错，如下所示：

```text
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.098.973 [externalinput.cc:963] [3729526][Parse][rdmaServerLevel]HCCL_RDMA_SL[1000] is invalid. except: [0, 7]
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.058 [externalinput.cc:169] [3729526][InitGroupStage][EnvConfig]errNo[0x0000000005000001] In init env variable param, parse HCCL_RDMA_SL failed. errno[1]
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.063 [externalinput.cc:47] [3729526][InitExternalInput]call trace: hcclRet -> 1
[ERROR]HCCL(3729526,python3.11):2025-10-23-17:30:40.099.068 [op_base.cc:866] [3729526][HcclGetRootInfo]call trace: hcclRet -> 1
```

### 可能的原因及解决方法

环境变量配置参数不符合要求，请基于日志打印的建议调整取值范围，如果仍然有疑问，请参照对应[环境变量参考](../hccl_env/README.md)。

## HCCL_SOCKET_IFNAME配置错误 (EI0001)

### 问题现象

在CANN日志中存在关键字"get host ip fail by socket Ifname"，如下所示：

```text
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.432 [sal.cc:501] [925892][InitGroupStage][EnvConfig]set ifname to [abc] by HCCL_SOCKET_IFNAME, but not found in the environment, ifnames in the environment is as follows
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.437 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[lo] ip[127.10.0.1%lo]
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.441 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[enp] ip[127.10.0.2%enp]
[ERROR] HCCL(925892,alltoall_test):2025-10-28-16:34:59.634.447 [sal.cc:504] [925892][InitGroupStage][EnvConfig]get host ip fail by socket Ifname. name[docker0] ip[172.17.0.1%docker0]
```

### 问题根因

通过HCCL_SOCKET_IFNAME环境变量指定了Host网卡，但在当前的环境上没有找到对应的网卡（若为容器场景需指定容器内可用的Host网卡），报错日志打印列举了当前环境上查询到的Host网卡。

### 解决方法

修改HCCL_SOCKET_IFNAME环境变量，指定为环境上存在的Host网卡。
