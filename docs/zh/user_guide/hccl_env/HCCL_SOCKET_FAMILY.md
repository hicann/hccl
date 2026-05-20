# HCCL_SOCKET_FAMILY

## 功能描述

该环境变量指定通信网卡使用的IP协议，支持如下两种配置：

- AF_INET：代表使用IPv4协议。
- AF_INET6：代表使用IPv6协议

**缺省使用IPv4协议。**

该环境变量有以下两种使用场景：

- 配置HCCL初始化时，Host侧通信网卡使用的IP协议版本。

  此场景下，该环境变量需要与[HCCL_SOCKET_IFNAME](HCCL_SOCKET_IFNAME.md)同时使用，当HCCL通过指定网卡名获取Host IP时，通过该环境变量指定使用网卡的socket通信IP协议。

- 配置HCCL初始化时，Device侧通信网卡使用的IP协议版本。

  此场景下，如果该环境变量指定的IP协议与实际获取到的网卡信息不匹配，则以实际环境上的网卡信息为准。

  例如，该环境变量指定为IPv6协议，但Device侧只存在IPv4协议的网卡，则实际会使用IPv4协议的网卡。

## 配置示例

```bash
export HCCL_SOCKET_FAMILY=AF_INET       #IPv4
export HCCL_SOCKET_FAMILY=AF_INET6      #IPv6
```

## 使用约束

无

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
