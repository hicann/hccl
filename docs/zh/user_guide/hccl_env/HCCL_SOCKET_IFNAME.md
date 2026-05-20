# HCCL_SOCKET_IFNAME

## 功能描述

配置HCCL初始化时Host使用的通信网卡名，HCCL将通过该网卡名获取Host IP，与root节点通信，以完成通信域的创建**。**

开发者可以选取以下规则中的一种进行配置：

- eth：使用所有以eth为前缀的网卡。

    若指定多个网卡前缀，多个网卡前缀间用英文逗号分隔。

    例如：export HCCL_SOCKET_IFNAME=eth,enp，表示使用所有以eth或enp为前缀的网卡。

- ^eth：不使用以eth为前缀的网卡。

    若指定多个网卡前缀，多个网卡前缀间用英文逗号分隔。

    例如：export HCCL_SOCKET_IFNAME=^eth,enp，表示不使用任何以eth或enp为前缀的网卡。

- =eth0：使用指定的eth0网卡。

    若指定多个网卡，多个网卡间用英文逗号分隔。

    例如：export HCCL_SOCKET_IFNAME==eth0,enp0，表示使用eth0网卡或enp0网卡。

- ^=eth0：不使用指定eth0网卡。

    若指定多个网卡，多个网卡间用英文逗号分隔。

    例如：export HCCL_SOCKET_IFNAME=^=eth0,enp0，表示不使用eth0与enp0网卡。

> [!NOTE]说明
>
> - HCCL_SOCKET_IFNAME中可配置多个网卡，取最先匹配到的网卡作为通信网卡。
> - 环境变量[HCCL_IF_IP](HCCL_IF_IP.md)的优先级高于HCCL_SOCKET_IFNAME。
> - 如果用户未指定HCCL_IF_IP和HCCL_SOCKET_IFNAME，按照如下优先级选择：
>    docker/lo以外网卡（网卡名字典序升序） \> docker网卡 \> lo网卡
>
> 如果不配置HCCL_IF_IP或HCCL_SOCKET_IFNAME，系统将按照优先级自动选择网卡。若当前节点选择的网卡与root节点选择的网卡链路不通，将导致HCCL建链失败。

## 配置示例

```bash
# 使用eth0或endvnic的网卡
export HCCL_SOCKET_IFNAME==eth0,endvnic
```

## 使用约束

无

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
