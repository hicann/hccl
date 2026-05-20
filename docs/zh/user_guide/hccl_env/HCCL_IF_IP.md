# HCCL_IF_IP

## 功能描述

当通信域的创建方式为“基于root节点信息创建”时，可通过此环境变量配置HCCL初始化时Host使用的通信IP地址。此IP地址用于与root节点通信，以完成通信域的创建**。**

格式为字符串，要求为常规IPv4或IPv6格式，目前只支持Host网卡，且只能配置一个IP地址。

HCCL按照如下优先级顺序选择Host通信网卡：

环境变量HCCL_IF_IP \> 环境变量[HCCL_SOCKET_IFNAME](HCCL_SOCKET_IFNAME.md)  \> docker/lo以外网卡（网卡名字典序升序） \> docker网卡 \> lo网卡。

> [!NOTE]说明
> 如果不配置HCCL_IF_IP或HCCL_SOCKET_IFNAME，系统将按照优先级自动选择网卡。若当前节点选择的网卡与root节点选择的网卡链路不通，将导致HCCL建链失败。

## 配置示例

```bash
export HCCL_IF_IP=10.10.10.1
```

## 使用约束

无

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
