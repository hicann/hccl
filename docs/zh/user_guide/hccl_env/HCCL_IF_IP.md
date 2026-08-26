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

## 产品支持情况

<!-- npu="950" id3 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id3 -->
<!-- npu="A3" id4 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id4 -->
<!-- npu="910b" id5 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id5 -->
<!-- npu="910" id1 -->
- Atlas 训练系列产品：支持
<!-- end id1 -->
<!-- npu="310p" id2 -->
- Atlas 推理系列产品：支持
<!-- end id2 -->
