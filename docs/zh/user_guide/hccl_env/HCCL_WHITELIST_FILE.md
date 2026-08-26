# HCCL_WHITELIST_FILE

## 功能描述

当通过HCCL_WHITELIST_DISABLE开启了通信白名单校验功能时，需要通过此环境变量配置指向HCCL通信白名单配置文件的路径，只有在通信白名单中的IP地址才允许进行集合通信。

HCCL通信白名单配置文件格式为：

```text
{ "host_ip": ["ip1", "ip2"], "device_ip": ["ip1", "ip2"] } 
```

其中：

- device_ip为预留字段，当前版本暂不支持。
- IP地址格式为点分十进制。

> [!NOTE]说明
> 白名单IP需要指定为集群通信使用的有效IP。

## 配置示例

```bash
export HCCL_WHITELIST_FILE=/home/test/whitelist
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
