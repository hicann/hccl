# HCCL_WHITELIST_DISABLE

## 功能描述

配置在使用HCCL时是否开启通信白名单。

- 0：开启白名单，校验HCCL通信白名单，只有在通信白名单中的IP地址才允许进行集合通信。
- 1：关闭白名单，无需校验HCCL通信白名单。

缺省值为1，默认关闭白名单。如果开启了白名单校验，需要通过[HCCL_WHITELIST_FILE](HCCL_WHITELIST_FILE.md)指定白名单配置文件路径。

## 配置示例

```bash
export HCCL_WHITELIST_DISABLE=1
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
