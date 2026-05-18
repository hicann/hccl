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

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
