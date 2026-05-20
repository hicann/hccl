# HCCL_ENTRY_LOG_ENABLE

## 功能描述

此环境变量用于控制是否实时打印通信算子的调用行为日志。

- 1：代表实时打印通信算子的调用行为日志，即调用一次通信算子，打印一条运行日志。
- 0：代表不打印通信算子的调用行为日志。

默认值为“0”。

HCCL的默认运行日志存储路径为：$HOME/ascend/log/run/plog/plog-_pid__\*.log，关于日志的详细说明可参见《[日志参考](https://hiascend.com/document/redirect/CannCommunitylogref)》。

## 配置示例

```bash
export HCCL_ENTRY_LOG_ENABLE=1
```

## 使用约束

仅用于集合通信算子的单算子调用场景。

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
