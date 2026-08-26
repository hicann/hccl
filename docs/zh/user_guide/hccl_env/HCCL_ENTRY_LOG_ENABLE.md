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
