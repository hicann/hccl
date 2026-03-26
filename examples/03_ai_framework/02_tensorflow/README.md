# 使用 Tensorflow 执行 AllReduce 操作

## 样例介绍

本样例展示如何使用 TensorFlow 接口执行 AllReduce 操作，包含以下功能点：

- 基于 `ranktable.json` 配置文件初始化通信域

## 环境准备

### 环境要求

本样例支持以下产品，集群拓扑为单机 8 卡：

- <term>Ascend 950PR</term> / <term>Ascend 950DT</term>
- <term>Atlas A3 训练系列产品</term> / <term>Atlas A3 推理系列产品</term>
- <term>Atlas A2 训练系列产品</term>
- <term>Atlas 训练系列产品</term> / <term>Atlas 推理系列产品</term>

注意：本样例代码基于 TensorFlow 1.x 框架开发，不兼容 TensorFlow 2.x。推荐使用 TensorFlow 1.15.0 版本。
### 配置环境变量

```bash
# 设置 CANN 环境变量，以 root 用户默认安装路径为例
source /usr/local/Ascend/cann/set_env.sh

# 设置 rank_table.json 配置文件路径
export RANK_TABLE_FILE=ranktable.json
```

## 执行样例

```bash
bash run_tensorflow.sh
```

> 注意：可通过设置 `HCCL_OP_EXPANSION_MODE` 环境变量配置通信算子的展开模式，不同产品型号支持的范围可参考[环境变量列表](https://hiascend.com/document/redirect/CannCommunityEnvRef)中该环境变量的使用方法。
>
> ```bash
> # 设置通信算子的展开模式为AI CPU通信引擎
> export HCCL_OP_EXPANSION_MODE=AI_CPU
> ```

## 结果示例

每个 rank 的数据初始化为 0~7，经过 AllReduce 操作后，每个 rank 的结果是所有 rank 对应位置数据的和（8 个 rank 的数据相加）。

```
INFO:tensorflow:{'allreduce_sum_output': array([ 0., 8., 16., 24., 32., 40., 48., 56. ], dtype=float32)}
device:0 tensorflow hccl test success
```
