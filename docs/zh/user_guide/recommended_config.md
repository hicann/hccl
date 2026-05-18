# 推荐业务配置

本节分别针对Atlas A3 训练系列产品/Atlas A3 推理系列产品、Atlas A2 训练系列产品/Atlas A2 推理系列产品的常见业务场景，提供推荐的业务配置。

> [!NOTE]说明
> 本节仅给出了推荐配置环境变量的功能说明和配置示例，详细使用说明可参见[环境变量参考](./hccl_env/README.md)。

## Atlas A3 训练系列产品/Atlas A3 推理系列产品

- **训练场景**

  | 环境变量 | 配置说明 |
  | --- | --- |
  | [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md) | 配置socket建链超时等待时间，默认值为120，单位s。该场景下，建议根据网络规模大小适当调整建链超时等待时间。export HCCL_CONNECT_TIMEOUT=1200 |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | 配置通信算子的展开模式。<br>该场景下建议保持默认值“AI_CPU”，代表通信算子在AI CPU展开。<br>export HCCL_OP_EXPANSION_MODE="AI_CPU" |

- **推理场景**
  - Prefill-Decode混合部署

    | 环境变量 | 配置说明 |
    | --- | --- |
    | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | 配置通信算子的展开模式。<br>该场景下建议配置为“AIV”，代表通信算子在Vector Core展开。<br>export HCCL_OP_EXPANSION_MODE="AIV" |
    | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | 是否开启确定性计算，用户可以根据使用场景选择开启或关闭，默认值为false，代表关闭确定性计算。<br>export HCCL_DETERMINISTIC=false |

  - Prefill-Decode分离部署

    | 环境变量 | 配置说明 |
    | --- | --- |
    | [HCCL_INTRA_ROCE_ENABLE](./hccl_env/HCCL_INTRA_ROCE_ENABLE.md) | 仅使用LLM-DataDist作为集群管理组件的场景下，建议通过此环境变量配置超节点内使用RoCE链路进行通信；非LLM-DataDist场景，无需配置。<br>export HCCL_INTRA_ROCE_ENABLE=1 |
    | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | 配置通信算子的展开模式。<br>该场景下建议配置为“AIV”，代表通信算子在Vector Core展开。<br>export HCCL_OP_EXPANSION_MODE="AIV" |
    | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | 是否开启确定性计算，用户可以根据使用场景选择开启或关闭，默认值为false，代表关闭确定性计算。<br>export HCCL_DETERMINISTIC=false |

- **强化学习训推一体**

  | 环境变量 | 配置说明 |
  | --- | --- |
  | [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md) | 配置socket建链超时等待时间，默认值为120，单位s。该场景下，建议根据网络规模大小适当调整建链超时等待时间。<br>export HCCL_CONNECT_TIMEOUT=1200 |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | 配置通信算子的展开模式。<br>该场景下建议保持默认值“AI_CPU”，代表通信算子在AI CPU展开。<br>export HCCL_OP_EXPANSION_MODE="AI_CPU"<br>需要注意：<br>针对推理通信域，需要通过通信域级别的配置参数将推理通信域的算子展开位置设置为“Vector Core”，针对PyTorch框架网络，可通过“hccl_op_expansion_mode”参数配置，配置方法如下：<br>options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()<br>   options.hccl_config ={"hccl_op_expansion_mode":3}<br>   torch.distributed.init_process_group(backend="hccl", pg_options=options)<br>PyTorch框架参数的详细介绍可在《[Ascend Extension for PyTorch 产品文档](https://hiascend.com/document/redirect/pytorchuserguide)》中搜索“通过pg_options配置HCCL通信域参数”查看。 |
  | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | 是否开启确定性计算，用户可以根据使用场景选择开启或关闭，默认值为false，代表关闭确定性计算。<br>export HCCL_DETERMINISTIC=false |

## Atlas A2 训练系列产品/Atlas A2 推理系列产品

- **训练场景**

  | 环境变量 | 配置说明 |
  | --- | --- |
  | [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md) | 配置socket建链超时等待时间，默认值为120，单位s。该场景下，建议根据网络规模大小适当调整建链超时等待时间。export HCCL_CONNECT_TIMEOUT=1200 |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | 配置通信算子的展开模式 。<br>该场景下建议保持默认值“HOST”，代表通信算子在Host侧CPU展开。<br>export HCCL_OP_EXPANSION_MODE="HOST" |
  | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | 是否开启确定性计算，用户可以根据使用场景选择开启或关闭，默认值为false，代表关闭确定性计算。<br>export HCCL_DETERMINISTIC=false |

- **推理场景**

  | 环境变量 | 配置说明 |
  | --- | --- |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | 配置通信算子的展开模式。<br>该场景下建议保持默认值“HOST”，代表通信算子在Host侧CPU展开。<br>export HCCL_OP_EXPANSION_MODE="HOST" |
  | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | 是否开启确定性计算，用户可以根据使用场景选择开启或关闭，默认值为false，代表关闭确定性计算。<br>export HCCL_DETERMINISTIC=false |

- **强化学习训推一体**

  | 环境变量 | 配置说明 |
  | --- | --- |
  | [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md) | 配置socket建链超时等待时间，默认值为120，单位s。该场景下，建议根据网络规模大小适当调整建链超时等待时间。<br>export HCCL_CONNECT_TIMEOUT=1200 |
  | [HCCL_OP_EXPANSION_MODE](./hccl_env/HCCL_OP_EXPANSION_MODE.md) | 配置通信算子的展开模式。<br>该场景下建议保持默认值“HOST”，代表通信算子在Host侧CPU展开。<br>export HCCL_OP_EXPANSION_MODE="HOST" |
  | [HCCL_DETERMINISTIC](./hccl_env/HCCL_DETERMINISTIC.md) | 是否开启确定性计算，用户可以根据使用场景选择开启或关闭，默认值为false，代表关闭确定性计算。<br>export HCCL_DETERMINISTIC=false |
