# HCCL_INTRA_ROCE_ENABLE

## 功能描述

用于配置Server内或超节点内是否使用RoCE链路进行通信。

- 针对<cann-filter npu-type="910">Atlas 训练系列产品与</cann-filter>Atlas A2 训练系列产品/Atlas A2 推理系列产品，该环境变量用于配置Server内是否使用RoCE链路进行通信，默认值0，可以单独配置，也可以与环境变量HCCL_INTRA_PCIE_ENABLE同时使用。支持的配置组合，以及不同组合下Server内使用的通信链路如下表所示：

   HCCL_INTRA_PCIE_ENABLE与HCCL_INTRA_ROCE_ENABLE支持的配置组合如下表所示：

  | HCCL_INTRA_PCIE_ENABLE | HCCL_INTRA_ROCE_ENABLE | Server内通信链路 |
  | --- | --- | --- |
  | 1 | 不配置 | PCIe |
  | 1 | 0 | PCIe |
  | 0 | 1 | RoCE |
  | 不配置 | 1 | RoCE |
  | 0 | 0 | PCIe |
  | 不配置 | 不配置 | PCIe |

    > [!NOTE]说明
    > - 不支持HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE同时配置为1。
    > - 不支持HCCL_INTRA_PCIE_ENABLE配置为0，HCCL_INTRA_ROCE_ENABLE不配置。
    > - 不支持HCCL_INTRA_PCIE_ENABLE不配置，HCCL_INTRA_ROCE_ENABLE配置为0。

- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，该环境变量仅在使用LLM-DataDist作为集群管理组件的场景下生效，用于配置超节点内是否使用RoCE链路进行通信，默认值0，配置说明如下：
  - 0：超节点内采用默认的HCCS链路或PCIe链路进行通信（包括LLM-DataDist通信与HCCL通信）。
  - 1：针对Atlas 800T A3 超节点、Atlas 800I A3 超节点与Atlas 900 A3 SuperPoD 超节点，超节点内LLM-DataDist通信采用RoCE链路，HCCL通信不受影响；针对A200T A3 Box8 超节点，LLM-DataDist与HCCL通信都采用RoCE链路。

## 配置示例

```bash
export HCCL_INTRA_ROCE_ENABLE=1
```

## 使用约束

[Atlas 200T A2 Box16 异构子框](https://support.huawei.com/enterprise/zh/doc/EDOC1100318274/287e0458)存在左右两个模组，分别为0\~7卡和8\~15卡，针对此产品：

**单机场景下，**当Server内采用PCIe链路通信时，若需要同时使用两个模组的卡，两个模组需使用相同的卡数且在同一平面，即0卡和8卡、1卡和9卡（以此类推）需要同时使用；当Server内采用RoCE链路通信时，无此限制。

## 支持的型号

Atlas A3 训练系列产品/Atlas A3 推理系列产品：仅在使用LLM-DataDist作为集群管理组件的场景下生效。

Atlas A2 训练系列产品/Atlas A2 推理系列产品：仅支持此处理器型号下的[Atlas 200T A2 Box16 异构子框](https://support.huawei.com/enterprise/zh/doc/EDOC1100318274/287e0458)。

<cann-filter npu-type="910">Atlas 训练系列产品：仅支持此处理器型号下的[Atlas 300T Pro 训练卡](https://support.huawei.com/enterprise/zh/ascend-computing/atlas-300t-pro-pid-256118195)。</cann-filter>
