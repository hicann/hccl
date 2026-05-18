# HCCL_INTER_HCCS_DISABLE

## 功能描述

此环境变量用于配置超节点模式组网中超节点内的通信链路类型，支持如下取值：

- TRUE：代表超节点内的AI节点间使用RoCE进行RDMA通信。
- FALSE：代表超节点内的AI节点间使用HCCS通信链路进行SDMA通信。

默认值为“FALSE”。

## 配置示例

```bash
export HCCL_INTER_HCCS_DISABLE=FALSE
```

## 使用约束

无

## 支持的型号

Atlas A3 训练系列产品/Atlas A3 推理系列产品
