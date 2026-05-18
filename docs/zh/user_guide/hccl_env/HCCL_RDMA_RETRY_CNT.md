# HCCL_RDMA_RETRY_CNT

## 功能描述

用于配置RDMA网卡的重传次数，需要配置为整数，取值范围为\[1,7\]，默认值为7。

## 配置示例

```bash
#重传次数配置为5
export HCCL_RDMA_RETRY_CNT=5
```

## 使用约束

无

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
