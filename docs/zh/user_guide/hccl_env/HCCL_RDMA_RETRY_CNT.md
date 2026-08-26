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
