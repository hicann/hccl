# HCCL_RDMA_SL

## 功能描述

用于配置RDMA网卡的service level，该值需要和网卡配置的PFC优先级保持一致，若配置不一致可能导致性能劣化。

该环境变量需要配置为整数，取值范围：\[0,7\]，默认值：4。

## 配置示例

```bash
# 优先级配置为3
export HCCL_RDMA_SL=3
```

## 使用约束

若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclRdmaServiceLevel”参数配置了RDMA网卡的service level，则以通信域粒度的配置优先。

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
