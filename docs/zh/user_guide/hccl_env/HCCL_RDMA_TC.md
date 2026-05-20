# HCCL_RDMA_TC

## 功能描述

用于配置RDMA网卡的traffic class。

该环境变量的取值范围为\[0,255\]，且需要配置为4的整数倍，默认值为132。

在RoCE V2协议中，该值对应IP报文头中ToS（Type of Service）域。共8个bit，其中，bit\[0,1\]固定为0，bit\[2,7\]为DSCP，因此，该值除以4即为DSCP的值。

![](figures/tos.png)

## 配置示例

```bash
# 该环境变量配置为25*4 = 100，则DSCP为25
export HCCL_RDMA_TC=100
```

## 使用约束

若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclRdmaTrafficClass”参数配置了RDMA网卡的traffic class，则以通信域粒度的配置优先。

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
