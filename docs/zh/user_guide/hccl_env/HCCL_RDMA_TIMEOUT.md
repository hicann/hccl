# HCCL_RDMA_TIMEOUT

## 功能描述

用于配置RDMA网卡重传超时时间的系数timeout。

RDMA网卡重传超时时间最小值的计算公式为：4.096μs \* 2^timeout，其中timeout为该环境变量配置值，且实际重传超时时间与用户网络状况有关。

- 针对Ascend 950PR/Ascend 950DT，该环境变量配置为整数，取值范围为\[5,24\]，默认值为20。
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，该环境变量配置为整数，取值范围为\[5,20\]，默认值为20。
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，该环境变量配置为整数，取值范围为\[5,20\]，默认值为20。
<cann-filter npu-type="910">
- 针对Atlas 训练系列产品，该环境变量配置为整数，取值范围为\[5,24\]，默认值为20。</cann-filter>
<cann-filter npu-type="310p">
- 针对Atlas 推理系列产品，该环境变量配置为整数，取值范围是\[5,24\]，默认值为20。</cann-filter>

## 配置示例

```bash
# RDMA网卡重传超时时间的系数配置为6，则网卡启用RDMA功能时，重传超时时间最小值为：4.096μs * 2^6
export HCCL_RDMA_TIMEOUT=6
```

## 使用约束

无

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
