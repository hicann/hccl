# HCOMM_TA_CTP_UB_TIMEOUT

## 功能描述

用于配置UB_CTP协议下的Jetty超时时间的系数timeout。

UB_CTP协议下的Jetty超时时间分为4档，档位的计算公式为：timeout / 8，其中timeout为该环境变量配置值，0档：512ms；1档：4s；2档：8s；3档：32s。UB_CTP协议直接使用环境变量配置值，不与TP总超时时间比较。时间配置建议按照0/8/16/24选择配置。

针对Ascend 950PR/Ascend 950DT，该环境变量配置为整数，取值范围为\[0,31\]，默认值为8。

## 配置示例

```bash
# UB_CTP协议超时时间的系数配置为8，则超时时间档位为：8 / 8 = 1，对应4s
export HCOMM_TA_CTP_UB_TIMEOUT=8
```

## 使用约束

无

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->
