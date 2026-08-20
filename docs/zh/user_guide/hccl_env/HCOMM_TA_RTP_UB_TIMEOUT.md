# HCOMM_TA_RTP_UB_TIMEOUT

## 功能描述

用于配置UB_RTP协议下的Jetty超时时间的系数timeout。

UB_RTP协议下的Jetty超时时间分为4档，档位的计算公式为：timeout / 8，其中timeout为该环境变量配置值，0档：512ms；1档：4s；2档：8s；3档：32s。软件内部会有拦截校验机制，在创建Jetty前，先查出TP的超时配置，如果环境变量配置的时间小于等于TP总超时时间，将Jetty超时时间自动升档为大于TP总超时时间的最小档位；若环境变量配置的时间大于TP总超时时间，则直接使用环境变量配置的档位。时间配置建议按照0/8/16/24选择配置。

针对Ascend 950PR/Ascend 950DT，该环境变量配置为整数，取值范围为\[0,31\]，默认值为16。

## 配置示例

```bash
# UB_RTP协议超时时间的系数配置为16，则超时时间档位为：16 / 8 = 2，对应8s
export HCOMM_TA_RTP_UB_TIMEOUT=16
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
