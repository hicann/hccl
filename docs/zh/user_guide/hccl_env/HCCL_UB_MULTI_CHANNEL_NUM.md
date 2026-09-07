# HCCL_UB_MULTI_CHANNEL_NUM

## 功能描述

CLOS拓扑+UB协议（CTP）通信场景下，开发者可通过本环境变量配置同一eid对之间建立的channel数量。

该环境变量需要配置为整数，取值范围：\[1,16\]，默认值：1。

每个channel底层对应一个独立jetty，多个channel并行搬运数据，以提升通信带宽（与ROCE多QP机制类似）。默认值为1，即不使能多channel。

## 配置示例

```bash
export HCCL_UB_MULTI_CHANNEL_NUM=4
```

## 使用约束

- 该环境变量仅在纯CLOS拓扑（Level0Shape::CLOS）+UB协议（CTP）、且算法运行于AICPU引擎时生效；mesh拓扑、ubmem协议、PCIE等链路场景下不生效（CCU引擎下多个channel复用同一jetty，该环境变量不生效）。
- 该环境变量仅支持集合通信算子（通过HcclConfigGetInfo读取通信域配置）；开发者自写用例直接创建channel的场景不支持。
- 取值超出\[1,16\]时，HCCL初始化或读取阶段会报错（解析阶段上报故障码EI0001，读取阶段返回HCCL_E_PARA）。
- 每个channel均会占用jetty等硬件资源，channel数量配置过大可能挤压其它通信域的资源，建议按需设置（如2或4）。
- 性能提示：仅在单个jetty为通信带宽瓶颈时，增加channel数量才有收益；若单个jetty已达到链路带宽上限，多channel无增益。建议结合性能验证结果配置该环境变量。

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
