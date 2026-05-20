# HCCL\_CONNECT\_TIMEOUT

## 功能描述

分布式训练或推理场景下，用于限制不同设备之间socket建链过程的超时等待时间。不同设备进程在集合通信初始化之前由于其他因素会导致执行不同步。该环境变量控制设备间的建链超时等待时间，在该配置时间内各设备进程等待其他设备建链同步。

该环境变量需要配置为整数，取值范围\[120,7200\]，默认值为120，单位s。

**需要注意的是**：实际的建链超时等待时间是该环境变量的值加上20秒。例如，如果该环境变量设置为150秒，则实际的超时等待时间为170秒。额外的20秒用于通知各个节点通信域初始化失败的原因。

> [!NOTE]说明
> 此环境变量的值会影响链路故障场景的异常上报时间。

## 配置示例

```bash
export HCCL_CONNECT_TIMEOUT=200
```

## 使用约束

无

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
