# HCCL\_EXEC\_TIMEOUT

## 功能描述

不同设备进程在分布式训练或推理过程中存在卡间执行任务不一致的场景（如仅特定进程会保存checkpoint数据），通过该环境变量可控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步。

- **针对Ascend 950PR/Ascend 950DT**：单位为s，取值范围为：\[0, 2147483647\]，默认值为1836，支持整数秒配置。当配置为0时代表永不超时。
- **针对Atlas A3 训练系列产品/Atlas A3 推理系列产品：**
  - “AI_CPU”与“AICPU_CacheDisable”模式下（参见[HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)）：单位为s，取值范围为：\[0, 2147483647\]，默认值为1836，支持十毫秒级精度配置（例如，需要50毫秒超时，则配置为0.05）。当配置为0时代表永不超时。
  - “AIV”模式下：单位为s，取值范围为\[0, 1091\]，默认值为1091，支持十毫秒级精度配置（例如，需要50毫秒超时，则配置为0.05）。若设置为0或超出最大值1091，将按照1091处理。

    “AIV”模式下实际生效的超时时间为interval\*N\*10<sup>-3</sup>毫秒，其中interval为硬件支持的算子超时最短时间间隔（可通过aclrtGetOpTimeoutInterval接口获取），N的取值为\[1, 254\]范围内整数，如果配置的超时时间不等于interval\*N\*10<sup>-3</sup>毫秒，则向上对齐至interval\*N\*10<sup>-3</sup>毫秒。

- **针对Atlas A2 训练系列产品/Atlas A2 推理系列产品：**
  - “HOST”与“HOST\_TS”模式下（参见[HCCL_OP_EXPANSION_MODE](HCCL_OP_EXPANSION_MODE.md)）：单位为s，取值范围为：\[0, 2147483647\]，默认值为1836，支持整数秒配置。当配置为0时代表永不超时。
  - “AIV”模式下：单位为s，取值范围为\[0, 1091\]，默认值为1091，支持十毫秒级精度配置（例如，需要50毫秒超时，则配置为0.05）。若设置为0或超出最大值1091，将按照1091处理。

    “AIV”模式下实际生效的超时时间为interval\*N\*10<sup>-3</sup>毫秒，其中interval为硬件支持的算子超时最短时间间隔（可通过aclrtGetOpTimeoutInterval接口获取），N的取值为\[1, 254\]范围内整数，如果配置的超时时间不等于interval\*N\*10<sup>-3</sup>毫秒，则向上对齐到interval\*N\*10<sup>-3</sup>毫秒。

<cann-filter npu-type="910">- **针对Atlas 训练系列产品：**单位为s，取值范围为：\(0, 17340\]，默认值为1836，支持整数秒配置。

    需要注意：针对Atlas 训练系列产品，系统实际设置的超时时间 = 环境变量的取值先整除“68”，然后再乘以“68”，单位s。如果环境变量的取值小于68，则默认按照68s进行处理。

    例如，假设HCCL_EXEC_TIMEOUT=600，则系统实际设置的超时时间为：600整除68乘以68 = 8\*68 = 544s。</cann-filter>

<cann-filter npu-type="310p">- **针对Atlas 推理系列产品：**单位为s，取值范围为：\(0, 17340\]，默认值为1836，支持整数秒配置。

    需要注意：针对Atlas 推理系列产品，系统实际设置的超时时间 = 环境变量的取值先整除“68”，然后再乘以“68”，单位s。如果环境变量的取值小于68，则默认按照68s进行处理。

    例如，假设HCCL_EXEC_TIMEOUT=600，则系统实际设置的超时时间为：600整除68乘以68 = 8\*68 = 544s。</cann-filter>

> [!NOTE]说明
> 一般情况下，用户保持默认值即可。当默认值无法满足设备间执行通信同步的需求时，可通过此环境变量适当增大设备间的同步等待时间。

## 配置示例

```bash
export HCCL_EXEC_TIMEOUT=1800
```

## 使用约束

若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclExecTimeOut”参数配置了设备间执行时的同步等待时间，则以通信域粒度的配置为准。

## 支持的型号

Ascend 950PR/Ascend 950DT

Atlas A3 训练系列产品/Atlas A3 推理系列产品

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

<cann-filter npu-type="910">Atlas 训练系列产品</cann-filter>

<cann-filter npu-type="310p">Atlas 推理系列产品（针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。）</cann-filter>
