# superDeviceId重复（EI0014）

## 问题现象

在CANN日志中存在关键字"superDeviceId\[\*\*\*\] in superPod\[\*\*\*\]is already exist"，如下所示：

```text
[ERROR] HCCL(169030,alltoall_test):2025-10-23-16:28:59.392.635 [topoinfo_exchange_agent.cc:695] [169030][InitGroupStage][RanktableCheck]devices have same superDeviceId[0x3000000] in superPod[super_pod_id_0]. Current device info: serverId[127.10.0.1], rankId[0], group[hccl_world_group]. Another device info: rankId[1].
```

## 可能原因

superDeviceId是Atlas A3 训练系列产品/Atlas A3 推理系列产品内Device在超节点系统中的物理ID，是超节点系统中Device的唯一标识。HCCL在一致性校验时发现一个超节点内有相同的superDeviceId，因此校验失败。superDeviceId可通过npu-smi命令查询：

```bash
npu-smi info -t spod-info -i id -c chip_id
```

- id：设备id，通过npu-smi info -l命令查出的NPU ID即为设备id。
- chip_id：芯片id，通过npu-smi info -m命令查出的Chip ID即为芯片id。

回显中的“SDID”即为superDeviceID。

出现此问题的可能原因是：

- 硬件配置异常。
- 通过[HCCL_LOGIC_SUPERPOD_ID](../hccl_env/HCCL_LOGIC_SUPERPOD_ID.md)环境变量将不同的物理超节点配置在了同一个逻辑超节点内，导致superDeviceId重复。

### 解决方法

修改硬件配置或正确配置[HCCL_LOGIC_SUPERPOD_ID](../hccl_env/HCCL_LOGIC_SUPERPOD_ID.md)环境变量，避免同一个超节点内出现superDeviceId相同的设备。
