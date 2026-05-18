# HcclReduceScatterV

## 产品支持情况

<cann-filter npu-type="950">

- Ascend 950PR/Ascend 950DT：支持</cann-filter>
<cann-filter npu-type="A3">
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持</cann-filter>
<cann-filter npu-type="910b">
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持</cann-filter>
<cann-filter npu-type="310p">
- Atlas 推理系列产品：支持</cann-filter>
<cann-filter npu-type="910">
- Atlas 训练系列产品：不支持</cann-filter>

<cann-filter npu-type="910b,310P">

> [!NOTE]说明
<cann-filter npu-type="910b">
> - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。</cann-filter>
<cann-filter npu-type="310p">
> - 针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。</cann-filter>
</cann-filter>

## 功能说明

集合通信算子ReduceScatterV的操作接口，与ReduceScatter操作类似，不同点是支持为通信域内不同的节点配置不同大小的数据量（同一rank不同编号的数据大小可设置，但不同rank间相同编号的数据大小需保持一致），取每个rank对应编号的数据进行归约操作后（支持sum、prod、max、min）后，再把结果按照编号分散到各个rank的输出buffer。

![reducescatterv](figures/reducescatterv.png)

## 函数原型

```c
HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendBuf | 输入 | 源数据buffer地址。 |
| sendCounts | 输入 | 参与ReduceScatterV操作的每个rank在sendBuf中的数据size，为uint64类型的数组。<br>该数组的第i个元素表示需要向rank i发送的数据量。 |
| sendDispls | 输入 | 参与ReduceScatterV操作的每个rank的数据在sendBuf中的偏移量（单位为dataType），为uint64类型的数组。<br>该数组的第i个元素表示向rank i发送的数据在sendBuf中的偏移量。 |
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出至此buffer中。 |
| recvCount | 输入 | 参与ReduceScatterV操作的rank对应recvBuf的数据size。<br>假设当前rank的编号为i，则recvCount的值需要与sendCounts数组中下标为i的元素值相同。 |
| dataType | 输入 | ReduceScatterV操作的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、int16、int32、float16、float32、bfp16。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、int16、int32、float16、float32、bfp16。<cann-filter npu-type="310p"><br>针对Atlas 300I Duo 推理卡，支持数据类型：int16、float16、float32。</cann-filter> |
| op | 输入 | Reduce的操作类型。<br>针对Ascend 950PR/Ascend 950DT，支持的操作类型为sum、max、min。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的操作类型为sum、max、min。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的操作类型为sum、max、min。<cann-filter npu-type="310p"><br>针对Atlas 300I Duo 推理卡，仅支持操作类型sum。</cann-filter> |
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 所有rank的sendCounts、sendDispls、dataType、op均应相同。
- 针对Ascend 950PR/Ascend 950DT，仅支持单Server场景，仅支持通信算子展开模式为CCU（Collective Communication Unit，集合通信加速单元）的场景。
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，仅支持单Server场景。
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持多机对称分布场景，不支持非对称分布（即卡数非对称）的场景。
- 针对Atlas 300I Duo 推理卡，仅支持单Server场景，单Server中最大支持部署2张Atlas 300I Duo 推理卡（即4个NPU）。
- 算子算子的输入输出地址（sendBuf与recvBuf）根据不同的数据类型，应满足如下对齐要求：
  - int8按照1 Byte地址对齐。
  - int16、float16、bfp16按照2 Byte地址对齐。
  - int32、float32按照4 Byte地址对齐。
  - int64按照8 Byte地址对齐。
