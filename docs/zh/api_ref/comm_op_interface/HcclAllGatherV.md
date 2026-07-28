# HcclAllGatherV

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

集合通信算子AllGatherV的操作接口，将通信域内所有节点的输入按照rank id重新排序，然后拼接起来，再将结果发送到所有节点的输出。

与AllGather算子不同的是，AllGatherV算子支持通信域内不同节点的输入配置不同大小的数据量。

![allgatherv](figures/allgatherv.png)

> [!NOTE]说明
> 针对AllGatherV操作，每个节点都接收按照rank id重新排序后的数据集合，即每个节点的AllGatherV输出都是一样的。

## 函数原型

```c
HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendBuf | 输入 | 源数据buffer地址。 |
| sendCount | 输入 | 参与AllGatherV操作的sendBuf的数据size。 |
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出至此buffer中。<br>recvBuf与sendBuf配置的地址不能相同。 |
| recvCounts | 输入 | 参与AllGatherV操作的每个rank在recvBuf中的数据size，为uint64类型的数组。<br>该数组的第i个元素表示需要从rank i接收的数据量，且该数据量需要与rank i的sendCount值相同。 |
| recvDispls | 输入 | 参与AllGatherV操作的每个rank的数据在recvBuf中的偏移量（单位为dataType），为uint64类型的数组。<br>该数组的第i个元素表示从rank i接收的数据应该放置在recvBuf中的起始偏移量。 |
| dataType | 输入 | AllGatherV操作的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/9.2.0-beta.1/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>不同的型号支持的数据类型不同，详细请参见[数据类型说明](#数据类型说明)。|
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

### 数据类型说明

<!-- npu="950" id11 -->
- 针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float8-e5m2、float8-e4m3、float8-e8m0、hifloat8、float16、float32、float64、bfp16。
<!-- end id11 -->
<!-- npu="A3" id12 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。
<!-- end id12 -->
<!-- npu="910b" id13 -->
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。
<!-- end id13 -->
<!-- npu="310p" id6 -->
- 针对Atlas 300I Duo 推理卡，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。
<!-- end id6 -->

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.2.0-beta.1/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 所有rank的recvCounts、recvDispls、dataType均应相同。
<!-- npu="A3" id15 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，仅支持单Server场景。
<!-- end id15 -->
<!-- npu="910b" id16 -->
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持多机对称分布场景，不支持非对称分布（即卡数非对称）的场景。
<!-- end id16 -->
<!-- npu="310p" id10 -->
- 针对Atlas 300I Duo 推理卡，仅支持单Server场景，单Server中最大支持部署2张Atlas 300I Duo 推理卡（即4个NPU）。
<!-- end id10 -->
