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
| dataType | 输入 | AllGatherV操作的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>不同的型号支持的数据类型不同，详细请参见[数据类型说明](#数据类型说明)。|
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

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| 返回值 | 说明 |
| --- | --- |
| HCCL_SUCCESS | 接口调用成功。 |
| HCCL_E_PTR | 传入的指针参数为空，如comm、recvCounts、recvDispls、stream等为nullptr（sendCount大于0时sendBuf、recvCounts非全0时recvBuf也不能为nullptr）。 |
| HCCL_E_PARA | 传入的参数无效，如count超过上限等。 |
| HCCL_E_NOT_SUPPORT | 操作不被支持，如dataType非法或当前型号不支持。 |
| HCCL_E_INTERNAL | 内部错误。 |

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
- 多个通信域下的所有通信算子在每个Device上需要保证串行下发，不允许乱序、多线程并发下发，也不支持线程重入。
- 在同一Device上，同一通信域内的所有通信算子的下发线程需要使用相同的Context。

## 调用示例

```c
// 申请集合通信操作的Device内存
uint32_t rankSize = 8;
uint64_t sendCount = 1;  // 每个rank发送的数据个数
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = rankSize * sendCount * sizeof(float);

void *sendBuf = nullptr;
void *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// 设置recvCounts和recvDispls，每个rank接收相同数量的数据
std::vector<uint64_t> recvCounts(rankSize, sendCount);
std::vector<uint64_t> recvDispls(rankSize);
for (uint32_t i = 0; i < rankSize; ++i) {
    recvDispls[i] = i * sendCount;
}

// 初始化通信域
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// 创建任务流
aclrtStream stream;
aclrtCreateStream(&stream);

// 执行AllGatherV，将通信域内所有rank的sendBuf按照rank id重新排序后拼接，再将结果发送到所有rank的recvBuf
HcclAllGatherV(sendBuf, sendCount, recvBuf, recvCounts.data(), recvDispls.data(), HCCL_DATA_TYPE_FP32, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放Device侧内存
aclrtFree(recvBuf);          // 释放Device侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```
