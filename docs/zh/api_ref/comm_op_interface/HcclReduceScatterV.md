# HcclReduceScatterV

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
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出至此buffer中。<br>recvBuf与sendBuf配置的地址不能相同。 |
| recvCount | 输入 | 参与ReduceScatterV操作的rank对应recvBuf的数据size。<br>假设当前rank的编号为i，则recvCount的值需要与sendCounts数组中下标为i的元素值相同。 |
| dataType | 输入 | ReduceScatterV操作的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>不同的型号支持的数据类型不同，详细请参见[dataType说明](#datatype说明)。|
| op | 输入 | Reduce的操作类型。<br>不同的型号支持的操作类型不同，详细请参见[op说明](#op说明)。|
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

### dataType说明

<!-- npu="950" id10 -->
- 针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。
<!-- end id10 -->
<!-- npu="A3" id11 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。
<!-- end id11 -->
<!-- npu="910b" id12 -->
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、int16、int32、float16、float32、bfp16。
<!-- end id12 -->
<!-- npu="310p" id6 -->
- 针对Atlas 300I Duo 推理卡，支持数据类型：int16、float16、float32。
<!-- end id6 -->

### op说明

<!-- npu="950" id14 -->
- 针对Ascend 950PR/Ascend 950DT，支持的操作类型为sum、prod、max、min，其中prod操作不支持int16、bfp16数据类型。
<!-- end id14 -->
<!-- npu="A3" id15 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的操作类型为sum、max、min。
<!-- end id15 -->
<!-- npu="910b" id16 -->
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的操作类型为sum、max、min。
<!-- end id16 -->
<!-- npu="310p" id7 -->
- 针对Atlas 300I Duo 推理卡，仅支持操作类型sum。
<!-- end id7 -->

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| 返回值 | 说明 |
| --- | --- |
| HCCL_SUCCESS | 接口调用成功。 |
| HCCL_E_PTR | 传入的指针参数为空，如comm、sendCounts、sendDispls、stream等为nullptr（recvCount大于0时recvBuf也不能为nullptr）。 |
| HCCL_E_PARA | 传入的参数无效，如count超过上限等。 |
| HCCL_E_NOT_SUPPORT | 操作不被支持，如dataType非法或当前型号不支持、prod操作不支持int16/bfp16数据类型等。 |
| HCCL_E_INTERNAL | 内部错误。 |

## 约束说明

- 所有rank的sendCounts、sendDispls、dataType、op均应相同。
<!-- npu="A3" id18 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，仅支持单Server场景。
<!-- end id18 -->
<!-- npu="910b" id19 -->
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持多机对称分布场景，不支持非对称分布（即卡数非对称）的场景。
<!-- end id19 -->
<!-- npu="310p" id8 -->
- 针对Atlas 300I Duo 推理卡，仅支持单Server场景，单Server中最大支持部署2张Atlas 300I Duo 推理卡（即4个NPU）。
<!-- end id8 -->
- 算子的输入输出地址（sendBuf与recvBuf）根据不同的数据类型，应满足如下对齐要求：
  - int8按照1Byte地址对齐。
  - int16、float16、bfp16按照2Byte地址对齐。
  - int32、float32按照4Byte地址对齐。
  - int64按照8Byte地址对齐。
- 多个通信域下的所有通信算子在每个Device上需要保证串行下发，不允许乱序、多线程并发下发，也不支持线程重入。
- 在同一Device上，同一通信域内的所有通信算子的下发线程需要使用相同的Context。

## 调用示例

```c
// 申请集合通信操作的Device内存
uint32_t rankSize = 8;
uint64_t recvCount = 1;  // 每个rank接收的数据个数
size_t recvSize = recvCount * sizeof(float);
size_t totalSendCount = rankSize * recvCount;
size_t sendSize = totalSendCount * sizeof(float);

void *sendBuf = nullptr;
void *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// 设置sendCounts和sendDispls，每个rank发送相同数量的数据
std::vector<uint64_t> sendCounts(rankSize, recvCount);
std::vector<uint64_t> sendDispls(rankSize);
for (uint32_t i = 0; i < rankSize; ++i) {
    sendDispls[i] = i * recvCount;
}

// 初始化通信域
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// 创建任务流
aclrtStream stream;
aclrtCreateStream(&stream);

// 执行ReduceScatterV，将所有rank的sendBuf相加后，再把结果按照编号分散到各个rank的recvBuf
HcclReduceScatterV(sendBuf, sendCounts.data(), sendDispls.data(), recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放Device侧内存
aclrtFree(recvBuf);          // 释放Device侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```
