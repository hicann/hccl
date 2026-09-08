# HcclAllReduce

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
- Atlas 训练系列产品：支持
<!-- end id5 -->

## 功能说明

集合通信算子AllReduce的操作接口，将通信域内所有节点的输入数据进行相加（或其他归约操作）后，再把结果发送到所有节点的输出buffer，其中归约操作类型由op参数指定。

![allreduce](figures/allreduce.png)

## 函数原型

```c
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendBuf | 输入 | 源数据buffer地址。 |
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出至此buffer中。 |
| count | 输入 | 参与allreduce操作的数据个数，比如只有一个int32数据参与，则count=1。 |
| dataType | 输入 | allreduce操作的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>不同的型号支持的数据类型不同，详细请参见[dataType说明](#datatype说明)。|
| op | 输入 | reduce的操作类型。<br>不同的型号支持的操作类型不同，详细请参见[op说明](#op说明)。|
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

### dataType说明

<!-- npu="950" id10 -->
- 针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、int16、int32、int64、uint64、float16、float32、float64、bfp16。
<!-- end id10 -->
<!-- npu="A3" id11 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。
<!-- end id11 -->
<!-- npu="910b" id12 -->
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。需要注意，针对int64数据类型，性能会有一定的劣化。
<!-- end id12 -->
<!-- npu="910" id13 -->
- 针对Atlas 训练系列产品，支持数据类型：int8、int32、int64、float16、float32。
<!-- end id13 -->
<!-- npu="310p" id6 -->
- 针对Atlas 300I Duo 推理卡，支持数据类型：int8、int16、int32、float16、float32。
<!-- end id6 -->

### op说明

<!-- npu="950" id14 -->
- 针对Ascend 950PR/Ascend 950DT，支持的操作类型为sum、prod、max、min，其中prod操作不支持int16、bfp16数据类型。
<!-- end id14 -->
<!-- npu="A3" id15 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持的操作类型为sum、prod、max、min，其中prod操作不支持int16、bfp16数据类型。
<!-- end id15 -->
<!-- npu="910b" id16 -->
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持的操作类型为sum、prod、max、min，其中prod操作不支持int16、bfp16数据类型。
<!-- end id16 -->
<!-- npu="910" id17 -->
- 针对Atlas 训练系列产品，支持的操作类型为sum、prod、max、min。
<!-- end id17 -->
<!-- npu="310p" id7 -->
- 针对Atlas 300I Duo 推理卡，支持的操作类型为sum、prod、max、min，其中prod、max、min操作不支持int16数据类型。
<!-- end id7 -->

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| 返回值 | 说明 |
| --- | --- |
| HCCL_SUCCESS | 接口调用成功。 |
| HCCL_E_PTR | 传入的指针参数为空，如comm、sendBuf、recvBuf、stream等为nullptr。 |
| HCCL_E_PARA | 传入的参数无效，如count超过上限等。 |
| HCCL_E_NOT_SUPPORT | 操作不被支持，如dataType非法或当前型号不支持、prod操作不支持int16/bfp16数据类型等。 |
| HCCL_E_INTERNAL | 内部错误。 |

## 约束说明

- 所有rank的count、dataType、op均应相同。
- 每个rank只能有一个输入。
- 算子的输入输出地址（sendBuf与recvBuf）根据不同的数据类型，应满足如下对齐要求：
  - int8按照1 Byte地址对齐。
  - int16、float16、bfp16按照2 Byte地址对齐。
  - int32、float32按照4 Byte地址对齐。
  - int64、uint64、float64按照8 Byte地址对齐。
- 多个通信域下的所有通信算子在每个Device上需要保证串行下发，不允许乱序、多线程并发下发，也不支持线程重入。
- 在同一Device上，同一通信域内的所有通信算子的下发线程需要使用相同的Context。

## 调用示例

```c
// 申请集合通信操作的Device 内存
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t count = 8;
size_t mallocSize = count * sizeof(float);
aclrtMalloc((void **)&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc((void **)&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);

// 初始化通信域
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// 创建任务流
aclrtStream stream;
aclrtCreateStream(&stream);

// 执行AllReduce，将通信域内所有节点的输入数据进行相加后，再把结果发送到所有节点的输出buffer
HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放Device侧内存
aclrtFree(recvBuf);          // 释放Device侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```
