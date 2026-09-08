# HcclScatter

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
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：支持
<!-- end id5 -->

## 功能说明

集合通信算子Scatter操作接口，将root节点的数据均分并散布至其他rank。

## 函数原型

```c
HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendBuf | 输入 | 源数据buffer地址。 |
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出至此buffer中。 |
| recvCount | 输入 | 参与scatter操作的recvBuf的数据个数，比如只有一个int32数据参与，则recvcount=1。 |
| dataType | 输入 | Scatter操作的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>不同的型号支持的数据类型不同，详细请参见[数据类型说明](#数据类型说明)。|
| root | 输入 | 作为scatter root的rank id。 |
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

### 数据类型说明

<!-- npu="950" id10 -->
- 针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float8-e5m2、float8-e4m3、float8-e8m0、hifloat8、float16、float32、float64、bfp16。
<!-- end id10 -->
<!-- npu="A3" id11 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。
<!-- end id11 -->
<!-- npu="910b" id12 -->
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。
<!-- end id12 -->
<!-- npu="910" id13 -->
- 针对Atlas 训练系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。
<!-- end id13 -->

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| 返回值 | 说明 |
| --- | --- |
| HCCL_SUCCESS | 接口调用成功。 |
| HCCL_E_PTR | 传入的指针参数为空，如comm、recvBuf等为nullptr（root节点的sendBuf也不能为nullptr）。 |
| HCCL_E_PARA | 传入的参数无效，如count超过上限、root越界等。 |
| HCCL_E_NOT_SUPPORT | 操作不被支持，如dataType非法或当前型号不支持、混合组网不支持Scatter等。 |
| HCCL_E_INTERNAL | 内部错误。 |

## 约束说明

- 所有rank的recvCount、dataType、root均应相同。
- 全局只能有1个root节点。
- 非root节点的sendBuf可以为空。root节点的sendBuf不能为空。
- 多个通信域下的所有通信算子在每个Device上需要保证串行下发，不允许乱序、多线程并发下发，也不支持线程重入。
- 在同一Device上，同一通信域内的所有通信算子的下发线程需要使用相同的Context。

## 调用示例

```c
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t sendCount = 8;
uint64_t recvCount = 1;
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = recvCount * sizeof(float);

// 申请Device内存用于接收Scatter结果
ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));
// 在root节点，申请Device内存用于存放发送数据
if (device == rootRank) {
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
}

// 初始化通信域
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, device, &hcclComm);

// 创建任务流
aclrtStream stream;
aclrtCreateStream(&stream);

// 执行Scatter，将通信域内root节点的数据均分并散布至其他rank
HcclScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, rootRank, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放Device侧内存
aclrtFree(recvBuf);          // 释放Device侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```
