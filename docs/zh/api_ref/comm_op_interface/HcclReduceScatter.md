# HcclReduceScatter

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
- Atlas 训练系列产品：支持</cann-filter>

<cann-filter npu-type="910b,310P">

> [!NOTE]说明
<cann-filter npu-type="910b">
> - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。</cann-filter>
<cann-filter npu-type="310p">
> - 针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。</cann-filter>
</cann-filter>

## 功能说明

集合通信算子ReduceScatter的操作接口，将通信域内所有rank的输入数据均分成rank size份，然后分别取每个rank的rank size之一份数据进行归约操作（如sum、prod、max、min）。最后，将结果按照编号分散到各个rank的输出buffer。

![reducescatter](figures/reducescatter.png)

## 函数原型

```c
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendBuf | 输入 | 源数据buffer地址。 |
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出至此buffer中。 |
| recvCount | 输入 | 参与ReduceScatter操作的recvBuf的数据size，sendBuf的数据size则等于recvCount * rank size。 |
| dataType | 输入 | ReduceScatter操作的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0-beta.3/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、int16、int32、int64、uint64、float16、float32、float64、bfp16。针对int64、uint64、float64，当前仅支持节点内通信。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。需要注意，针对int64数据类型，性能会有一定的劣化。<cann-filter npu-type="910"><br>针对Atlas 训练系列产品，支持数据类型：int8、int32、int64、float16、float32。</cann-filter><cann-filter npu-type="310p"><br>针对Atlas 300I Duo 推理卡，支持数据类型：int8、int16、int32、float16、float32。</cann-filter> |
| op | 输入 | Reduce的操作类型，目前支持操作类型为sum、prod、max、min。<br> 说明： 针对Ascend 950PR/Ascend 950DT，支持的操作类型为sum、max、min。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，当前版本“prod”操作不支持int16、bfp16数据类型。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，当前版本“prod”操作不支持int16、bfp16数据类型。<cann-filter npu-type="310p"><br>针对Atlas 300I Duo 推理卡，当前版本“prod”、“max”、“min”操作不支持int16数据类型。</cann-filter> |
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0-beta.3/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- 所有rank的recvCount、dataType、op均应相同。
<cann-filter npu-type="310p">- 针对Atlas 300I Duo 推理卡，仅支持单Server场景，单Server中最大支持部署16张Atlas 300I Duo 推理卡（即32个NPU）。</cann-filter>
- 算子算子的输入输出地址（sendBuf与recvBuf）根据不同的数据类型，应满足如下对齐要求：
  - int8按照1 Byte地址对齐。
  - int16、float16、bfp16按照2 Byte地址对齐。
  - int32、float32按照4 Byte地址对齐。
  - int64、uint64、float64按照8 Byte地址对齐。

## 调用示例

```c
uint32_t rankSize = 8;
uint64_t recvCount = 1;  // 每个节点接收的数据数量
uint64_t sendSize = rankSize * recvCount * sizeof(float);
uint64_t recvSize = recvCount * sizeof(float);

// 申请集合通信操作的 Device 内存
void *sendBuf = nullptr, *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// 初始化通信域和流
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// 执行 ReduceScatter，将所有 rank 的 sendBuf 相加后，再把结果按照 rank_id 顺序均匀分散到各个 rank 的 recvBuf
HcclReduceScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放 Device 侧内存
aclrtFree(recvBuf);          // 释放 Device 侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```
