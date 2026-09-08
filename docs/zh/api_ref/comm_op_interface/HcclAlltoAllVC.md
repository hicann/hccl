# HcclAlltoAllVC

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
- Atlas 训练系列产品：不支持
<!-- end id5 -->

## 功能说明

集合通信算子AlltoAllVC操作接口，向通信域内所有rank发送数据（数据量可以定制），并从所有rank接收数据。相比于AlltoAllV，AlltoAllVC通过输入参数sendCountMatrix传入所有rank的收发参数。

![alltoallvc](figures/alltoallvc.png)

## 函数原型

```c
HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendBuf | 输入 | 源数据buffer地址。 |
| sendCountMatrix | 输入 | 表示发送数据量的二维uint64数组，数组形状为`[rankSize][rankSize]`，`sendCountMatrix[i][j] = n`表示rank i发给rank j的数据量为n。<br>例如，若“sendType”为float32，则`sendCountMatrix[i][j] = n`表示rank i发送给rank j n个float32数据。 |
| sendType | 输入 | 发送数据的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>不同的型号支持的数据类型不同，详细请参见[数据类型说明](#数据类型说明)。|
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出至此buffer中。<br>recvBuf与sendBuf配置的地址不能相同，且两者的内存范围不能重叠。 |
| recvType | 输入 | 接收数据的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>不同的型号支持的数据类型不同，详细请参见[数据类型说明](#数据类型说明)。 |
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

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| 返回值 | 说明 |
| --- | --- |
| HCCL_SUCCESS | 接口调用成功。 |
| HCCL_E_PTR | 传入的指针参数为空，如comm、sendBuf、recvBuf、stream、sendCountMatrix等为nullptr。 |
| HCCL_E_PARA | 传入的参数无效，如sendBuf与recvBuf地址相同等。 |
| HCCL_E_NOT_SUPPORT | 操作不被支持，如dataType非法或当前型号不支持。 |
| HCCL_E_INTERNAL | 内部错误。 |

## 约束说明

AlltoAllVC操作的性能与NPU之间共享数据的缓存区大小有关，当通信数据量超过缓存区大小时性能将出现明显下降。若业务中AlltoAllVC通信数据量较大，建议通过配置环境变量[HCCL_BUFFSIZE](../../user_guide/hccl_env/HCCL_BUFFSIZE.md)适当增大缓存区大小以提升通信性能。
- 多个通信域下的所有通信算子在每个Device上需要保证串行下发，不允许乱序、多线程并发下发，也不支持线程重入。
- 在同一Device上，同一通信域内的所有通信算子的下发线程需要使用相同的Context。

## 调用示例

```c
// 申请集合通信操作的Device内存
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

// 设置收发数据量，收发数据量相同
std::vector<uint64_t> sendCountMatrix(rankSize * rankSize);
for (uint32_t i = 0; i < rankSize; ++i) {
    for (uint32_t j = 0; j < rankSize; ++j) {
        sendCountMatrix[i * rankSize + j] = count / rankSize;
    }
}

// 执行AlltoAllVC，向通信域内所有rank发送相同数据量的数据，并从所有rank接收相同数据量的数据，可定制数据量
HcclAlltoAllVC(sendBuf, sendCountMatrix.data(), HCCL_DATA_TYPE_FP32, recvBuf, HCCL_DATA_TYPE_FP32, hcclComm, stream);

// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放Device侧内存
aclrtFree(recvBuf);          // 释放Device侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```
