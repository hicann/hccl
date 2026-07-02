# HcclAlltoAllV

## 产品支持情况

<cann-filter npu-type="950">

- Ascend 950PR/Ascend 950DT：支持</cann-filter>
<cann-filter npu-type="A3">
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持</cann-filter>
<cann-filter npu-type="910b">
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持</cann-filter>
<cann-filter npu-type="310p">
- Atlas 推理系列产品：支持</cann-filter>
<cann-filter npu-type="910">- Atlas 训练系列产品：支持</cann-filter>

<cann-filter npu-type="910b,310P">

> [!NOTE]说明
<cann-filter npu-type="910b">
> - 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。</cann-filter>
<cann-filter npu-type="310p">
> - 针对Atlas 推理系列产品，仅支持Atlas 300I Duo 推理卡。</cann-filter>
</cann-filter>

## 功能说明

集合通信算子AlltoAllV操作接口，向通信域内所有rank发送数据（数据量可以定制），并从所有rank接收数据。

![alltoallv](figures/alltoallv.png)

## 函数原型

```c
HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendBuf | 输入 | 源数据buffer地址。 |
| sendCounts | 输入 | 表示发送数据量的uint64数组，“sendCounts\[i] = n”表示本rank发给rank i的数据量为n。<br>例如，若“sendType”为float32，“sendCounts\[i] = n”表示本rank发给rank i n个float32数据。 |
| sdispls | 输入 | 表示发送偏移量的uint64数组，“sdispls\[i] = n”表示本rank发给rank i的数据在sendBuf的起始位置相对sendBuf的偏移量，以sendType为基本单位。 |
| sendType | 输入 | 发送数据的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0-beta.2/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float8-e5m2、float8-e4m3、float8-e8m0、hifloat8、float16、float32、float64、bfp16。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<cann-filter npu-type="910"><br>针对Atlas 训练系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。</cann-filter><cann-filter npu-type="310p"><br>针对Atlas 300I Duo 推理卡，支持的数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。</cann-filter> |
| recvBuf | 输出 | 目的数据buffer地址，集合通信结果输出至此buffer中。<br>recvBuf与sendBuf配置的地址不能相同。 |
| recvCounts | 输入 | 表示接收数据量的uint64数组，“recvCounts\[i] = n”表示本rank从rank i收到的数据量为n。<br>例如，若“recvType”为float32，“recvCounts\[i] = n”表示本rank从rank i收到n个float32数据。 |
| rdispls | 输入 | 表示接收偏移量的uint64数组，“rdispls\[i] = n”表示本rank从rank i收到的数据存放在recvBuf的起始位置相对recvBuf的偏移量，以recvType为基本单位。 |
| recvType | 输入 | 接收数据的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0-beta.2/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float8-e5m2、float8-e4m3、float8-e8m0、hifloat8、float16、float32、float64、bfp16。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<cann-filter npu-type="910"><br>针对Atlas 训练系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。</cann-filter><cann-filter npu-type="310p"><br>针对Atlas 300I Duo 推理卡，支持的数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。</cann-filter> |
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0-beta.2/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- AlltoAllV操作的性能与NPU之间共享数据的缓存区大小有关，当通信数据量超过缓存区大小时性能将出现明显下降。若业务中AlltoAllV通信数据量较大，建议通过配置环境变量[HCCL_BUFFSIZE](../../user_guide/hccl_env/HCCL_BUFFSIZE.md)适当增大缓存区大小以提升通信性能。
<cann-filter npu-type="910">

- 针对Atlas 训练系列产品，AlltoAllV的通信域需要满足如下约束：

  集群组网下，单Server 1p、2p通信域要在同一个cluster内（Server内0-3卡和4-7卡各为一个cluster），单Server 4p、8p和多Server通信域中rank要以cluster为基本单位，并且Server间cluster选取要一致。

- 针对Atlas 训练系列产品，如果是单Server场景，要求网卡的状态是“up”，否则此接口会执行失败。</cann-filter>
<cann-filter npu-type="310p">
- 针对Atlas 300I Duo 推理卡，仅支持单Server场景，单Server中最大支持部署2张Atlas 300I Duo 推理卡（即4个NPU）。</cann-filter>

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
std::vector<uint64_t> sendCounts(rankSize, 1);
std::vector<uint64_t> recvCounts(rankSize, 1);
std::vector<uint64_t> sdispls(rankSize);
std::vector<uint64_t> rdispls(rankSize);
for (size_t i = 0; i < rankSize; ++i) {
    sdispls[i] = i;
    rdispls[i] = i;
}
// 执行AlltoAllV，向通信域内所有rank发送相同数据量的数据，并从所有rank接收相同数据量的数据，可定制数据量
HcclAlltoAllV(sendBuf, sendCounts.data(), sdispls.data(), HCCL_DATA_TYPE_FP32,
              recvBuf, recvCounts.data(), rdispls.data(), HCCL_DATA_TYPE_FP32, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放Device侧内存
aclrtFree(recvBuf);          // 释放Device侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```
