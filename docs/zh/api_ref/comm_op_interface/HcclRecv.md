# HcclRecv

## 产品支持情况

<cann-filter npu-type="950">

- Ascend 950PR/Ascend 950DT：支持</cann-filter>
<cann-filter npu-type="A3">
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持</cann-filter>
<cann-filter npu-type="910b">
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持</cann-filter>
<cann-filter npu-type="310p">
- Atlas 推理系列产品：不支持</cann-filter>
<cann-filter npu-type="910">
- Atlas 训练系列产品：支持</cann-filter>

<cann-filter npu-type="910b">

> [!NOTE]说明
> 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。
</cann-filter>

## 功能说明

点对点通信Receive操作接口，从源节点接收数据到当前节点的指定位置上。

## 函数原型

```c
HcclResult HcclRecv(void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| recvBuf | 输入 | 数据接收buffer地址。 |
| count | 输入 | 接收数据的个数。 |
| dataType | 输入 | 接收数据的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0-beta.2/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float8-e5m2、float8-e4m3、float8-e8m0、hifloat8、float16、float32、float64、bfp16。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<cann-filter npu-type="910"><br>针对Atlas 训练系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。</cann-filter> |
| srcRank | 输入 | 通信域内数据发送端的rank编号。 |
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0-beta.2/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

HcclSend与HcclRecv接口采用同步调用方式，且必须配对使用。即一个进程调用HcclSend接口后，需要等到与之配对的HcclRecv接口接收数据后，才可以进行下一个接口调用，如下图所示。

![](figures/send_recv.png)

## 调用示例

```c
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t count = 8;
size_t mallocSize = count * sizeof(float);

// 初始化通信域
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// 创建任务流
aclrtStream stream;
aclrtCreateStream(&stream);

// 执行 Send/Recv 操作，0/2/4/6卡发送数据，1/3/5/7接收数据
// HcclSend 与 HcclRecv 接口采用同步调用方式，且必须配对使用
if (deviceId % 2 == 0) {
    // 申请 Device 内存用于存放输入数据
    aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    // 初始化输入数据
    aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE);
    // 执行 Send 操作
    HcclSend(sendBuf, count, HCCL_DATA_TYPE_FP32, deviceId + 1, hcclComm, stream);
} else {
    // 申请 Device 内存用于接收数据
    aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    // 执行 Recv 操作
    HcclRecv(recvBuf, count, HCCL_DATA_TYPE_FP32, deviceId - 1, hcclComm, stream);
}

// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放 Device 侧内存
aclrtFree(recvBuf);          // 释放 Device 侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```
