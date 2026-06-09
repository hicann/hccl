# HcclBatchSendRecv

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

异步批量点对点通信操作接口，调用一次接口可以完成本rank上的多个收发任务，本rank发送和接收之间是异步的，发送和接收任务之间不会相互阻塞。

## 函数原型

```c
HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| sendRecvInfo | 输入 | 本rank需要下发的收发任务列表的首地址。<br>HcclSendRecvItem类型，详细可参见[HcclSendRecvItem](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclSendRecvItem.md)。<br>针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float8-e5m2、float8-e4m3、float8-e8m0、hifloat8、float16、float32、float64、bfp16。<br>针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。<br>针对Atlas 训练系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。</cann-filter> |
| itemNum | 输入 | 本rank需要接收和发送的任务个数。 |
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)：接口成功返回HCCL_SUCCESS，其他失败。

## 约束说明

- “异步”是指同一张卡上的接收和发送任务是异步的，不会相互阻塞。但是在卡间，收发任务依旧是同步的，因此，卡间的收发任务也同HcclSend、HcclRecv一样，必须是一一对应的。
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，在大规模集群下（ranksize\>500）使用此接口时，并发执行数不能超过3个。
- 针对[Atlas 200T A2 Box16 异构子框](https://support.huawei.com/enterprise/zh/doc/EDOC1100318274/287e0458)，若Server内卡间出现建链失败的情况（错误码：EI0010），需要将环境变量HCCL\_INTRA\_ROCE\_ENABLE配置为1，HCCL\_INTRA\_PCIE\_ENABLE配置为0，让Server内采用RoCE环路进行多卡间的通信（请确保Server上存在RoCE网卡，且具有send/recv收发关系的设备之间RDMA链路互通），环境变量配置示例如下：

    ```bash
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_INTRA_PCIE_ENABLE=0
    ```

## 调用示例

```c
// 申请集合通信操作的 Device 内存
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

// 执行 Send/Recv，将数据发送至下一节点，同时接收上一节点的数据
// HcclBatchSendRecv 可以同时下发本 rank 上的多个收发任务
uint32_t next = (deviceId + 1) % count;
uint32_t prev = (deviceId - 1 + count) % count;
HcclSendRecvItem sendRecvInfo[2];
sendRecvInfo[0] = HcclSendRecvItem{HCCL_SEND, sendBuf, count, HCCL_DATA_TYPE_FP32, next};
sendRecvInfo[1] = HcclSendRecvItem{HCCL_RECV, recvBuf, count, HCCL_DATA_TYPE_FP32, prev};
HcclBatchSendRecv(sendRecvInfo, 2, hcclComm, stream);

// 阻塞等待任务流中的集合通信任务执行完成
ACLCHECK(aclrtSynchronizeStream(stream));

// 释放资源
aclrtFree(sendBuf);          // 释放 Device 侧内存
aclrtFree(recvBuf);          // 释放 Device 侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```
