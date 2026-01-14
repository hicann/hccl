# HcclBatchSendRecv<a name="ZH-CN_TOPIC_0000002486993298"></a>

## 产品支持情况<a name="zh-cn_topic_0000001811681609_section10594071513"></a>

<a name="zh-cn_topic_0000001811681609_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001811681609_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001811681609_p1883113061818"><a name="zh-cn_topic_0000001811681609_p1883113061818"></a><a name="zh-cn_topic_0000001811681609_p1883113061818"></a><span id="zh-cn_topic_0000001811681609_ph20833205312295"><a name="zh-cn_topic_0000001811681609_ph20833205312295"></a><a name="zh-cn_topic_0000001811681609_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001811681609_p783113012187"><a name="zh-cn_topic_0000001811681609_p783113012187"></a><a name="zh-cn_topic_0000001811681609_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001811681609_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001811681609_p48327011813"><a name="zh-cn_topic_0000001811681609_p48327011813"></a><a name="zh-cn_topic_0000001811681609_p48327011813"></a><span id="zh-cn_topic_0000001811681609_ph583230201815"><a name="zh-cn_topic_0000001811681609_ph583230201815"></a><a name="zh-cn_topic_0000001811681609_ph583230201815"></a><term id="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001811681609_p7948163910184"><a name="zh-cn_topic_0000001811681609_p7948163910184"></a><a name="zh-cn_topic_0000001811681609_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001811681609_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001811681609_p14832120181815"><a name="zh-cn_topic_0000001811681609_p14832120181815"></a><a name="zh-cn_topic_0000001811681609_p14832120181815"></a><span id="zh-cn_topic_0000001811681609_ph1292674871116"><a name="zh-cn_topic_0000001811681609_ph1292674871116"></a><a name="zh-cn_topic_0000001811681609_ph1292674871116"></a><term id="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001811681609_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001811681609_p19948143911820"><a name="zh-cn_topic_0000001811681609_p19948143911820"></a><a name="zh-cn_topic_0000001811681609_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]说明 
> 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。

## 功能说明<a name="zh-cn_topic_0000001811681609_section212645315215"></a>

异步批量点对点通信操作接口，调用一次接口可以完成本rank上的多个收发任务，本rank发送和接收之间是异步的，发送和接收任务之间不会相互阻塞。

## 函数原型<a name="zh-cn_topic_0000001811681609_section13125135314218"></a>

```
HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
```

## 参数说明<a name="zh-cn_topic_0000001811681609_section1812717539212"></a>

<a name="zh-cn_topic_0000001811681609_table18137135310213"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001811681609_row1417285311217"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001811681609_p131726530216"><a name="zh-cn_topic_0000001811681609_p131726530216"></a><a name="zh-cn_topic_0000001811681609_p131726530216"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001811681609_p01721653524"><a name="zh-cn_topic_0000001811681609_p01721653524"></a><a name="zh-cn_topic_0000001811681609_p01721653524"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001811681609_p7172195319214"><a name="zh-cn_topic_0000001811681609_p7172195319214"></a><a name="zh-cn_topic_0000001811681609_p7172195319214"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001811681609_row1117295311215"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001811681609_p195466529172"><a name="zh-cn_topic_0000001811681609_p195466529172"></a><a name="zh-cn_topic_0000001811681609_p195466529172"></a>sendRecvInfo</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001811681609_p161721753129"><a name="zh-cn_topic_0000001811681609_p161721753129"></a><a name="zh-cn_topic_0000001811681609_p161721753129"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001811681609_p5172353028"><a name="zh-cn_topic_0000001811681609_p5172353028"></a><a name="zh-cn_topic_0000001811681609_p5172353028"></a>本rank需要下发的收发任务列表的首地址。</p>
<p id="zh-cn_topic_0000001811681609_p14928105415418"><a name="zh-cn_topic_0000001811681609_p14928105415418"></a><a name="zh-cn_topic_0000001811681609_p14928105415418"></a>HcclSendRecvItem类型，详细可参见<a href="HcclSendRecvItem.md#ZH-CN_TOPIC_0000002519072197">HcclSendRecvItem</a>。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001811681609_row41722531724"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001811681609_p38341698166"><a name="zh-cn_topic_0000001811681609_p38341698166"></a><a name="zh-cn_topic_0000001811681609_p38341698166"></a>itemNum</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001811681609_p617275320217"><a name="zh-cn_topic_0000001811681609_p617275320217"></a><a name="zh-cn_topic_0000001811681609_p617275320217"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001811681609_p41729531229"><a name="zh-cn_topic_0000001811681609_p41729531229"></a><a name="zh-cn_topic_0000001811681609_p41729531229"></a>本rank需要接收和发送的任务个数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001811681609_row1117220531028"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001811681609_p15172653628"><a name="zh-cn_topic_0000001811681609_p15172653628"></a><a name="zh-cn_topic_0000001811681609_p15172653628"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001811681609_p1517219536214"><a name="zh-cn_topic_0000001811681609_p1517219536214"></a><a name="zh-cn_topic_0000001811681609_p1517219536214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001811681609_p834310091714"><a name="zh-cn_topic_0000001811681609_p834310091714"></a><a name="zh-cn_topic_0000001811681609_p834310091714"></a>集合通信操作所在的通信域。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001811681609_row18172165315213"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001811681609_p101724536217"><a name="zh-cn_topic_0000001811681609_p101724536217"></a><a name="zh-cn_topic_0000001811681609_p101724536217"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001811681609_p917215320212"><a name="zh-cn_topic_0000001811681609_p917215320212"></a><a name="zh-cn_topic_0000001811681609_p917215320212"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001811681609_p88321247141619"><a name="zh-cn_topic_0000001811681609_p88321247141619"></a><a name="zh-cn_topic_0000001811681609_p88321247141619"></a>本rank所使用的stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001811681609_section1513715531221"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193)：接口成功返回HCCL\_SUCCESS，其他失败。

## 约束说明<a name="zh-cn_topic_0000001811681609_section86843302218"></a>

-   “异步”是指同一张卡上的接收和发送任务是异步的，不会相互阻塞。但是在卡间，收发任务依旧是同步的，因此，卡间的收发任务也同HcclSend、HcclRecv一样，必须是一一对应的。
-   任务列表中不能有重复的send/recv任务，重复指向（从）同一rank发送（接收）的两个任务。
-   当前版本此接口不支持Virtual Pipeline（VPP）开启的场景。
-   针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，在大规模集群下（ranksize\>500）使用此接口时，并发执行数不能超过3个。
-   针对[Atlas 200T A2 Box16 异构子框](https://support.huawei.com/enterprise/zh/doc/EDOC1100318274/287e0458)，若Server内卡间出现建链失败的情况（错误码：EI0010），需要将环境变量HCCL\_INTRA\_ROCE\_ENABLE配置为1，HCCL\_INTRA\_PCIE\_ENABLE配置为0，让Server内采用RoCE环路进行多卡间的通信（请确保Server上存在RoCE网卡，且具有send/recv收发关系的设备之间RDMA链路互通），环境变量配置示例如下：

    ```
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_INTRA_PCIE_ENABLE=0
    ```

## 调用示例<a name="zh-cn_topic_0000001811681609_section204039211474"></a>

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

