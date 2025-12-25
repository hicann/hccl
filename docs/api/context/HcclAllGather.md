# HcclAllGather<a name="ZH-CN_TOPIC_0000002486832340"></a>

## 产品支持情况<a name="zh-cn_topic_0000001265081270_section10594071513"></a>

<a name="zh-cn_topic_0000001265081270_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265081270_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001265081270_p1883113061818"><a name="zh-cn_topic_0000001265081270_p1883113061818"></a><a name="zh-cn_topic_0000001265081270_p1883113061818"></a><span id="zh-cn_topic_0000001265081270_ph20833205312295"><a name="zh-cn_topic_0000001265081270_ph20833205312295"></a><a name="zh-cn_topic_0000001265081270_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001265081270_p783113012187"><a name="zh-cn_topic_0000001265081270_p783113012187"></a><a name="zh-cn_topic_0000001265081270_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265081270_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265081270_p48327011813"><a name="zh-cn_topic_0000001265081270_p48327011813"></a><a name="zh-cn_topic_0000001265081270_p48327011813"></a><span id="zh-cn_topic_0000001265081270_ph583230201815"><a name="zh-cn_topic_0000001265081270_ph583230201815"></a><a name="zh-cn_topic_0000001265081270_ph583230201815"></a><term id="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265081270_p7948163910184"><a name="zh-cn_topic_0000001265081270_p7948163910184"></a><a name="zh-cn_topic_0000001265081270_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265081270_p14832120181815"><a name="zh-cn_topic_0000001265081270_p14832120181815"></a><a name="zh-cn_topic_0000001265081270_p14832120181815"></a><span id="zh-cn_topic_0000001265081270_ph1292674871116"><a name="zh-cn_topic_0000001265081270_ph1292674871116"></a><a name="zh-cn_topic_0000001265081270_ph1292674871116"></a><term id="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265081270_p19948143911820"><a name="zh-cn_topic_0000001265081270_p19948143911820"></a><a name="zh-cn_topic_0000001265081270_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]说明 
> 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。

## 功能说明<a name="zh-cn_topic_0000001265081270_section59721402"></a>

集合通信算子AllGather的操作接口，将通信域内所有节点的输入按照rank id重新排序，然后拼接起来，再将结果发送到所有节点的输出。

![](figures/allgather-0.png)

> [!NOTE]说明
> 针对AllGather操作，每个节点都接收按照rank id重新排序后的数据集合，即每个节点的AllGather输出都是一样的。

## 函数原型<a name="zh-cn_topic_0000001265081270_section66288034"></a>

```
HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
```

## 参数说明<a name="zh-cn_topic_0000001265081270_section621706"></a>

<a name="zh-cn_topic_0000001265081270_table51170717"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265081270_row27848947"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265081270_p41172271"><a name="zh-cn_topic_0000001265081270_p41172271"></a><a name="zh-cn_topic_0000001265081270_p41172271"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265081270_p46619622"><a name="zh-cn_topic_0000001265081270_p46619622"></a><a name="zh-cn_topic_0000001265081270_p46619622"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265081270_p18093058"><a name="zh-cn_topic_0000001265081270_p18093058"></a><a name="zh-cn_topic_0000001265081270_p18093058"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265081270_row56251627"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p60087944"><a name="zh-cn_topic_0000001265081270_p60087944"></a><a name="zh-cn_topic_0000001265081270_p60087944"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p35285314"><a name="zh-cn_topic_0000001265081270_p35285314"></a><a name="zh-cn_topic_0000001265081270_p35285314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p39538176"><a name="zh-cn_topic_0000001265081270_p39538176"></a><a name="zh-cn_topic_0000001265081270_p39538176"></a>源数据buffer地址。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row20299268"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p33628036"><a name="zh-cn_topic_0000001265081270_p33628036"></a><a name="zh-cn_topic_0000001265081270_p33628036"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p39516413"><a name="zh-cn_topic_0000001265081270_p39516413"></a><a name="zh-cn_topic_0000001265081270_p39516413"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p46712854"><a name="zh-cn_topic_0000001265081270_p46712854"></a><a name="zh-cn_topic_0000001265081270_p46712854"></a>目的数据buffer地址，集合通信结果输出至此buffer中。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row17762502"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p29476530"><a name="zh-cn_topic_0000001265081270_p29476530"></a><a name="zh-cn_topic_0000001265081270_p29476530"></a>sendCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p38788755"><a name="zh-cn_topic_0000001265081270_p38788755"></a><a name="zh-cn_topic_0000001265081270_p38788755"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p54881489"><a name="zh-cn_topic_0000001265081270_p54881489"></a><a name="zh-cn_topic_0000001265081270_p54881489"></a>参与allgather操作的sendBuf的数据size，recvBuf的数据size则等于count * rank size。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row24171358"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p11722994"><a name="zh-cn_topic_0000001265081270_p11722994"></a><a name="zh-cn_topic_0000001265081270_p11722994"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p10038421"><a name="zh-cn_topic_0000001265081270_p10038421"></a><a name="zh-cn_topic_0000001265081270_p10038421"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p51201049151217"><a name="zh-cn_topic_0000001265081270_p51201049151217"></a><a name="zh-cn_topic_0000001265081270_p51201049151217"></a>allgather操作的数据类型，<a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>类型。</p>
<p id="zh-cn_topic_0000001265081270_p578154617218"><a name="zh-cn_topic_0000001265081270_p578154617218"></a><a name="zh-cn_topic_0000001265081270_p578154617218"></a> 针对<span id="zh-cn_topic_0000001265081270_ph13754548217"><a name="zh-cn_topic_0000001265081270_ph13754548217"></a><a name="zh-cn_topic_0000001265081270_ph13754548217"></a><term id="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。</p>
<p id="zh-cn_topic_0000001265081270_p94179211177"><a name="zh-cn_topic_0000001265081270_p94179211177"></a><a name="zh-cn_topic_0000001265081270_p94179211177"></a> 针对<span id="zh-cn_topic_0000001265081270_ph14880920154918"><a name="zh-cn_topic_0000001265081270_ph14880920154918"></a><a name="zh-cn_topic_0000001265081270_ph14880920154918"></a><term id="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001265081270_zh-cn_topic_0000001312391781_term16184138172215"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row3143429"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p53291177"><a name="zh-cn_topic_0000001265081270_p53291177"></a><a name="zh-cn_topic_0000001265081270_p53291177"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p21618074"><a name="zh-cn_topic_0000001265081270_p21618074"></a><a name="zh-cn_topic_0000001265081270_p21618074"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p6233535"><a name="zh-cn_topic_0000001265081270_p6233535"></a><a name="zh-cn_topic_0000001265081270_p6233535"></a>集合通信操作所在的通信域。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081270_row56101816"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081270_p47953287"><a name="zh-cn_topic_0000001265081270_p47953287"></a><a name="zh-cn_topic_0000001265081270_p47953287"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081270_p59011071"><a name="zh-cn_topic_0000001265081270_p59011071"></a><a name="zh-cn_topic_0000001265081270_p59011071"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081270_p15167431"><a name="zh-cn_topic_0000001265081270_p15167431"></a><a name="zh-cn_topic_0000001265081270_p15167431"></a>本rank所使用的stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001265081270_section5595356"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193)：接口成功返回HCCL\_SUCCESS，其他失败。

## 约束说明<a name="zh-cn_topic_0000001265081270_section50358210"></a>

所有rank的sendCount、dataType均应相同。

## 调用示例<a name="zh-cn_topic_0000001265081270_section204039211474"></a>

```c
// 申请集合通信操作的 Device 内存
void *sendBuf = nullptr, *recvBuf = nullptr;
uint32_t rankSize = 8;
uint64_t sendCount = 1;  // 每个节点发送的数据个数
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = rankSize * sendCount * sizeof(float);
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_FIRST);

// 初始化通信域和流
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, devId, &hcclComm);

// 创建任务流
aclrtStream stream;
aclrtCreateStream(&stream);

// 执行 AllGather，将通信域内所有 rank 的 sendBuf 按照 rank_id 顺序拼接起来，再将结果发送到所有 rank 的 recvBuf
HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放 Device 侧内存
aclrtFree(recvBuf);          // 释放 Device 侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```

