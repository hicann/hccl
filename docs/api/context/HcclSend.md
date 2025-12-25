# HcclSend<a name="ZH-CN_TOPIC_0000002519073171"></a>

## 产品支持情况<a name="zh-cn_topic_0000001265081266_section10594071513"></a>

<a name="zh-cn_topic_0000001265081266_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265081266_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001265081266_p1883113061818"><a name="zh-cn_topic_0000001265081266_p1883113061818"></a><a name="zh-cn_topic_0000001265081266_p1883113061818"></a><span id="zh-cn_topic_0000001265081266_ph20833205312295"><a name="zh-cn_topic_0000001265081266_ph20833205312295"></a><a name="zh-cn_topic_0000001265081266_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001265081266_p783113012187"><a name="zh-cn_topic_0000001265081266_p783113012187"></a><a name="zh-cn_topic_0000001265081266_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265081266_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265081266_p48327011813"><a name="zh-cn_topic_0000001265081266_p48327011813"></a><a name="zh-cn_topic_0000001265081266_p48327011813"></a><span id="zh-cn_topic_0000001265081266_ph583230201815"><a name="zh-cn_topic_0000001265081266_ph583230201815"></a><a name="zh-cn_topic_0000001265081266_ph583230201815"></a><term id="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265081266_p7948163910184"><a name="zh-cn_topic_0000001265081266_p7948163910184"></a><a name="zh-cn_topic_0000001265081266_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081266_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001265081266_p14832120181815"><a name="zh-cn_topic_0000001265081266_p14832120181815"></a><a name="zh-cn_topic_0000001265081266_p14832120181815"></a><span id="zh-cn_topic_0000001265081266_ph1292674871116"><a name="zh-cn_topic_0000001265081266_ph1292674871116"></a><a name="zh-cn_topic_0000001265081266_ph1292674871116"></a><term id="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001265081266_p19948143911820"><a name="zh-cn_topic_0000001265081266_p19948143911820"></a><a name="zh-cn_topic_0000001265081266_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]说明 
> 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。

## 功能说明<a name="zh-cn_topic_0000001265081266_section212645315215"></a>

点对点通信Send操作接口，将当前节点指定位置的数据发送至目的节点的指定位置上。

## 函数原型<a name="zh-cn_topic_0000001265081266_section13125135314218"></a>

```
HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,HcclComm comm, aclrtStream stream)
```

## 参数说明<a name="zh-cn_topic_0000001265081266_section1812717539212"></a>

<a name="zh-cn_topic_0000001265081266_table18137135310213"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001265081266_row1417285311217"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001265081266_p131726530216"><a name="zh-cn_topic_0000001265081266_p131726530216"></a><a name="zh-cn_topic_0000001265081266_p131726530216"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001265081266_p01721653524"><a name="zh-cn_topic_0000001265081266_p01721653524"></a><a name="zh-cn_topic_0000001265081266_p01721653524"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001265081266_p7172195319214"><a name="zh-cn_topic_0000001265081266_p7172195319214"></a><a name="zh-cn_topic_0000001265081266_p7172195319214"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001265081266_row1117295311215"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081266_p14172185315214"><a name="zh-cn_topic_0000001265081266_p14172185315214"></a><a name="zh-cn_topic_0000001265081266_p14172185315214"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081266_p161721753129"><a name="zh-cn_topic_0000001265081266_p161721753129"></a><a name="zh-cn_topic_0000001265081266_p161721753129"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081266_p5172353028"><a name="zh-cn_topic_0000001265081266_p5172353028"></a><a name="zh-cn_topic_0000001265081266_p5172353028"></a>源数据buffer地址。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081266_row41722531724"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081266_p11729539216"><a name="zh-cn_topic_0000001265081266_p11729539216"></a><a name="zh-cn_topic_0000001265081266_p11729539216"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081266_p617275320217"><a name="zh-cn_topic_0000001265081266_p617275320217"></a><a name="zh-cn_topic_0000001265081266_p617275320217"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081266_p41729531229"><a name="zh-cn_topic_0000001265081266_p41729531229"></a><a name="zh-cn_topic_0000001265081266_p41729531229"></a>发送数据的个数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081266_row1117220531028"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081266_p15172653628"><a name="zh-cn_topic_0000001265081266_p15172653628"></a><a name="zh-cn_topic_0000001265081266_p15172653628"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081266_p1517219536214"><a name="zh-cn_topic_0000001265081266_p1517219536214"></a><a name="zh-cn_topic_0000001265081266_p1517219536214"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081266_p631813971515"><a name="zh-cn_topic_0000001265081266_p631813971515"></a><a name="zh-cn_topic_0000001265081266_p631813971515"></a>发送数据的数据类型，<a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>类型。</p>
<p id="zh-cn_topic_0000001265081266_p1752124102514"><a name="zh-cn_topic_0000001265081266_p1752124102514"></a><a name="zh-cn_topic_0000001265081266_p1752124102514"></a> 针对<span id="zh-cn_topic_0000001265081266_ph13754548217"><a name="zh-cn_topic_0000001265081266_ph13754548217"></a><a name="zh-cn_topic_0000001265081266_ph13754548217"></a><term id="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。</p>
<p id="zh-cn_topic_0000001265081266_p852044961512"><a name="zh-cn_topic_0000001265081266_p852044961512"></a><a name="zh-cn_topic_0000001265081266_p852044961512"></a> 针对<span id="zh-cn_topic_0000001265081266_ph14880920154918"><a name="zh-cn_topic_0000001265081266_ph14880920154918"></a><a name="zh-cn_topic_0000001265081266_ph14880920154918"></a><term id="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001265081266_zh-cn_topic_0000001312391781_term16184138172215"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64，bfp16。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081266_row18172165315213"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081266_p101724536217"><a name="zh-cn_topic_0000001265081266_p101724536217"></a><a name="zh-cn_topic_0000001265081266_p101724536217"></a>destRank</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081266_p917215320212"><a name="zh-cn_topic_0000001265081266_p917215320212"></a><a name="zh-cn_topic_0000001265081266_p917215320212"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081266_p151721553528"><a name="zh-cn_topic_0000001265081266_p151721553528"></a><a name="zh-cn_topic_0000001265081266_p151721553528"></a>通信域内数据接收端的rank编号。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081266_row16172165312215"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081266_p20172853822"><a name="zh-cn_topic_0000001265081266_p20172853822"></a><a name="zh-cn_topic_0000001265081266_p20172853822"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081266_p17172853121"><a name="zh-cn_topic_0000001265081266_p17172853121"></a><a name="zh-cn_topic_0000001265081266_p17172853121"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081266_p191724539212"><a name="zh-cn_topic_0000001265081266_p191724539212"></a><a name="zh-cn_topic_0000001265081266_p191724539212"></a>集合通信操作所在的通信域。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001265081266_row6172135311215"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001265081266_p4172135314211"><a name="zh-cn_topic_0000001265081266_p4172135314211"></a><a name="zh-cn_topic_0000001265081266_p4172135314211"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001265081266_p19172553129"><a name="zh-cn_topic_0000001265081266_p19172553129"></a><a name="zh-cn_topic_0000001265081266_p19172553129"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001265081266_p017215310217"><a name="zh-cn_topic_0000001265081266_p017215310217"></a><a name="zh-cn_topic_0000001265081266_p017215310217"></a>本rank所使用的stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001265081266_section1513715531221"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193)：接口成功返回HCCL\_SUCCESS，其他失败。

## 约束说明<a name="zh-cn_topic_0000001265081266_section86843302218"></a>

HcclSend与HcclRecv接口采用同步调用方式，且必须配对使用。即一个进程调用HcclSend接口后，需要等到与之配对的HcclRecv接口接收数据后，才可以进行下一个接口调用，如下图所示。

![](figures/zh-cn_image_0000001532063748.png)

## 调用示例<a name="zh-cn_topic_0000001265081266_section204039211474"></a>

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
    aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_FIRST);
    // 初始化输入数据
    aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE);
    // 执行 Send 操作
    HcclSend(sendBuf, count, HCCL_DATA_TYPE_FP32, deviceId + 1, hcclComm, stream);
} else {
    // 申请 Device 内存用于接收数据
    aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_FIRST);
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

