# HcclAllReduce<a name="ZH-CN_TOPIC_0000002518992195"></a>

## 产品支持情况<a name="zh-cn_topic_0000001312641237_section161778316247"></a>

<a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001312641237_p1883113061818"><a name="zh-cn_topic_0000001312641237_p1883113061818"></a><a name="zh-cn_topic_0000001312641237_p1883113061818"></a><span id="zh-cn_topic_0000001312641237_ph20833205312295"><a name="zh-cn_topic_0000001312641237_ph20833205312295"></a><a name="zh-cn_topic_0000001312641237_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p783113012187"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p783113012187"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312641237_p48327011813"><a name="zh-cn_topic_0000001312641237_p48327011813"></a><a name="zh-cn_topic_0000001312641237_p48327011813"></a><span id="zh-cn_topic_0000001312641237_ph583230201815"><a name="zh-cn_topic_0000001312641237_ph583230201815"></a><a name="zh-cn_topic_0000001312641237_ph583230201815"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p7948163910184"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p7948163910184"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312641237_p14832120181815"><a name="zh-cn_topic_0000001312641237_p14832120181815"></a><a name="zh-cn_topic_0000001312641237_p14832120181815"></a><span id="zh-cn_topic_0000001312641237_ph1292674871116"><a name="zh-cn_topic_0000001312641237_ph1292674871116"></a><a name="zh-cn_topic_0000001312641237_ph1292674871116"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p19948143911820"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p19948143911820"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001264921398_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]说明 
> 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。

## 功能说明<a name="zh-cn_topic_0000001312641237_section48254661"></a>

集合通信算子AllReduce的操作接口，将通信域内所有节点的输入数据进行相加（或其他归约操作）后，再把结果发送到所有节点的输出buffer，其中归约操作类型由op参数指定。

![](figures/allreduce.png)

## 函数原型<a name="zh-cn_topic_0000001312641237_section57557412"></a>

```
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## 参数说明<a name="zh-cn_topic_0000001312641237_section31638772"></a>

<a name="zh-cn_topic_0000001312641237_table66592127"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312641237_row61502840"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001312641237_p15674164"><a name="zh-cn_topic_0000001312641237_p15674164"></a><a name="zh-cn_topic_0000001312641237_p15674164"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.03%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001312641237_p61647805"><a name="zh-cn_topic_0000001312641237_p61647805"></a><a name="zh-cn_topic_0000001312641237_p61647805"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="62.77%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001312641237_p27416314"><a name="zh-cn_topic_0000001312641237_p27416314"></a><a name="zh-cn_topic_0000001312641237_p27416314"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312641237_row6128980"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p26685362"><a name="zh-cn_topic_0000001312641237_p26685362"></a><a name="zh-cn_topic_0000001312641237_p26685362"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p14030717"><a name="zh-cn_topic_0000001312641237_p14030717"></a><a name="zh-cn_topic_0000001312641237_p14030717"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p62746268"><a name="zh-cn_topic_0000001312641237_p62746268"></a><a name="zh-cn_topic_0000001312641237_p62746268"></a>源数据buffer地址。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row27845503"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p40893240"><a name="zh-cn_topic_0000001312641237_p40893240"></a><a name="zh-cn_topic_0000001312641237_p40893240"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p24018105"><a name="zh-cn_topic_0000001312641237_p24018105"></a><a name="zh-cn_topic_0000001312641237_p24018105"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p66418347"><a name="zh-cn_topic_0000001312641237_p66418347"></a><a name="zh-cn_topic_0000001312641237_p66418347"></a>目的数据buffer地址，集合通信结果输出至此buffer中。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row60894213"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p33484259"><a name="zh-cn_topic_0000001312641237_p33484259"></a><a name="zh-cn_topic_0000001312641237_p33484259"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p27870469"><a name="zh-cn_topic_0000001312641237_p27870469"></a><a name="zh-cn_topic_0000001312641237_p27870469"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p42915540"><a name="zh-cn_topic_0000001312641237_p42915540"></a><a name="zh-cn_topic_0000001312641237_p42915540"></a>参与allreduce操作的数据个数，比如只有一个int32数据参与，则count=1。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row50695543"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p12698356"><a name="zh-cn_topic_0000001312641237_p12698356"></a><a name="zh-cn_topic_0000001312641237_p12698356"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p21933905"><a name="zh-cn_topic_0000001312641237_p21933905"></a><a name="zh-cn_topic_0000001312641237_p21933905"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p1532218361093"><a name="zh-cn_topic_0000001312641237_p1532218361093"></a><a name="zh-cn_topic_0000001312641237_p1532218361093"></a>allreduce操作的数据类型，<a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>类型。</p>
<p id="zh-cn_topic_0000001312641237_p1516617265207"><a name="zh-cn_topic_0000001312641237_p1516617265207"></a><a name="zh-cn_topic_0000001312641237_p1516617265207"></a> 针对<span id="zh-cn_topic_0000001312641237_ph13754548217"><a name="zh-cn_topic_0000001312641237_ph13754548217"></a><a name="zh-cn_topic_0000001312641237_ph13754548217"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。</p>
<p id="zh-cn_topic_0000001312641237_p195289510913"><a name="zh-cn_topic_0000001312641237_p195289510913"></a><a name="zh-cn_topic_0000001312641237_p195289510913"></a> 针对<span id="zh-cn_topic_0000001312641237_ph14880920154918"><a name="zh-cn_topic_0000001312641237_ph14880920154918"></a><a name="zh-cn_topic_0000001312641237_ph14880920154918"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>，支持数据类型：int8、int16、int32、int64、float16、float32、bfp16。需要注意，针对int64数据类型，性能会有一定的劣化。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row17907308"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p41205809"><a name="zh-cn_topic_0000001312641237_p41205809"></a><a name="zh-cn_topic_0000001312641237_p41205809"></a>op</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p49336210"><a name="zh-cn_topic_0000001312641237_p49336210"></a><a name="zh-cn_topic_0000001312641237_p49336210"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p36810105"><a name="zh-cn_topic_0000001312641237_p36810105"></a><a name="zh-cn_topic_0000001312641237_p36810105"></a>reduce的操作类型，目前支持操作类型为sum、prod、max、min。</p>
<div class="note" id="zh-cn_topic_0000001312641237_note5731134315341"><a name="zh-cn_topic_0000001312641237_note5731134315341"></a><a name="zh-cn_topic_0000001312641237_note5731134315341"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="zh-cn_topic_0000001312641237_p9984151202012"><a name="zh-cn_topic_0000001312641237_p9984151202012"></a><a name="zh-cn_topic_0000001312641237_p9984151202012"></a> 针对<span id="zh-cn_topic_0000001312641237_ph79242619219"><a name="zh-cn_topic_0000001312641237_ph79242619219"></a><a name="zh-cn_topic_0000001312641237_ph79242619219"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term1253731311225_2"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>，“prod”操作不支持int16、bfp16数据类型。</p>
<p id="zh-cn_topic_0000001312641237_p10731124313342"><a name="zh-cn_topic_0000001312641237_p10731124313342"></a><a name="zh-cn_topic_0000001312641237_p10731124313342"></a> 针对<span id="zh-cn_topic_0000001312641237_ph49172713419"><a name="zh-cn_topic_0000001312641237_ph49172713419"></a><a name="zh-cn_topic_0000001312641237_ph49172713419"></a><term id="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215_1"><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215_1"></a><a name="zh-cn_topic_0000001312641237_zh-cn_topic_0000001312391781_term16184138172215_1"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>，“prod”操作不支持int16、bfp16数据类型。</p>
</div></div>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row62855489"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p58129833"><a name="zh-cn_topic_0000001312641237_p58129833"></a><a name="zh-cn_topic_0000001312641237_p58129833"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p10896009"><a name="zh-cn_topic_0000001312641237_p10896009"></a><a name="zh-cn_topic_0000001312641237_p10896009"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p10161546"><a name="zh-cn_topic_0000001312641237_p10161546"></a><a name="zh-cn_topic_0000001312641237_p10161546"></a>集合通信操作所在的通信域。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312641237_row24345054"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312641237_p25792371"><a name="zh-cn_topic_0000001312641237_p25792371"></a><a name="zh-cn_topic_0000001312641237_p25792371"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.03%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312641237_p8807292"><a name="zh-cn_topic_0000001312641237_p8807292"></a><a name="zh-cn_topic_0000001312641237_p8807292"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.77%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312641237_p42302050"><a name="zh-cn_topic_0000001312641237_p42302050"></a><a name="zh-cn_topic_0000001312641237_p42302050"></a>本rank所使用的stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001312641237_section16313497"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193)：接口成功返回HCCL\_SUCCESS，其他失败。

## 约束说明<a name="zh-cn_topic_0000001312641237_section12603749"></a>

-   所有rank的count、dataType、op均应相同。
-   每个rank只能有一个输入。

## 调用示例<a name="zh-cn_topic_0000001312641237_section204039211474"></a>

```c
// 申请集合通信操作的 Device 内存
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t count = 8;
size_t mallocSize = count * sizeof(float);
aclrtMalloc((void **)&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void **)&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_FIRST);

// 初始化通信域
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// 创建任务流
aclrtStream stream;
aclrtCreateStream(&stream);

// 执行 AllReduce，将通信域内所有节点的输入数据进行相加后，再把结果发送到所有节点的输出buffer
HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放 Device 侧内存
aclrtFree(recvBuf);          // 释放 Device 侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```

