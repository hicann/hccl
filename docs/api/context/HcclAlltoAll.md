# HcclAlltoAll<a name="ZH-CN_TOPIC_0000002486992306"></a>

## AI处理器支持情况<a name="zh-cn_topic_0000001690107441_section10594071513"></a>

<a name="zh-cn_topic_0000001690107441_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001690107441_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001690107441_p1883113061818"><a name="zh-cn_topic_0000001690107441_p1883113061818"></a><a name="zh-cn_topic_0000001690107441_p1883113061818"></a><span id="zh-cn_topic_0000001690107441_ph20833205312295"><a name="zh-cn_topic_0000001690107441_ph20833205312295"></a><a name="zh-cn_topic_0000001690107441_ph20833205312295"></a>AI处理器类型</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001690107441_p783113012187"><a name="zh-cn_topic_0000001690107441_p783113012187"></a><a name="zh-cn_topic_0000001690107441_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001690107441_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001690107441_p48327011813"><a name="zh-cn_topic_0000001690107441_p48327011813"></a><a name="zh-cn_topic_0000001690107441_p48327011813"></a><span id="zh-cn_topic_0000001690107441_ph583230201815"><a name="zh-cn_topic_0000001690107441_ph583230201815"></a><a name="zh-cn_topic_0000001690107441_ph583230201815"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225"></a>Ascend 910C</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001690107441_p7948163910184"><a name="zh-cn_topic_0000001690107441_p7948163910184"></a><a name="zh-cn_topic_0000001690107441_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001690107441_p14832120181815"><a name="zh-cn_topic_0000001690107441_p14832120181815"></a><a name="zh-cn_topic_0000001690107441_p14832120181815"></a><span id="zh-cn_topic_0000001690107441_ph1292674871116"><a name="zh-cn_topic_0000001690107441_ph1292674871116"></a><a name="zh-cn_topic_0000001690107441_ph1292674871116"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term11962195213215"></a>Ascend 910B</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001690107441_p19948143911820"><a name="zh-cn_topic_0000001690107441_p19948143911820"></a><a name="zh-cn_topic_0000001690107441_p19948143911820"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row48321040114912"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.1.3.1.1 mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001690107441_p5133163315209"><a name="zh-cn_topic_0000001690107441_p5133163315209"></a><a name="zh-cn_topic_0000001690107441_p5133163315209"></a><span id="zh-cn_topic_0000001690107441_ph71511010202"><a name="zh-cn_topic_0000001690107441_ph71511010202"></a><a name="zh-cn_topic_0000001690107441_ph71511010202"></a>注：AI处理器与昇腾产品的对应关系，请参见《<a href="https://www.hiascend.com/document/detail/zh/AscendFAQ/ProduTech/productform/hardwaredesc_0001.html" target="_blank" rel="noopener noreferrer">昇腾产品形态说明</a>》。</span></p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]说明 
> 针对Ascend 910B，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。

## 功能说明<a name="zh-cn_topic_0000001690107441_section37208511199"></a>

集合通信算子AlltoAll操作接口，向通信域内所有rank发送相同数据量的数据，并从所有rank接收相同数据量的数据。

![](figures/allreduce-5.png)

AlltoAll操作将输入数据在特定的维度切分成特定的块数，并按顺序发送给其他rank，同时从其他rank接收输入数据，按顺序在特定的维度拼接数据。

## 函数原型<a name="zh-cn_topic_0000001690107441_section35919731916"></a>

```
HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf, uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## 参数说明<a name="zh-cn_topic_0000001690107441_section2586134311199"></a>

<a name="zh-cn_topic_0000001690107441_table0576473316"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001690107441_row1060511716320"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001690107441_p146051071139"><a name="zh-cn_topic_0000001690107441_p146051071139"></a><a name="zh-cn_topic_0000001690107441_p146051071139"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001690107441_p1160527939"><a name="zh-cn_topic_0000001690107441_p1160527939"></a><a name="zh-cn_topic_0000001690107441_p1160527939"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001690107441_p86058714320"><a name="zh-cn_topic_0000001690107441_p86058714320"></a><a name="zh-cn_topic_0000001690107441_p86058714320"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001690107441_row166054719318"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p111231019101719"><a name="zh-cn_topic_0000001690107441_p111231019101719"></a><a name="zh-cn_topic_0000001690107441_p111231019101719"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p51231519111711"><a name="zh-cn_topic_0000001690107441_p51231519111711"></a><a name="zh-cn_topic_0000001690107441_p51231519111711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p612301916172"><a name="zh-cn_topic_0000001690107441_p612301916172"></a><a name="zh-cn_topic_0000001690107441_p612301916172"></a>源数据buffer地址。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row460577337"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p412311914178"><a name="zh-cn_topic_0000001690107441_p412311914178"></a><a name="zh-cn_topic_0000001690107441_p412311914178"></a>sendCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p01231219171717"><a name="zh-cn_topic_0000001690107441_p01231219171717"></a><a name="zh-cn_topic_0000001690107441_p01231219171717"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p5123171961713"><a name="zh-cn_topic_0000001690107441_p5123171961713"></a><a name="zh-cn_topic_0000001690107441_p5123171961713"></a>表示向每个rank发送的数据量。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row206057717312"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p81241419101717"><a name="zh-cn_topic_0000001690107441_p81241419101717"></a><a name="zh-cn_topic_0000001690107441_p81241419101717"></a>sendType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p151241419151715"><a name="zh-cn_topic_0000001690107441_p151241419151715"></a><a name="zh-cn_topic_0000001690107441_p151241419151715"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p142831647144118"><a name="zh-cn_topic_0000001690107441_p142831647144118"></a><a name="zh-cn_topic_0000001690107441_p142831647144118"></a>发送数据的数据类型，<a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>类型。</p>
<p id="zh-cn_topic_0000001690107441_p28501558152815"><a name="zh-cn_topic_0000001690107441_p28501558152815"></a><a name="zh-cn_topic_0000001690107441_p28501558152815"></a> 针对<span id="zh-cn_topic_0000001690107441_ph11201161819282"><a name="zh-cn_topic_0000001690107441_ph11201161819282"></a><a name="zh-cn_topic_0000001690107441_ph11201161819282"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Ascend 910C</term></span>，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。</p>
<p id="zh-cn_topic_0000001690107441_p12283142716193"><a name="zh-cn_topic_0000001690107441_p12283142716193"></a><a name="zh-cn_topic_0000001690107441_p12283142716193"></a> 针对<span id="zh-cn_topic_0000001690107441_ph161710422273"><a name="zh-cn_topic_0000001690107441_ph161710422273"></a><a name="zh-cn_topic_0000001690107441_ph161710422273"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215"></a>Ascend 910B</term></span>，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row146051372315"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p512431910174"><a name="zh-cn_topic_0000001690107441_p512431910174"></a><a name="zh-cn_topic_0000001690107441_p512431910174"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p151241419131716"><a name="zh-cn_topic_0000001690107441_p151241419131716"></a><a name="zh-cn_topic_0000001690107441_p151241419131716"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p161241219201719"><a name="zh-cn_topic_0000001690107441_p161241219201719"></a><a name="zh-cn_topic_0000001690107441_p161241219201719"></a>目的数据buffer地址，集合通信结果输出至此buffer中。</p>
<p id="zh-cn_topic_0000001690107441_p57711733112312"><a name="zh-cn_topic_0000001690107441_p57711733112312"></a><a name="zh-cn_topic_0000001690107441_p57711733112312"></a>recvBuf与sendBuf配置的地址不能相同。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row14605137334"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p121254195173"><a name="zh-cn_topic_0000001690107441_p121254195173"></a><a name="zh-cn_topic_0000001690107441_p121254195173"></a>recvCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p4125181921717"><a name="zh-cn_topic_0000001690107441_p4125181921717"></a><a name="zh-cn_topic_0000001690107441_p4125181921717"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p612571911172"><a name="zh-cn_topic_0000001690107441_p612571911172"></a><a name="zh-cn_topic_0000001690107441_p612571911172"></a>表示从每个rank接收的数据量。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row6173151115173"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p13125319181717"><a name="zh-cn_topic_0000001690107441_p13125319181717"></a><a name="zh-cn_topic_0000001690107441_p13125319181717"></a>recvType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p11125141921712"><a name="zh-cn_topic_0000001690107441_p11125141921712"></a><a name="zh-cn_topic_0000001690107441_p11125141921712"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p1712581910177"><a name="zh-cn_topic_0000001690107441_p1712581910177"></a><a name="zh-cn_topic_0000001690107441_p1712581910177"></a>接收数据的数据类型，<a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>类型。</p>
<p id="zh-cn_topic_0000001690107441_p14891165052917"><a name="zh-cn_topic_0000001690107441_p14891165052917"></a><a name="zh-cn_topic_0000001690107441_p14891165052917"></a> 针对<span id="zh-cn_topic_0000001690107441_ph10615564290"><a name="zh-cn_topic_0000001690107441_ph10615564290"></a><a name="zh-cn_topic_0000001690107441_ph10615564290"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_2"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_2"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term1253731311225_2"></a>Ascend 910C</term></span>，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。</p>
<p id="zh-cn_topic_0000001690107441_p12838161092110"><a name="zh-cn_topic_0000001690107441_p12838161092110"></a><a name="zh-cn_topic_0000001690107441_p12838161092110"></a> 针对<span id="zh-cn_topic_0000001690107441_ph17071145182910"><a name="zh-cn_topic_0000001690107441_ph17071145182910"></a><a name="zh-cn_topic_0000001690107441_ph17071145182910"></a><term id="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215_1"><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215_1"></a><a name="zh-cn_topic_0000001690107441_zh-cn_topic_0000001312391781_term16184138172215_1"></a>Ascend 910B</term></span>，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row9173121111174"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p6125101981716"><a name="zh-cn_topic_0000001690107441_p6125101981716"></a><a name="zh-cn_topic_0000001690107441_p6125101981716"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p1312561917172"><a name="zh-cn_topic_0000001690107441_p1312561917172"></a><a name="zh-cn_topic_0000001690107441_p1312561917172"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p512551951715"><a name="zh-cn_topic_0000001690107441_p512551951715"></a><a name="zh-cn_topic_0000001690107441_p512551951715"></a>集合通信操作所在的通信域。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001690107441_row2017331101714"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001690107441_p712516198177"><a name="zh-cn_topic_0000001690107441_p712516198177"></a><a name="zh-cn_topic_0000001690107441_p712516198177"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001690107441_p812551921714"><a name="zh-cn_topic_0000001690107441_p812551921714"></a><a name="zh-cn_topic_0000001690107441_p812551921714"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001690107441_p41253199175"><a name="zh-cn_topic_0000001690107441_p41253199175"></a><a name="zh-cn_topic_0000001690107441_p41253199175"></a>本rank所使用的stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001690107441_section12554172517195"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193)：接口成功返回HCCL\_SUCCESS，其他失败。

## 约束说明<a name="zh-cn_topic_0000001690107441_section92549325194"></a>

-   所有rank的sendCount、sendType、recvCount、recvType均应相同。
-   AlltoAll操作的性能与NPU之间共享数据的缓存区大小有关，当通信数据量超过缓存区大小时性能将出现明显下降。若业务中AlltoAll通信数据量较大，建议通过配置环境变量HCCL\_BUFFSIZE适当增大缓存区大小以提升通信性能。

## 调用示例<a name="zh-cn_topic_0000001690107441_section204039211474"></a>

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

// 执行 AlltoAll，向通信域内所有 rank 发送相同数据量的数据，并从所有 rank 接收相同数据量的数据
size_t perCount = count / rankSize;
HcclAlltoAll(sendBuf, perCount, HCCL_DATA_TYPE_FP32, recvBuf, perCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放 Device 侧内存
aclrtFree(recvBuf);          // 释放 Device 侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```

