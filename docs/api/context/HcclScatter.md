# HcclScatter<a name="ZH-CN_TOPIC_0000002486992308"></a>

## 产品支持情况<a name="zh-cn_topic_0000001779917185_section10594071513"></a>

<a name="zh-cn_topic_0000001779917185_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001779917185_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001779917185_p1883113061818"><a name="zh-cn_topic_0000001779917185_p1883113061818"></a><a name="zh-cn_topic_0000001779917185_p1883113061818"></a><span id="zh-cn_topic_0000001779917185_ph20833205312295"><a name="zh-cn_topic_0000001779917185_ph20833205312295"></a><a name="zh-cn_topic_0000001779917185_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001779917185_p783113012187"><a name="zh-cn_topic_0000001779917185_p783113012187"></a><a name="zh-cn_topic_0000001779917185_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001779917185_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001779917185_p48327011813"><a name="zh-cn_topic_0000001779917185_p48327011813"></a><a name="zh-cn_topic_0000001779917185_p48327011813"></a><span id="zh-cn_topic_0000001779917185_ph583230201815"><a name="zh-cn_topic_0000001779917185_ph583230201815"></a><a name="zh-cn_topic_0000001779917185_ph583230201815"></a><term id="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001779917185_p7948163910184"><a name="zh-cn_topic_0000001779917185_p7948163910184"></a><a name="zh-cn_topic_0000001779917185_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001779917185_p14832120181815"><a name="zh-cn_topic_0000001779917185_p14832120181815"></a><a name="zh-cn_topic_0000001779917185_p14832120181815"></a><span id="zh-cn_topic_0000001779917185_ph1292674871116"><a name="zh-cn_topic_0000001779917185_ph1292674871116"></a><a name="zh-cn_topic_0000001779917185_ph1292674871116"></a><term id="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001779917185_p19948143911820"><a name="zh-cn_topic_0000001779917185_p19948143911820"></a><a name="zh-cn_topic_0000001779917185_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]说明 
> 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。

## 功能说明<a name="zh-cn_topic_0000001779917185_section662317284469"></a>

集合通信算子Scatter操作接口，将root节点的数据均分并散布至其他rank。

## 函数原型<a name="zh-cn_topic_0000001779917185_section13621172844611"></a>

```
HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
```

## 参数说明<a name="zh-cn_topic_0000001779917185_section1462482884619"></a>

<a name="zh-cn_topic_0000001779917185_table24749807"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001779917185_row60665573"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001779917185_p14964341"><a name="zh-cn_topic_0000001779917185_p14964341"></a><a name="zh-cn_topic_0000001779917185_p14964341"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001779917185_p4152081"><a name="zh-cn_topic_0000001779917185_p4152081"></a><a name="zh-cn_topic_0000001779917185_p4152081"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001779917185_p774306"><a name="zh-cn_topic_0000001779917185_p774306"></a><a name="zh-cn_topic_0000001779917185_p774306"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001779917185_row62718864"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p47063234"><a name="zh-cn_topic_0000001779917185_p47063234"></a><a name="zh-cn_topic_0000001779917185_p47063234"></a>sendBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p54025633"><a name="zh-cn_topic_0000001779917185_p54025633"></a><a name="zh-cn_topic_0000001779917185_p54025633"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p14000148"><a name="zh-cn_topic_0000001779917185_p14000148"></a><a name="zh-cn_topic_0000001779917185_p14000148"></a>源数据buffer地址。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row58892473"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p5560998"><a name="zh-cn_topic_0000001779917185_p5560998"></a><a name="zh-cn_topic_0000001779917185_p5560998"></a>recvBuf</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p47787675"><a name="zh-cn_topic_0000001779917185_p47787675"></a><a name="zh-cn_topic_0000001779917185_p47787675"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p45596481"><a name="zh-cn_topic_0000001779917185_p45596481"></a><a name="zh-cn_topic_0000001779917185_p45596481"></a>目的数据buffer地址，集合通信结果输出至此buffer中。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row7715150"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p20947391"><a name="zh-cn_topic_0000001779917185_p20947391"></a><a name="zh-cn_topic_0000001779917185_p20947391"></a>recvCount</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p19017142"><a name="zh-cn_topic_0000001779917185_p19017142"></a><a name="zh-cn_topic_0000001779917185_p19017142"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p63993496"><a name="zh-cn_topic_0000001779917185_p63993496"></a><a name="zh-cn_topic_0000001779917185_p63993496"></a>参与scatter操作的recvBuf的数据个数，比如只有一个int32数据参与，则count=1。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row39070558"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p10598606"><a name="zh-cn_topic_0000001779917185_p10598606"></a><a name="zh-cn_topic_0000001779917185_p10598606"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p53180767"><a name="zh-cn_topic_0000001779917185_p53180767"></a><a name="zh-cn_topic_0000001779917185_p53180767"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p7991229191118"><a name="zh-cn_topic_0000001779917185_p7991229191118"></a><a name="zh-cn_topic_0000001779917185_p7991229191118"></a>Scatter操作的数据类型，<a href="HcclDataType.md#ZH-CN_TOPIC_0000002486992310">HcclDataType</a>类型。</p>
<p id="zh-cn_topic_0000001779917185_p16850153843018"><a name="zh-cn_topic_0000001779917185_p16850153843018"></a><a name="zh-cn_topic_0000001779917185_p16850153843018"></a> 针对<span id="zh-cn_topic_0000001779917185_ph13754548217"><a name="zh-cn_topic_0000001779917185_ph13754548217"></a><a name="zh-cn_topic_0000001779917185_ph13754548217"></a><term id="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225_1"><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225_1"></a><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term1253731311225_1"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span>，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。</p>
<p id="zh-cn_topic_0000001779917185_p94179211177"><a name="zh-cn_topic_0000001779917185_p94179211177"></a><a name="zh-cn_topic_0000001779917185_p94179211177"></a> 针对<span id="zh-cn_topic_0000001779917185_ph841715341959"><a name="zh-cn_topic_0000001779917185_ph841715341959"></a><a name="zh-cn_topic_0000001779917185_ph841715341959"></a><term id="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term16184138172215"><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term16184138172215"></a><a name="zh-cn_topic_0000001779917185_zh-cn_topic_0000001312391781_term16184138172215"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span>，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row46964992"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p7641038"><a name="zh-cn_topic_0000001779917185_p7641038"></a><a name="zh-cn_topic_0000001779917185_p7641038"></a>root</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p14944322"><a name="zh-cn_topic_0000001779917185_p14944322"></a><a name="zh-cn_topic_0000001779917185_p14944322"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p2530563"><a name="zh-cn_topic_0000001779917185_p2530563"></a><a name="zh-cn_topic_0000001779917185_p2530563"></a>作为scatter root的rank id。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row11144211"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p30265903"><a name="zh-cn_topic_0000001779917185_p30265903"></a><a name="zh-cn_topic_0000001779917185_p30265903"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p35619075"><a name="zh-cn_topic_0000001779917185_p35619075"></a><a name="zh-cn_topic_0000001779917185_p35619075"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p66572856"><a name="zh-cn_topic_0000001779917185_p66572856"></a><a name="zh-cn_topic_0000001779917185_p66572856"></a>集合通信操作所在的通信域。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001779917185_row62284798"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001779917185_p11903911"><a name="zh-cn_topic_0000001779917185_p11903911"></a><a name="zh-cn_topic_0000001779917185_p11903911"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001779917185_p24692740"><a name="zh-cn_topic_0000001779917185_p24692740"></a><a name="zh-cn_topic_0000001779917185_p24692740"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001779917185_p53954942"><a name="zh-cn_topic_0000001779917185_p53954942"></a><a name="zh-cn_topic_0000001779917185_p53954942"></a>本rank所使用的stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001779917185_section663212844612"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193)：接口成功返回HCCL\_SUCCESS，其他失败。

## 约束说明<a name="zh-cn_topic_0000001779917185_section16632182812468"></a>

-   所有rank的recvCount、dataType、root均应相同。
-   全局只能有1个root节点。
-   非root节点的sendBuf可以为空。root节点的sendBuf不能为空。

## 调用示例<a name="zh-cn_topic_0000001779917185_section204039211474"></a>

```c
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t sendCount = 8;
uint64_t recvCount = 1;
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = recvCount * sizeof(float);

// 申请 Device 内存用于接收 Scatter 结果
ACLCHECK(aclrtMalloc(&recvBuf, recvCount, ACL_MEM_MALLOC_HUGE_FIRST));
// 在 root 节点，申请 Device 内存用于存放发送数据
if (device == rootRank) {
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_FIRST));
}

// 初始化通信域
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, device, &hcclComm);

// 创建任务流
aclrtStream stream;
aclrtCreateStream(&stream);

// 执行 Scatter，将通信域内 root 节点的数据均分并散布至其他 rank
HcclScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, rootRank, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(sendBuf);          // 释放 Device 侧内存
aclrtFree(recvBuf);          // 释放 Device 侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```

