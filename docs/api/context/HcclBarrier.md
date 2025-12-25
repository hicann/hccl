# HcclBarrier<a name="ZH-CN_TOPIC_0000002486832344"></a>

## 产品支持情况<a name="zh-cn_topic_0000001312481233_section10594071513"></a>

<a name="zh-cn_topic_0000001312481233_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312481233_row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001312481233_p1883113061818"><a name="zh-cn_topic_0000001312481233_p1883113061818"></a><a name="zh-cn_topic_0000001312481233_p1883113061818"></a><span id="zh-cn_topic_0000001312481233_ph20833205312295"><a name="zh-cn_topic_0000001312481233_ph20833205312295"></a><a name="zh-cn_topic_0000001312481233_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="zh-cn_topic_0000001312481233_p783113012187"><a name="zh-cn_topic_0000001312481233_p783113012187"></a><a name="zh-cn_topic_0000001312481233_p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312481233_row220181016240"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312481233_p48327011813"><a name="zh-cn_topic_0000001312481233_p48327011813"></a><a name="zh-cn_topic_0000001312481233_p48327011813"></a><span id="zh-cn_topic_0000001312481233_ph583230201815"><a name="zh-cn_topic_0000001312481233_ph583230201815"></a><a name="zh-cn_topic_0000001312481233_ph583230201815"></a><term id="zh-cn_topic_0000001312481233_zh-cn_topic_0000001312391781_term1253731311225"><a name="zh-cn_topic_0000001312481233_zh-cn_topic_0000001312391781_term1253731311225"></a><a name="zh-cn_topic_0000001312481233_zh-cn_topic_0000001312391781_term1253731311225"></a>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312481233_p7948163910184"><a name="zh-cn_topic_0000001312481233_p7948163910184"></a><a name="zh-cn_topic_0000001312481233_p7948163910184"></a>√</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312481233_row173226882415"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001312481233_p14832120181815"><a name="zh-cn_topic_0000001312481233_p14832120181815"></a><a name="zh-cn_topic_0000001312481233_p14832120181815"></a><span id="zh-cn_topic_0000001312481233_ph1292674871116"><a name="zh-cn_topic_0000001312481233_ph1292674871116"></a><a name="zh-cn_topic_0000001312481233_ph1292674871116"></a><term id="zh-cn_topic_0000001312481233_zh-cn_topic_0000001312391781_term11962195213215"><a name="zh-cn_topic_0000001312481233_zh-cn_topic_0000001312391781_term11962195213215"></a><a name="zh-cn_topic_0000001312481233_zh-cn_topic_0000001312391781_term11962195213215"></a>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term></span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="zh-cn_topic_0000001312481233_p19948143911820"><a name="zh-cn_topic_0000001312481233_p19948143911820"></a><a name="zh-cn_topic_0000001312481233_p19948143911820"></a>√</p>
</td>
</tr>
</tbody>
</table>

> [!NOTE]说明 
> 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。

## 功能说明<a name="zh-cn_topic_0000001312481233_section662317284469"></a>

将指定通信域内所有rank的stream阻塞，直到所有rank都下发执行该操作为止。

## 函数原型<a name="zh-cn_topic_0000001312481233_section13621172844611"></a>

```
HcclResult HcclBarrier(HcclComm comm, aclrtStream stream)
```

## 参数说明<a name="zh-cn_topic_0000001312481233_section1462482884619"></a>

<a name="zh-cn_topic_0000001312481233_table176338284461"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312481233_row1970492813463"><th class="cellrowborder" valign="top" width="20.200000000000003%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000001312481233_p177042287466"><a name="zh-cn_topic_0000001312481233_p177042287466"></a><a name="zh-cn_topic_0000001312481233_p177042287466"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="17.169999999999998%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000001312481233_p13704112813461"><a name="zh-cn_topic_0000001312481233_p13704112813461"></a><a name="zh-cn_topic_0000001312481233_p13704112813461"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="62.629999999999995%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000001312481233_p14704192813467"><a name="zh-cn_topic_0000001312481233_p14704192813467"></a><a name="zh-cn_topic_0000001312481233_p14704192813467"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312481233_row57041728184620"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312481233_p97051928174613"><a name="zh-cn_topic_0000001312481233_p97051928174613"></a><a name="zh-cn_topic_0000001312481233_p97051928174613"></a>comm</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312481233_p107056288461"><a name="zh-cn_topic_0000001312481233_p107056288461"></a><a name="zh-cn_topic_0000001312481233_p107056288461"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312481233_p7705182819463"><a name="zh-cn_topic_0000001312481233_p7705182819463"></a><a name="zh-cn_topic_0000001312481233_p7705182819463"></a>集合通信操作所在的通信域。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312481233_row19705628124613"><td class="cellrowborder" valign="top" width="20.200000000000003%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000001312481233_p11705182844619"><a name="zh-cn_topic_0000001312481233_p11705182844619"></a><a name="zh-cn_topic_0000001312481233_p11705182844619"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" width="17.169999999999998%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000001312481233_p19705162814462"><a name="zh-cn_topic_0000001312481233_p19705162814462"></a><a name="zh-cn_topic_0000001312481233_p19705162814462"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="62.629999999999995%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000001312481233_p670512874611"><a name="zh-cn_topic_0000001312481233_p670512874611"></a><a name="zh-cn_topic_0000001312481233_p670512874611"></a>本rank所使用的stream。</p>
</td>
</tr>
</tbody>
</table>

## 返回值<a name="zh-cn_topic_0000001312481233_section663212844612"></a>

[HcclResult](HcclResult.md#ZH-CN_TOPIC_0000002519072193)：接口成功返回HCCL\_SUCCESS，其他失败。

## 约束说明<a name="zh-cn_topic_0000001312481233_section16632182812468"></a>

无

