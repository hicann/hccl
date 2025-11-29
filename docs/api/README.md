# 接口列表<a name="ZH-CN_TOPIC_0000002486851520"></a>

HCCL提供了C语言的通信算子接口，框架开发者可以调用这些接口进行单算子模式下的框架适配，实现分布式能力。

<a name="zh-cn_topic_0000001312721317_table1554562693420"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001312721317_row125461626123420"><th class="cellrowborder" valign="top" width="25.46%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001312721317_p8546826113410"><a name="zh-cn_topic_0000001312721317_p8546826113410"></a><a name="zh-cn_topic_0000001312721317_p8546826113410"></a><strong id="zh-cn_topic_0000001312721317_b225214248340"><a name="zh-cn_topic_0000001312721317_b225214248340"></a><a name="zh-cn_topic_0000001312721317_b225214248340"></a>接口</strong></p>
</th>
<th class="cellrowborder" valign="top" width="74.53999999999999%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001312721317_p10546122613347"><a name="zh-cn_topic_0000001312721317_p10546122613347"></a><a name="zh-cn_topic_0000001312721317_p10546122613347"></a><strong id="zh-cn_topic_0000001312721317_b132536244347"><a name="zh-cn_topic_0000001312721317_b132536244347"></a><a name="zh-cn_topic_0000001312721317_b132536244347"></a>简介</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001312721317_row967319459273"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p116731455279"><a name="zh-cn_topic_0000001312721317_p116731455279"></a><a name="zh-cn_topic_0000001312721317_p116731455279"></a><a href="./context/HcclAllReduce.md">HcclAllReduce</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p11287144214307"><a name="zh-cn_topic_0000001312721317_p11287144214307"></a><a name="zh-cn_topic_0000001312721317_p11287144214307"></a>集合通信算子AllReduce的操作接口，将通信域内所有节点的输入数据进行相加（或其他归约操作）后，再把结果发送到所有节点的输出buffer，其中归约操作类型由op参数指定。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row48931643182715"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p17893204382717"><a name="zh-cn_topic_0000001312721317_p17893204382717"></a><a name="zh-cn_topic_0000001312721317_p17893204382717"></a><a href="./context/HcclBroadcast.md">HcclBroadcast</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p10286104283019"><a name="zh-cn_topic_0000001312721317_p10286104283019"></a><a name="zh-cn_topic_0000001312721317_p10286104283019"></a>集合通信算子Broadcast的操作接口，将通信域内root节点的数据广播到其他rank。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row498016411272"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p098016412276"><a name="zh-cn_topic_0000001312721317_p098016412276"></a><a name="zh-cn_topic_0000001312721317_p098016412276"></a><a href="./context/HcclAllGather.md">HcclAllGather</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p52852426305"><a name="zh-cn_topic_0000001312721317_p52852426305"></a><a name="zh-cn_topic_0000001312721317_p52852426305"></a>集合通信算子AllGather的操作接口，将通信域内所有节点的输入按照rank id重新排序，然后拼接起来，再将结果发送到所有节点的输出。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row1030115031119"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p23013071117"><a name="zh-cn_topic_0000001312721317_p23013071117"></a><a name="zh-cn_topic_0000001312721317_p23013071117"></a><a href="./context/HcclAllGatherV.md">HcclAllGatherV</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p680714122118"><a name="zh-cn_topic_0000001312721317_p680714122118"></a><a name="zh-cn_topic_0000001312721317_p680714122118"></a>集合通信算子AllGatherV的操作接口，将通信域内所有节点的输入按照rank id重新排序，然后拼接起来，再将结果发送到所有节点的输出。</p>
<p id="zh-cn_topic_0000001312721317_p5604103112110"><a name="zh-cn_topic_0000001312721317_p5604103112110"></a><a name="zh-cn_topic_0000001312721317_p5604103112110"></a>与AllGather算子不同的是，AllGatherV算子支持通信域内不同节点的输入配置不同大小的数据量。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row1569917547284"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p10178125714617"><a name="zh-cn_topic_0000001312721317_p10178125714617"></a><a name="zh-cn_topic_0000001312721317_p10178125714617"></a><a href="./context/HcclReduceScatter.md">HcclReduceScatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p22858424300"><a name="zh-cn_topic_0000001312721317_p22858424300"></a><a name="zh-cn_topic_0000001312721317_p22858424300"></a>集合通信算子ReduceScatter的操作接口，将所有rank的输入相加（或其他归约操作）后，再把结果按照rank编号均匀分散到各个rank的输出buffer，每个进程拿到其他进程1/rank_size份的数据进行归约操作。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row198415527106"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p484125214102"><a name="zh-cn_topic_0000001312721317_p484125214102"></a><a name="zh-cn_topic_0000001312721317_p484125214102"></a><a href="./context/HcclReduceScatterV.md">HcclReduceScatterV</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p40902273"><a name="zh-cn_topic_0000001312721317_p40902273"></a><a name="zh-cn_topic_0000001312721317_p40902273"></a>集合通信算子ReduceScatterV的操作接口，将所有rank的输入相加（或其他归约操作）后，再把结果按照rank编号分散到各个rank的输出buffer，每个进程拿到其他进程对应rank编号的数据进行归约操作。</p>
<p id="zh-cn_topic_0000001312721317_p147680373120"><a name="zh-cn_topic_0000001312721317_p147680373120"></a><a name="zh-cn_topic_0000001312721317_p147680373120"></a>与ReduceScatter算子不同的是，ReduceScatterV算子支持为通信域内不同的节点配置不同大小的数据量。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row075325292818"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p616718244304"><a name="zh-cn_topic_0000001312721317_p616718244304"></a><a name="zh-cn_topic_0000001312721317_p616718244304"></a><a href="./context/HcclReduce.md">HcclReduce</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p class="msonormal" id="zh-cn_topic_0000001312721317_p2284124216302"><a name="zh-cn_topic_0000001312721317_p2284124216302"></a><a name="zh-cn_topic_0000001312721317_p2284124216302"></a>集合通信算子Reduce的操作接口，将所有rank的数据相加（或其他归约操作）后，再把结果发送到root节点的指定位置上。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row159481335268"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p775018350304"><a name="zh-cn_topic_0000001312721317_p775018350304"></a><a name="zh-cn_topic_0000001312721317_p775018350304"></a><a href="./context/HcclAlltoAll.md">HcclAlltoAll</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p1283154273017"><a name="zh-cn_topic_0000001312721317_p1283154273017"></a><a name="zh-cn_topic_0000001312721317_p1283154273017"></a>集合通信算子AlltoAll操作接口，向通信域内所有rank发送相同数据量的数据，并从所有rank接收相同数据量的数据。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row1387718321301"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p647573913015"><a name="zh-cn_topic_0000001312721317_p647573913015"></a><a name="zh-cn_topic_0000001312721317_p647573913015"></a><a href="./context/HcclAlltoAllV.md">HcclAlltoAllV</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p12284742143010"><a name="zh-cn_topic_0000001312721317_p12284742143010"></a><a name="zh-cn_topic_0000001312721317_p12284742143010"></a>集合通信算子AlltoAllV操作接口，向通信域内所有rank发送数据（数据量可以定制），并从所有rank接收数据。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row178091240103313"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p1580912402331"><a name="zh-cn_topic_0000001312721317_p1580912402331"></a><a name="zh-cn_topic_0000001312721317_p1580912402331"></a><a href="./context/HcclAlltoAllVC.md">HcclAlltoAllVC</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p88091440193317"><a name="zh-cn_topic_0000001312721317_p88091440193317"></a><a name="zh-cn_topic_0000001312721317_p88091440193317"></a>集合通信算子AlltoAllV<span id="zh-cn_topic_0000002417514657_ph20748105914110"><a name="zh-cn_topic_0000002417514657_ph20748105914110"></a><a name="zh-cn_topic_0000002417514657_ph20748105914110"></a>C</span>操作接口，向通信域内所有rank发送数据（数据量可以定制），并从所有rank接收数据。相比于AlltoAllV，AlltoAllV<span id="zh-cn_topic_0000002417514657_ph1509131064715"><a name="zh-cn_topic_0000002417514657_ph1509131064715"></a><a name="zh-cn_topic_0000002417514657_ph1509131064715"></a>C</span>通过输入参数sendCountMatrix传入所有rank的收发参数。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row2913102219481"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p645175273118"><a name="zh-cn_topic_0000001312721317_p645175273118"></a><a name="zh-cn_topic_0000001312721317_p645175273118"></a><a href="./context/HcclBarrier.md">HcclBarrier</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p791372217484"><a name="zh-cn_topic_0000001312721317_p791372217484"></a><a name="zh-cn_topic_0000001312721317_p791372217484"></a>将指定通信域内所有rank的stream阻塞，直到所有rank都下发执行该操作为止。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row384384553515"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p11538195022820"><a name="zh-cn_topic_0000001312721317_p11538195022820"></a><a name="zh-cn_topic_0000001312721317_p11538195022820"></a><a href="./context/HcclScatter.md">HcclScatter</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p1584416459354"><a name="zh-cn_topic_0000001312721317_p1584416459354"></a><a name="zh-cn_topic_0000001312721317_p1584416459354"></a>集合通信算子Scatter操作接口，将root节点的数据均分并散布至其他rank。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row14675121613328"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p167651663216"><a name="zh-cn_topic_0000001312721317_p167651663216"></a><a name="zh-cn_topic_0000001312721317_p167651663216"></a><a href="./context/HcclSend.md">HcclSend</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p1367731611321"><a name="zh-cn_topic_0000001312721317_p1367731611321"></a><a name="zh-cn_topic_0000001312721317_p1367731611321"></a>点对点通信Send操作接口，将当前节点指定位置的数据发送至目的节点的指定位置上。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row119221338194819"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p3999134233116"><a name="zh-cn_topic_0000001312721317_p3999134233116"></a><a name="zh-cn_topic_0000001312721317_p3999134233116"></a><a href="./context/HcclRecv.md">HcclRecv</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p9923133818487"><a name="zh-cn_topic_0000001312721317_p9923133818487"></a><a name="zh-cn_topic_0000001312721317_p9923133818487"></a>点对点通信Receive操作接口，从源节点接收数据到当前节点的指定位置上。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001312721317_row6327155483515"><td class="cellrowborder" valign="top" width="25.46%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001312721317_p7327165413359"><a name="zh-cn_topic_0000001312721317_p7327165413359"></a><a name="zh-cn_topic_0000001312721317_p7327165413359"></a><a href="./context/HcclBatchSendRecv.md">HcclBatchSendRecv</a></p>
</td>
<td class="cellrowborder" valign="top" width="74.53999999999999%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001312721317_p732712548358"><a name="zh-cn_topic_0000001312721317_p732712548358"></a><a name="zh-cn_topic_0000001312721317_p732712548358"></a>异步批量点对点通信操作接口，调用一次接口可以完成本rank上的多个收发任务，本rank发送和接收之间是异步的，发送和接收任务之间不会相互阻塞。</p>
</td>
</tr>
</tbody>
</table>

