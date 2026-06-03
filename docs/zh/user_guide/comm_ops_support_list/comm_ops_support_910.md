# Atlas 训练系列产品

本节提供Atlas 训练系列产品的通信算子支持情况。

- 单算子零拷贝：为了降低内存拷贝开销，使得HCCL可以直接对业务传入的内存进行操作，提升通信性能。
- 通信算子重执行：网络故障导致通信闪断时，HCCL会尝试重新执行此通信算子，提升通信稳定性。
- 确定性计算：归约类通信算子在相同的硬件和输入下，多次执行将产生相同的输出。

> [!NOTE]说明
>
> - Atlas 训练系列产品，通信算子仅支持HOST展开。
> - 本节表格中“√”代表支持，“×”代表不支持，“NA”代表不涉及，Atlas 训练系列产品不支持单算子零拷贝与重执行。
> - 未列出的算子与网络运行模式代表不支持。

<table><thead align="left"><tr><th><p>算子</p>
</th>
<th><p>网络运行模式</p>
</th>
<th><p>单算子零拷贝</p>
</th>
<th><p>确定性计算</p>
</th>
<th><p>重执行</p>
</th>
<th><p>节点内通信</p>
</th>
<th><p>节点间通信</p>
</th>
</tr>
</thead>
<tbody><tr><td rowspan="2"><p>Broadcast</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="2"><p>AllGather</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="2"><p>Reduce</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="2"><p>AllReduce</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="2"><p>Scatter</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="2"><p>ReduceScatter</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="2"><p>AlltoAll</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="2"><p>AlltoAllV</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="2"><p>Send</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="2"><p>Recv</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td rowspan="2"><p>BatchSendRecv</p>
</td>
<td><p>单算子模式</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
<tr><td><p>图模式Ascend IR</p>
</td>
<td><p>×</p>
</td>
<td><p>NA</p>
</td>
<td><p>×</p>
</td>
<td><p>√</p>
</td>
<td><p>√</p>
</td>
</tr>
</tbody>
</table>
