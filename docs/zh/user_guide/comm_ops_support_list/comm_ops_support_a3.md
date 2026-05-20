# Atlas A3 训练系列产品/Atlas A3 推理系列产品

本节提供Atlas A3 训练系列产品/Atlas A3 推理系列产品的通信算子支持情况。

- 单算子零拷贝：为了降低内存拷贝开销，使得HCCL可以直接对业务传入的内存进行操作，提升通信性能。
- 通信算子重执行：网络故障导致通信闪断时，HCCL会尝试重新执行此通信算子，提升通信稳定性。
- 确定性计算：归约类通信算子在相同的硬件和输入下，多次执行将产生相同的输出。

> [!NOTE]说明
>
> - 下面按照通信算子的展开模式进行通信算子支持情况的呈现，未列出的展开模式代表不支持。
> - 本节表格中“√”代表支持，“×”代表不支持，“NA”代表不涉及。
> - 未列出的算子与网络运行模式代表不支持。

## AI CPU

| 算子 | 网络运行模式 | 单算子零拷贝 | 确定性计算 | 重执行 | 节点内通信 | 超节点内通信 | 超节点间通信 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Broadcast | 单算子模式 | √ | NA | √ | √ | √ | √ |
| 图模式Ascend IR | NA | NA | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | NA | √ | √ | √ | √ |  |
| AllGather | 单算子模式 | √ | NA | √ | √ | √ | √ |
| 图模式Ascend IR | NA | NA | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | NA | √ | √ | √ | √ |  |
| AllGatherV | 单算子模式 | × | NA | √ | √ | √ | √ |
| 图模式Ascend IR | NA | NA | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | NA | √ | √ | √ | √ |  |
| Reduce | 单算子模式 | × | √ | √ | √ | √ | √ |
| 图模式Ascend IR | NA | √ | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | √ | √ | √ | √ | √ |  |
| AllReduce | 单算子模式 | √ | √ | √ | √ | √ | √ |
| 图模式Ascend IR | NA | √ | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | √ | √ | √ | √ | √ |  |
| Scatter | 单算子模式 | × | NA | √ | √ | √ | √ |
| 图捕获模式aclgraph | NA | NA | √ | √ | √ | √ |  |
| ReduceScatter | 单算子模式 | √ | √ | √ | √ | √ | √ |
| 图模式Ascend IR | NA | √ | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | √ | √ | √ | √ | √ |  |
| ReduceScatterV | 单算子模式 | × | √ | √ | √ | √ | √ |
| 图模式Ascend IR | NA | √ | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | √ | √ | √ | √ | √ |  |
| AlltoAll | 单算子模式 | × | NA | √ | √ | √ | √ |
| 图模式Ascend IR | NA | NA | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | NA | √ | √ | √ | √ |  |
| AlltoAllV | 单算子模式 | × | NA | √ | √ | √ | √ |
| 图模式Ascend IR | NA | NA | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | NA | √ | √ | √ | √ |  |
| AlltoAllVC | 单算子模式 | × | NA | √ | √ | √ | √ |
| 图模式Ascend IR | NA | NA | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | NA | √ | √ | √ | √ |  |
| Send | 单算子模式 | × | NA | √ | √ | √ | √ |
| 图模式Ascend IR | NA | NA | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | NA | √ | √ | √ | √ |  |
| Recv | 单算子模式 | × | NA | √ | √ | √ | √ |
| 图模式Ascend IR | NA | NA | √ | √ | √ | √ |  |
| 图捕获模式aclgraph | NA | NA | √ | √ | √ | √ |  |
| BatchSendRecv | 单算子模式 | × | NA | √ | √ | √ | √ |
| 图捕获模式aclgraph | NA | NA | √ | √ | √ | √ |  |

## AIV

| 算子 | 网络运行模式 | 单算子零拷贝 | 确定性计算 | 重执行 | 节点内通信 | 超节点内通信 | 超节点间通信 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Broadcast | 单算子模式 | × | NA | × | √ | × | × |
| 图模式Ascend IR | × | NA | × | √ | × | × |  |
| 图捕获模式aclgraph | × | NA | × | × | × | × |  |
| AllGather | 单算子模式 | × | NA | × | √ | √ | × |
| 图模式Ascend IR | × | NA | × | √ | √ | × |  |
| 图捕获模式aclgraph | × | NA | × | √ | √ | × |  |
| AllReduce | 单算子模式 | × | √ | × | √ | √ | × |
| 图模式Ascend IR | × | √ | × | √ | √ | × |  |
| 图捕获模式aclgraph | × | √ | × | √ | √ | × |  |
| ReduceScatter | 单算子模式 | × | √ | × | √ | √ | × |
| 图模式Ascend IR | × | √ | × | √ | √ | × |  |
| 图捕获模式aclgraph | × | √ | × | √ | √ | × |  |
| AlltoAll | 单算子模式 | × | NA | × | √ | √ | × |
| 图模式Ascend IR | × | NA | × | √ | √ | × |  |
| 图捕获模式aclgraph | × | NA | × | √ | √ | × |  |
| AlltoAllV | 单算子模式 | × | NA | × | √ | √ | × |
| 图模式Ascend IR | × | NA | × | √ | √ | × |  |
| 图捕获模式aclgraph | × | NA | × | √ | √ | × |  |
| AlltoAllVC | 单算子模式 | × | NA | × | √ | √ | × |
| 图模式Ascend IR | × | NA | × | √ | √ | × |  |
| 图捕获模式aclgraph | × | NA | × | √ | √ | × |  |
| AllGather | 单算子模式 | × | NA | × | √ | √ | × |
| 图模式Ascend IR | × | NA | × | √ | √ | × |  |
| 图捕获模式aclgraph | × | NA | × | √ | √ | × |  |
| ReduceScatterV | 单算子模式 | × | NA | × | √ | √ | × |
| 图模式Ascend IR | × | NA | × | √ | √ | × |  |
| 图捕获模式aclgraph | × | NA | × | √ | √ | × |  |

> [!NOTE]说明
> AIV模式在小数据量通信时性能较优，主要用于推理场景，此模式下：
>
> - 单算子零拷贝会引入执行时内存协商，导致通信时延变大，所以当前AIV模式下不支持单算子零拷贝。
> - 重执行特性会增加执行耗时，所以AIV模式不支持算子重执行。
> - 仅支持超节点内通信，不支持超节点间通信。
