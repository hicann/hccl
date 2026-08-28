# Mainstream Framework Integration

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T04:00:16.526Z pushedAt=2026-08-13T10:14:11.186Z -->

## Scenario Description

The position of HCCL in the system is shown in the following figure.

![Position of HCCL in the System](figures/hccl_location.png)

AI frameworks primarily have three programming execution modes: single-operator mode, graph mode (Ascend IR), and graph capture mode (aclgraph). Accordingly, HCCL provides corresponding working modes for each.

- In single-operator mode and graph capture mode (aclgraph), the AI framework directly calls the C language APIs of HCCL to dispatch communication operators to the acceleration engine for execution. For details about HCCL communication operator APIs, see [Communication Operators](../api_ref/comm_op_interface/README.md).

- In graph mode (Ascend IR), the AI framework uses Ascend operator IR to construct the computation process of a model into a graph, and dispatches communication operators in the graph to the acceleration engine for execution through Graph Engine (GE). For details about graph mode, see *[Graph Development Guide](https://hiascend.com/en/document/redirect/CannCommunityGraphGuide)*. For the definition of Ascend IR, see "Ascend IR Operator Specification" in *[Operator Library Interface Reference](https://hiascend.com/en/document/redirect/CannCommunityOplist)*.

For PyTorch and MindSpore frameworks, HCCL invocation has been integrated into the TorchNPU and MindSpore framework code. Developers can specify HCCL as the distributed backend and directly use the framework's native communication APIs to implement distributed capabilities. For detailed usage, see *[TorchNPU Product Documentation](https://www.hiascend.com/document/detail/en/Pytorch/latest/index/index.html)* and the [MindSpore official website](https://www.mindspore.cn/en).

For the TensorFlow framework, HCCL interfaces with TensorFlow through the TensorFlow adapter plugin TF Adapter. For detailed usage, see *[TensorFlow Model Migration Guide](https://hiascend.com/en/document/redirect/canntfmigr)*.

## Sample Code

- [PyTorch Framework Invocation](../../../examples/03_ai_framework/01_pytorch)

- [TensorFlow Framework Invocation](../../../examples/03_ai_framework/02_tensorflow)
