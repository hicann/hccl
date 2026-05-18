# HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT

## 功能描述

多机通信且Host操作系统小页内存页表大小非4KB的场景，当通信算子下发性能Host Bound时，开发者可设置此环境变量，通过PCIe Direct的方式提交RDMA任务，提升通信算子下发性能。

此环境变量支持如下取值：

- TRUE：代表通过PCIe Direct（Host与Device之间的高速通信接口）的方式提交RDMA任务。
- FALSE（默认值）：代表通过HDC（Host Device Communication，主机设备通信）的方式提交RDMA任务。

此环境变量仅对Host侧小页内存页表大小非4KB的场景生效，若Host侧小页内存页表大小是4KB，无论此环境变量取值如何，都采用PCIe Direct的方式提交RDMA任务。

注意：

- 此环境变量设置为TRUE时，会额外消耗Device侧的大页内存（每个通信链路会额外多占用1MB的大页内存）。
- 如果开发者既想通过此环境变量提升通信算子下发性能，又想节省Device侧大页内存占用，可通过[HCCL_ALGO](HCCL_ALGO.md)环境变量将server间通信算法设置为ring，以控制通信链路数量。

  ```bash
  export HCCL_ALGO="level0:NA;level1:ring"
  ```

## 配置示例

```bash
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
```

## 使用约束

使用此环境变量时，需要满足[功能描述](#功能描述)中所述场景，即：

- 多机通信场景。
- Host操作系统小页内存页表大小非4KB。

## 支持的型号

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）
