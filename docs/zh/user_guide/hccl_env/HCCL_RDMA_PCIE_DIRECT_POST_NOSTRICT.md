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

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品：不支持
<!-- end id5 -->
