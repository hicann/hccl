# HCCL_RDMA_QPS_PER_CONNECTION

## 功能描述

两个rank之间RDMA通信时会默认创建1个QP（Queue Pair）进行数据传输，若开发者想让两个rank之间的RDMA通信使用多个QP，可通过此环境变量实现。

此环境变量代表两个rank间需要使用的QP个数，需要配置为整数，取值范围：\[1,32\]，建议配置范围：\[1,8\]，QP个数超过8时无法确保性能收益，还可能会造成由于内存占用过多导致业务运行失败的情况。默认值：1。

假设HCCL_RDMA_QPS_PER_CONNECTION环境变量配置为N1，则会在每两个rank之间创建N1个QP，两个rank之间通过RDMA传递的业务数据会平均分配到N1个QP上并行收发。

开启多QP传输的功能后，开发者可通过环境变量[HCCL_MULTI_QP_THRESHOLD](HCCL_MULTI_QP_THRESHOLD.md)设置每个QP分担数据量的最小阈值；若开发者想指定每个QP使用的源端口号，可通过环境变量[HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md)实现。

## 配置示例

```bash
export HCCL_RDMA_QPS_PER_CONNECTION=4
```

## 使用约束

- 该环境变量仅支持单算子调用方式，不支持静态图模式。
- QP相关配置的优先级如下：

    管理面多QP配置（通过hccn_tool工具的`-s multi_qp`参数配置） > NSLB的QP配置（通过hccn_tool工具的`-t nslb-dp`参数配置）> 环境变量HCCL_RDMA_QP_PORT_CONFIG_PATH >  环境变量HCCL_RDMA_QPS_PER_CONNECTION。

## 支持的型号

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

Atlas A3 训练系列产品/Atlas A3 推理系列产品
