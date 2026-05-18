# HCCL_RDMA_QP_PORT_CONFIG_PATH

## 功能描述

两个rank之间RDMA通信时会默认创建1个QP（Queue Pair）进行数据传输，若开发者想让两个rank之间的RDMA通信使用多个QP，并指定多QP通信时使用的源端口号，可通过此环境变量实现。

开发者可通过此环境变量指定<srcIP,dstIP\>与端口映射关系配置文件的存储路径。当<srcIP,dstIP\>配置多个端口号时，系统将开启多QP通信，且所配置的端口号即为每个QP使用的源端口。

该环境变量配置示例如下：

```bash
export HCCL_RDMA_QP_PORT_CONFIG_PATH=/home/tmp
```

其中“/home/tmp”为<srcIP,dstIP\>与端口映射关系配置文件“MultiQpSrcPort.cfg”的存储路径，支持配置为绝对路径或相对路径，该路径最大长度需要小于等于4096个字符。

“MultiQpSrcPort.cfg”文件需要用户自定义（注意文件命名需要保持为“MultiQpSrcPort.cfg”），配置格式如下：

```text
srcIP1,dstIP1=srcPort0,srcPort1,...,srcPortN
srcIPN,dstIPN=srcPort0,srcPort1,...,srcPortN
```

- 该文件支持的最大配置行数为128\*1024=131072。
- 每个<srcIP,dstIP\>地址对最多支持配置32个端口，但建议不超过8个端口。QP个数超过8时，无法确保性能收益，且可能导致内存占用过多从而引发业务运行失败。
- 每个<srcIP,dstIP\>地址对在该文件中仅允许出现一次。
- srcIP、dstIP需要为常规IPv4格式和IPv6格式。
- srcIP、dstIP支持配置为“0.0.0.0”，代表所有IP地址。

“MultiQpSrcPort.cfg”文件配置示例如下：

```text
192.168.100.2,192.168.100.3=61100,61101,61102
192.168.100.4,192.168.100.5=61100,61101,61102,61104
0.0.0.0,192.168.100.122=65515,65516,65513
```

## 配置示例

```bash
export HCCL_RDMA_QP_PORT_CONFIG_PATH=/home/tmp
```

## 使用约束

- 该环境变量仅支持单算子调用方式，不支持静态图模式。
- 该环境变量的优先级高于环境变量[HCCL_RDMA_QPS_PER_CONNECTION](HCCL_RDMA_QPS_PER_CONNECTION.md)，此环境变量配置后，两个rank间通信时使用的QP个数以“MultiQpSrcPort.cfg”文件中配置的源端口号个数为准。
- QP相关配置的优先级如下：

    管理面多QP配置（通过hccn_tool工具的“-s multi_qp”参数配置） \> NSLB的QP配置（通过hccn_tool工具的"-t nslb-dp"参数配置）\> 环境变量HCCL_RDMA_QP_PORT_CONFIG_PATH \>  环境变量HCCL_RDMA_QPS_PER_CONNECTION。

## 支持的型号

Atlas A2 训练系列产品/Atlas A2 推理系列产品（针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，仅支持Atlas 800T A2 训练服务器、Atlas 900 A2 PoD 集群基础单元、Atlas 200T A2 Box16 异构子框。）

Atlas A3 训练系列产品/Atlas A3 推理系列产品
