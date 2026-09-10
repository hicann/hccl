# HCCL_HOST_RDMA_UDP_PORTS_LIST

## 功能描述

在Host网卡RoCE通信场景下，若开发者需要指定多QP使用的UDP源端口，可通过此环境变量按NPU物理设备ID配置UDP源端口列表。

该环境变量的配置格式如下：

```text
<phy_dev_id>:<src_port0>,<src_port1>,...,<src_portN>[;<phy_dev_id>:<src_port0>,<src_port1>,...,<src_portN>]
```

- `phy_dev_id`为NPU物理设备ID，配置为非负十进制整数。同一个NPU物理设备ID只能配置一次。
- `src_port`为UDP源端口，配置为十进制整数，取值范围为\[1,65535\]。\[1,1023\]为系统保留端口，应避免使用这些端口。每个NPU物理设备ID最多配置32个UDP源端口。
- 多个NPU物理设备的配置之间使用英文分号（`;`）分隔，同一个NPU物理设备的多个UDP源端口之间使用英文逗号（`,`）分隔，配置中不能包含空格。
- 环境变量值的长度不能超过32\*1024个字符。

HCCL根据当前使用的NPU物理设备ID选择对应的UDP源端口列表。如果QP数量大于UDP源端口数量，则从列表中的第一个端口开始循环使用；如果QP数量小于UDP源端口数量，则仅使用与QP数量相同的前几个端口。

该环境变量仅用于指定UDP源端口，不改变两个rank之间使用的QP数量。多QP数量可通过环境变量[HCCL_RDMA_QPS_PER_CONNECTION](HCCL_RDMA_QPS_PER_CONNECTION.md)配置。默认不配置此环境变量，此时系统按照其他配置或默认方式选择UDP源端口。

## 配置示例

```bash
export HCCL_HOST_RDMA_UDP_PORTS_LIST="0:10000,10015;1:10016,10031"
```

以上配置表示：NPU物理设备0使用UDP源端口10000和10015，NPU物理设备1使用UDP源端口10016和10031。

## 使用约束

- 该环境变量仅在Host网卡RoCE通信场景下生效。
- 环境变量中未配置当前使用的NPU物理设备ID时，本环境变量不指定UDP源端口。
- 本环境变量优先级高于环境变量[HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md)：已配置本环境变量时，使用本环境变量中的UDP源端口列表；否则，回退使用HCCL_RDMA_QP_PORT_CONFIG_PATH指定配置文件中的端口。
- 配置格式或取值非法时，通信域初始化失败。

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->
