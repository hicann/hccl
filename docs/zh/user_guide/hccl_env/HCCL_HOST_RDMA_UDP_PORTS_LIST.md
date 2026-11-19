# HCCL_HOST_RDMA_UDP_PORTS_LIST

## 功能描述

使用Host侧RoCE网卡时，可通过此环境变量按NPU物理设备ID配置UDP源端口列表，指定多QP使用的UDP源端口。

此外，用户也可通过手工创建`/etc/hcomm.cfg`文件来配置QP数量及UDP源端口列表。

该环境变量的配置格式如下：

```text
<phy_dev_id>:<src_port0>,<src_port1>,...,<src_portN>[;<phy_dev_id>:<src_port0>,<src_port1>,...,<src_portN>]
```

- `phy_dev_id`为NPU物理设备ID，配置为非负十进制整数。同一个NPU物理设备ID只能配置一次。
- `src_port`为UDP源端口，配置为十进制整数，取值范围为\[1,65535\]。\[1,1023\]为系统保留端口，应避免使用这些端口。每个NPU物理设备ID最多配置32个UDP源端口。
- 多个NPU物理设备的配置之间使用英文分号（`;`）分隔，同一个NPU物理设备的多个UDP源端口之间使用英文逗号（`,`）分隔，配置中不能包含空格。
- 环境变量值的长度不能超过32\*1024个字符。

HCCL根据当前使用的NPU物理设备ID选择对应的UDP源端口列表。如果QP数量大于UDP源端口数量，则从列表中的第一个端口开始循环使用；如果QP数量小于UDP源端口数量，则仅使用与QP数量相同的前几个端口。

需要同时指定QP数量和UDP源端口时，可将[HCCL_RDMA_QPS_PER_CONNECTION](HCCL_RDMA_QPS_PER_CONNECTION.md)与本环境变量配套使用。前者配置使用的QP数量，本环境变量指定UDP源端口列表。默认不配置本环境变量，此时系统按照其他配置或默认方式选择UDP源端口。

## 配置示例

```bash
export HCCL_RDMA_QPS_PER_CONNECTION=2
export HCCL_HOST_RDMA_UDP_PORTS_LIST="0:10000,10015;1:10016,10031"
```

使用2个QP，NPU物理设备0使用UDP源端口10000和10015，NPU物理设备1使用UDP源端口10016和10031。

## 使用约束

- 该环境变量仅在使用Host侧RoCE网卡时生效。
- 仅通过`HCCL_HOST_RDMA_UDP_PORTS_LIST`为当前NPU物理设备配置1个UDP源端口，未设置`HCCL_RDMA_QPS_PER_CONNECTION`且无其他QP数量配置生效时，创建的QP数量以Host侧RoCE网卡支持的总线路径数量为准。若设置了`HCCL_RDMA_QPS_PER_CONNECTION`，或`HCCL_RDMA_QP_PORT_CONFIG_PATH`指定的文件中有匹配当前IP对的配置，则按[配置文件方式](#配置文件方式)中的优先级确定QP数量。
- 环境变量中未配置当前使用的NPU物理设备ID时，本环境变量不指定UDP源端口。
- 本环境变量的优先级低于`/etc/hcomm.cfg`中的UDP源端口配置，高于环境变量[HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md)。当`/etc/hcomm.cfg`未提供当前使用的NPU物理设备的配置时，优先使用本环境变量中匹配该设备的UDP源端口列表；本环境变量未提供该设备的配置时，继续使用HCCL_RDMA_QP_PORT_CONFIG_PATH中匹配当前IP对的配置或默认方式。
- 配置格式或取值非法时，通信域初始化失败。

## 配置文件方式

使用Host侧RoCE网卡时，可通过`/etc/hcomm.cfg`按NPU物理设备ID配置QP（Queue Pair）数量和UDP源端口列表，使RDMA通信使用多个QP。

HCCL根据当前使用的NPU物理设备ID选择对应配置。同一物理设备的配置适用于该设备通过Host侧RoCE网卡进行的通信。

### 配置格式

配置文件的路径固定为`/etc/hcomm.cfg`。每个NPU物理设备的配置格式如下：

```text
udp_port_mode_<phy_dev_id>=multi_qp
multi_qp_count_<phy_dev_id>=<qp_count>
multi_qp_udp_ports_<phy_dev_id>=<src_port0>,<src_port1>,...,<src_portN>
```

- `phy_dev_id`为NPU物理设备ID，配置为非负十进制整数，取值范围为\[0,4294967295\]。除设备ID为0外，不能包含前导零，例如设备ID为1时不能写为`01`。同一设备的各配置项使用相同的物理设备ID。
- `udp_port_mode`为UDP端口配置模式，目前只支持配置为`multi_qp`。
- `qp_count`为QP数量，配置为十进制整数，取值范围为\[1,32\]。当`qp_count`为1时，创建的QP数量以Host侧RoCE网卡支持的总线路径数量为准。
- `src_port`为UDP源端口，配置为十进制整数，取值范围为\[1,65535\]。\[1,1023\]为系统保留端口，应避免使用这些端口。`multi_qp_udp_ports`的值可以为空。填写端口时，每个NPU物理设备可指定1至32个UDP源端口，端口之间使用英文逗号（`,`）分隔，列表内部不能包含空格。
- 每个配置项单独占一行，空行和以`#`开头的注释行会被忽略，不支持在配置项取值后添加行内注释。

UDP源端口列表非空时，如果QP数量大于端口数量，则从列表中的第一个端口开始循环使用。如果QP数量小于端口数量，则仅使用与QP数量相同的前几个端口。

### 配置示例

在`/etc/hcomm.cfg`中添加以下内容：

```text
# NPU physical device 0
udp_port_mode_0=multi_qp
multi_qp_count_0=4
multi_qp_udp_ports_0=10000,10015,10030,10045

# NPU physical device 1
udp_port_mode_1=multi_qp
multi_qp_count_1=1
multi_qp_udp_ports_1=
```

NPU物理设备0使用4个QP，UDP源端口依次为10000、10015、10030和10045。NPU物理设备1按Host侧RoCE网卡支持的总线路径数量创建QP。

### 使用约束

- 本配置仅在使用Host侧RoCE网卡时生效。
- 每个配置的NPU物理设备必须同时提供`udp_port_mode`、`multi_qp_count`和`multi_qp_udp_ports`。端口列表留空时，仍须保留`multi_qp_udp_ports_<phy_dev_id>=`这一行。
- 任一设备缺少必配项、物理设备ID格式非法或取值非法时，通信域初始化失败，不会回退到其他配置方式。重复配置同一配置项时，使用第一次出现的取值。
- 文件不存在或无法打开时，不使用文件配置，继续使用其他配置或默认方式。文件中未配置当前使用的NPU物理设备ID时，该设备继续使用其他配置或默认方式。
- 使用Host侧RoCE网卡时，QP数量和UDP源端口列表的选择顺序如下。表中优先级由高到低，仅使用匹配当前设备或IP对的配置。

    | 配置内容 | 选择顺序 |
    | --- | --- |
    | QP数量 | 通信通道显式指定的QP数量 > `/etc/hcomm.cfg`中的`multi_qp_count` > [HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md)指定文件中的端口数量 > [HCCL_RDMA_QPS_PER_CONNECTION](HCCL_RDMA_QPS_PER_CONNECTION.md)（默认值为1）。按此顺序选出的值为1时，创建的QP数量以Host侧RoCE网卡支持的总线路径数量为准。 |
    | UDP源端口列表 | 当前设备的端口列表 > [HCCL_RDMA_QP_PORT_CONFIG_PATH](HCCL_RDMA_QP_PORT_CONFIG_PATH.md)指定文件中匹配当前IP对的端口列表 > 默认方式（由驱动分配UDP源端口号） |

- 配置在进程内首次初始化配置管理器时读取，成功后缓存。修改文件后需重新启动通信业务进程，使新配置生效。

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas推理系列产品：不支持
<!-- end id5 -->
