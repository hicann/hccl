# rank table配置资源信息（Ascend 950PR/Ascend 950DT）

对Ascend 950PR/Ascend 950DT，rank table文件需要配合拓扑文件初始化HCCL通信域。

> [!NOTE]说明
>
> - rank table文件大小最大支持1GB。
> - rank table文件为JSON格式，本节所示JSON文件示例中的注释仅为方便理解，实际使用时，请删除JSON文件中的注释。

## 拓扑文件配置

拓扑文件配置了物理拓扑信息和路由信息，存储路径为“/usr/local/Ascend/driver/topo/”，无需用户手工配置，产品出厂时自动配置，用户仅需了解其格式及字段含义。以2个AI Server，每个AI Server中两个NPU为例，组网如下所示：

![通信连接示例](figures/comm_connect_a5_example.png)

拓扑文件为JSON格式，配置示例如下：

```json
{
    "version": "2.0",
    "peer_count": 2,
    "peer_list":[
        { "local_id": 0},
        { "local_id": 1}
    ],
    "edge_count": 3,
    "edge_list": [
        {
            "net_layer": 0,           // Server内连接
            "link_type": "PEER2PEER",
            "protocols": ["UB_CTP"],
            "local_a": 0,              // 上图所示的NPU0
            "local_a_ports": ["1/0"],  // Server内连接端口为Die1的0号端口
            "local_b": 1,              // 上图所示的NPU1 
            "local_b_ports": ["1/0"],  // Server内连接端口为Die1的0号端口
            "topo_instance_id": 0,
            "topo_type": "1DMESH",
            "position": "DEVICE"
        },{
            "net_layer": 1,           // Server间连接
            "link_type": "PEER2NET",
            "protocols": ["UB_CTP"],
            "local_a": 0,             // 上图所示的NPU0
            "local_a_ports": ["0/4","0/5","0/6","0/7","1/5","1/6"],  // Server间连接端口为Die0的4、5、6、7号端口，Die1的5、6号端口
            "topo_instance_id": 0,
            "topo_type": "CLOS",
            "position": "DEVICE"
        },{
            "net_layer": 1,
            "link_type": "PEER2NET",
            "protocols": ["UB_CTP"],
            "local_a": 1,             // Server间连接
            "local_a_ports": ["0/4","0/5","0/6","0/7","1/5","1/6"],  // Server间连接端口为Die0的4、5、6、7号端口，Die1的5、6号端口
            "topo_instance_id": 0,
            "topo_type": "CLOS",
            "position": "DEVICE"
        }
    ]
}
```

网络拓扑文件配置说明如下所示：

| 一级配置项 | 二级配置项 | 配置说明 |
| --- | --- | --- |
| version |  | 必选。<br>拓扑文件模板版本信息。<br>固定配置为：2.0。 |
| peer_count |  | 必选。<br>当前AI Server包含的NPU个数，取值范围：[1,65]。 |
| peer_list |  | 必选。<br>当前AI Server包含的NPU列表。 |
|  | local_id | 必选。<br>NPU的物理ID，取值范围：[0,64]。 |
| edge_count |  | 必选。<br>物理连接边数，取值范围：[0, UINT32_MAX]。 |
| edge_list |  | 必选。<br>物理连接边列表。 |
|  | net_layer | 必选。<br>当前物理链路所属的网络层次，取值范围：[0,7]。 |
|  | link_type | 必选。<br>当前物理链路的连接类型，支持以下取值：<br>  - PEER2PEER<br>  - PEER2NET |
|  | protocols | 必选。<br>当前链路支持的协议列表，支持以下取值：<br>  - UB-CTP<br>  - UB_TP<br>  - ROCE<br>  - HCCS<br>  - TCP<br>  - UB_MEM<br>  - UB_OE |
|  | local_a | 必选。<br>通信链路一端NPU的物理ID。<br>该ID须存在于“peer_list”中。 |
|  | local_a_ports | 必选。<br>当前层次的通信链路在“local_a”对应的NPU中使用的端口列表，字符串类型，长度为1~32，列表中每个值的格式为：Die ID/端口ID。<br>例如："local_a_ports": ["1/0"]，代表Die1的0号端口。 |
|  | local_b | 必选。<br>通信链路另一端NPU的物理ID。<br>该ID须存在于“peer_list”中。 |
|  | local_b_ports | 必选。<br>当前层次的通信链路在“local_b”对应的NPU中使用的端口列表，字符串类型，长度为1~32，每个端口的配置格式为：Die ID/端口ID，多个端口之间用“,”分隔。<br>例如："local_a_ports": ["1/0"]，代表Die1的0号端口。 |
|  | topo_instance_id | 可选。<br>拓扑实例ID，取值范围：[0, UINT32_MAX]。 |
|  | topo_type | 可选。<br>拓扑实例的拓扑类型，支持以下取值：<br>- CLOS（默认值）：所有节点均可通，例如通过交换机连接的胖树结构。<br>- 1DMESH：Device和Device之间直连，全互联拓扑。 |
|  | position | 可选。<br>通信链路使用的网卡所属位置，支持以下取值：<br>  - HOST<br>  - DEVICE（默认值） |

## rank table文件配置

以两个AI Server，每个AI Server中两个NPU为例，配置示例如下：

```json
{
    "status": "completed",         // rank table可用标识，completed为可用
    "version": "2.0",
    "rank_count": 4,
    "rank_list": [
        {
            "rank_id": 0,
            "local_id": 0,
            "device_id": 0,
            "device_port": 16666,
            "level_list":  [
                {
                    "net_layer": 0,                 // Server内连接
                    "net_instance_id": "az0-rack0-pod0",
                    "net_type": "TOPO_FILE_DESC",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "IPV4",
                            "addr": "172.16.0.10",
                            "ports": ["1/0"]        // 连接端口为Die1的0号端口
                        }
                    ]
                },{
                    "net_layer": 1,                // Server间连接
                    "net_instance_id": "az0",
                    "net_type": "CLOS",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "IPV4",
                            "addr": "172.16.0.15",
                            "ports": ["0/4","0/5","0/6","0/7"],  // 连接端口为Die0的4、5、6、7号端口
                            "plane_id": "plane0"
                        },{
                            "addr_type": "IPV4",
                            "addr": "172.16.0.5",
                            "ports": ["1/5","1/6"],  // 连接端口为Die1的5、6号端口
                            "plane_id": "plane1"
                        }
                    ]
                }
            ]
        },
        {
            "rank_id": 1,
            "local_id": 1,
            "device_id": 1,
            "device_port": 16667,
            "level_list": [
                {
                    "net_layer": 0,
                    "net_instance_id" : "az0-rack0-pod0",
                    "net_type": "TOPO_FILE_DESC",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "IPV4",
                            "addr": "172.16.0.28",
                            "ports": ["1/0"]
                        }
                    ]
                },{
                    "net_layer": 1,
                    "net_instance_id": "az0",
                    "net_type": "CLOS",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "IPV4",
                            "addr": "172.16.0.33",
                            "ports": ["0/4","0/5","0/6","0/7"],
                            "plane_id": "plane2"
                        },{
                            "addr_type": "IPV4",
                            "addr": "172.16.0.23",
                            "ports": ["1/5","1/6"],
                            "plane_id": "plane3"
                        }
                    ]
                }
            ]
        },{
            "rank_id": 2,
            "local_id": 0,
            "device_id": 0,
            "device_port": 16668,
            "level_list": [
                {
                    "net_layer": 0,
                    "net_instance_id": "az0-rack0-pod1",
                    "net_type": "TOPO_FILE_DESC",
                    "net_attr": "",
                    "rank_addr_list": [
                    {
                    "addr_type": "IPV4",
                    "addr": "172.16.1.10",
                    "ports": ["1/0"]
                    }
                    ]
                },{
                    "net_layer": 1,
                    "net_instance_id": "az0",
                    "net_type": "CLOS",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "IPV4",
                            "addr": "172.16.1.15",
                            "ports": ["0/4","0/5","0/6","0/7"],
                            "plane_id": "plane0"
                        },{
                            "addr_type": "IPV4",
                            "addr": "172.16.1.5",
                            "ports": ["1/5","1/6"],
                            "plane_id": "plane1"
                        }
                    ]
                }
            ]
        },
        {
            "rank_id": 3,
            "local_id": 1,
            "device_id": 1,
            "device_port": 16669,
            "level_list": [
                {
                    "net_layer": 0,
                    "net_instance_id": "az0-rack0-pod1",
                    "net_type": "TOPO_FILE_DESC",
                    "net_attr": "",
                    "rank_addr_list": [
                    {
                    "addr_type": "IPV4",
                    "addr": "172.16.1.28",
                    "ports": ["1/0"]
                    }
                    ]
                },{
                    "net_layer": 1,
                    "net_instance_id": "az0",
                    "net_type": "CLOS",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "IPV4",
                            "addr": "172.16.1.33",
                            "ports": ["0/4","0/5","0/6","0/7"],
                            "plane_id": "plane2"
                        },{
                            "addr_type": "IPV4",
                            "addr": "172.16.1.23",
                            "ports": ["1/5","1/6"],
                            "plane_id": "plane3"
                        }
                    ]
                }
            ]
        }
    ]
}
```

rank table文件配置说明如下所示：

| 一级配置项 | 二级配置项 | 三级配置项 | 四级配置项 | 配置说明 |
| --- | --- | --- | --- | --- |
| status |  |  |  | 必选，rank table可用标识。<br>  - completed：表示rank table可用。<br>  - initializing：表示rank table不可用。 |
| version |  |  |  | 必选。<br>rank table模板版本信息。<br>固定配置为：2.0。 |
| rank_count |  |  |  | 必选。<br>参与集合通信的rank数量，即NPU数量，取值范围：[1, 65536]。 |
| rank_list |  |  |  | 必选。<br>参与集合通信的rank列表。 |
|  | rank_id |  |  | 必选。<br>rank唯一标识，请配置为整数，从0开始配置，且全局唯一，取值范围：[0, NPU总数-1]。<br>为方便管理，建议rank_id按照NPU物理连接顺序进行排序，即将物理连接上较近的NPU编排在一起。 |
|  | local_id |  |  | 必选。<br>NPU在当前AI Server中的唯一标识。请从0开始配置，取值范围：[0,64]。 |
|  | device_id |  |  | 必选。<br>NPU的物理ID，取值范围：[0,64]。 |
|  | device_port |  |  | 可选。<br>Device网卡的通信端口，取值范围为[1,65535]，需要确保指定的端口未被其他进程占用。需要注意，[1,1023]为系统保留端口，应避免使用这些端口。<br>单卡多进程的业务场景下，建议配置此字段，并且不同的业务进程需要设置不同的端口号，否则业务可能会因为端口冲突运行失败。 |
|  | level_list |  |  | 必选。<br>rank在每个网络层次的资源信息。<br>此列表下数组长度不能超过8。 |
|  |  | net_layer |  | 必选。<br>网络层次，取值范围：[0,7]。<br>net_layer需要从0开始编号，以升序的形式排列。 |
|  |  | net_instance_id |  | 必选。<br>该网络层次下的实例ID，用户自定义，同一个net_layer下保持唯一，长度不超过1024。 |
|  |  | net_type |  | 必选。<br>该网络层次的网络类型。<br>当“net_layer”取值为“0”时，该参数仅支持配置为“TOPO_FILE_DESC”，代表通过拓扑文件描述。<br>当“net_layer”取值为非0时，该参数支持以下两种配置：<br>  - CLOS：代表所有节点均可互通，例如通过交换机连接的胖树结构。<br>  - TOPO_FILE_DESC：代表网络类型通过拓扑文件描述。
|  |  | net_attr |  | 可选。<br>预留字段，表示该网络层次的其他额外信息。 |
|  |  | rank_addr_list |  | 必选。<br>当前rank在该网络层次使用的网络地址信息。<br>此列表下数组长度不能超过24，每个Die需要单独配置。 |
|  |  |  | addr_type | 必选。<br>当前rank的地址类型，支持以下取值：<br>  - EID<br>  - IPv4<br>  - IPv6 |
|  |  |  | addr | 必选。<br>当前rank的IP地址，字符串类型，长度范围1~256，地址需要符合“addr_type”指定的地址格式。 |
|  |  |  | ports | 必选。<br>“addr”绑定的端口列表，一个地址可以绑定到多个端口，每个端口的配置格式为：Die ID/端口ID，多个端口之间用“,”分隔。<br>注意：一个端口在同一个网络层次下，只能映射到一个地址。最多支持配置16个端口，支持的最大字符串长度为32。 |
|  |  |  | plane_id | 可选。<br>网络平面ID，默认值为“0”。 |
