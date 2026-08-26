# Rank Table Configuration Resource Information (Ascend 950PR/Ascend 950DT)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:53:11.484Z pushedAt=2026-08-03T08:24:20.308Z -->

For Ascend 950PR/Ascend 950DT, the rank table file must be used together with the Topology File to initialize the HCCL communication domain.

> [!NOTE]Note
>
> - The rank table file supports a maximum size of 1 GB.
> - The rank table file is in JSON format. The comments in the JSON file examples shown in this section are only for ease of understanding. Delete the comments in the JSON file during actual use.

## Topology File Configuration

The topology file stores physical topology and routing information. It is located in `/usr/local/Ascend/driver/topo/` and does not require manual configuration—it is pre-configured at the factory. Users only need to understand its format and field meanings. The following figure shows the networking topology, using two AI servers with two NPUs each as an example.

![Communication Connection Example](figures/comm_connect_a5_example.png)

The Topology File is in JSON format. A configuration example is shown below:

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
            "net_layer": 0,           // Intra-server connection
            "link_type": "PEER2PEER",
            "protocols": ["UB_CTP"],
            "local_a": 0,              // NPU 0 shown in the preceding figure
            "local_a_ports": ["1/0"],  // Intra-server connection port: port 0 of Die 1
            "local_b": 1,              // NPU 1 shown in the preceding figure
            "local_b_ports": ["1/0"],  // Intra-server connection port: port 0 of Die 1
            "topo_instance_id": 0,
            "topo_type": "1DMESH",
            "position": "DEVICE"
        },{
            "net_layer": 1,           // Inter-server connection
            "link_type": "PEER2NET",
            "protocols": ["UB_CTP"],
            "local_a": 0,             // NPU0 shown in the figure above
            "local_a_ports": ["0/4","0/5","0/6","0/7","1/5","1/6"],  // The inter-server connection ports are ports 4, 5, 6, and 7 of Die 0, and ports 5 and 6 of Die 1.
            "topo_instance_id": 0,
            "topo_type": "CLOS",
            "position": "DEVICE"
        },{
            "net_layer": 1,
            "link_type": "PEER2NET",
            "protocols": ["UB_CTP"],
            "local_a": 1,             // Inter-server connection
            "local_a_ports": ["0/4","0/5","0/6","0/7","1/5","1/6"],  // The inter-server connection ports are ports 4, 5, 6, and 7 of Die 0, and ports 5 and 6 of Die 1.
            "topo_instance_id": 0,
            "topo_type": "CLOS",
            "position": "DEVICE"
        }
    ]
}
```

The network topology file configuration is described as follows:

| Level-1 Configuration Item | Level-2 Configuration Item | Configuration Description |
| --- | --- | --- |
| version |  | Mandatory.<br>Topology file template version. Fixed value: `2.0`. |
| peer_count |  | Mandatory.<br>Number of NPUs in the current AI server. Value range: `[1, 65]`. |
| peer_list |  | Mandatory.<br>List of NPUs in the current AI server. |
|  | local_id | Mandatory.<br>Physical ID of the NPU. Value range: `[0, 64]`. |
| edge_count |  | Mandatory.<br>Number of physical connection edges. Value range: `[0, UINT32_MAX]`. |
| edge_list |  | Mandatory.<br>List of physical connection edges. |
|  | net_layer | Mandatory.<br>Network layer to which the current physical link belongs. Value range: `[0, 7]`. |
|  | link_type | Mandatory.<br>Connection type of the current physical link. Supported values:<br>  - `PEER2PEER`<br>  - `PEER2NET` |
|  | protocols | Mandatory.<br>List of protocols supported by the current link. Supported values:<br>  - `UB_CTP`<br>  - `UB_TP`<br>  - `ROCE`<br>  - `HCCS`<br>  - `TCP`<br>  - `UB_MEM`<br>  - `UBOE` |
|  | local_a | Mandatory.<br>Physical ID of the NPU at one end of the communication link.<br>This ID must exist in `peer_list`. |
|  | local_a_ports | Mandatory.<br>List of ports on the `local_a` NPU used for the communication link at this layer. String type, 1–32 characters. Each port is formatted as `Die ID/Port ID`.<br>Example: `"local_a_ports": ["1/0"]` indicates port 0 of Die 1. |
|  | local_b | Mandatory.<br>Physical ID of the NPU at the other end of the communication link.<br>This ID must exist in `peer_list`. |
|  | local_b_ports | Mandatory.<br>List of ports on the `local_b` NPU used for the communication link at this layer. String type, 1–32 characters. Each port is formatted as `Die ID/Port ID`, and multiple ports are separated by commas.<br>Example: `"local_b_ports": ["1/0"]` indicates port 0 of Die 1. |
|  | topo_instance_id | Optional.<br>Topology instance ID. Value range: `[0, UINT32_MAX]`. |
|  | topo_type | Optional.<br>Topology type of the topology instance. Supported values:<br>  - `CLOS` (default): All nodes are reachable, e.g., a fat-tree structure connected through switches.<br>  - `1DMESH`: Devices are directly connected in a fully interconnected topology. |
|  | position | Optional.<br>Location of the NIC used by the communication link. Supported values:<br>  - `HOST`<br>  - `DEVICE` (default) |

## Rank Table File Configuration

The following example uses two AI Servers, each with two NPUs, to demonstrate the rank table configuration for the IPv4 address type:

```json
{
    "status": "completed",         // Rank table availability flag. "completed" indicates available.
    "version": "2.0",
    "rank_count": 4,
    "rank_list": [
        {
            "rank_id": 0,
            "local_id": 0,
            "device_id": 0,
            "device_port": 16666,
            "host_port": 60001,
            "level_list":  [
                {
                    "net_layer": 0,                 // Intra-server connection.
                    "net_instance_id": "az0-rack0-pod0",
                    "net_type": "TOPO_FILE_DESC",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "IPV4",
                            "addr": "172.16.0.10",
                            "ports": ["1/0"]        // The connection port is port 0 of Die1.
                        }
                    ]
                },{
                    "net_layer": 1,                // Connection between servers.
                    "net_instance_id": "az0",
                    "net_type": "CLOS",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "IPV4",
                            "addr": "172.16.0.15",
                            "ports": ["0/4","0/5","0/6","0/7"],  // The connected ports are ports 4, 5, 6, and 7 of Die0.
                            "plane_id": "plane0"
                        },{
                            "addr_type": "IPV4",
                            "addr": "172.16.0.5",
                            "ports": ["1/5","1/6"],  // The connected ports are ports 5 and 6 of Die1.
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
            "host_port": 60002,
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
            "host_port": 60003,
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
            "host_port": 60004,
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

The following uses two NPUs as an example to show the rank table configuration of the EID address type. In the example, net_layer 0 uses EID to configure the NPU communication address on the device side, and net_layer 3 uses IPv4 to configure the host-side communication address:

```json
{
    "status": "completed",
    "version": "2.0",
    "rank_count": 2,
    "rank_list": [
        {
            "rank_id": 0,            // Unique identifier of the rank.
            "device_id": 0,          // NPU physical ID
            "local_id": 0,           // Unique identifier of the NPU in the current AI Server
            "level_list": [
                {
                    "net_layer": 0,  // Device-side connection
                    "net_instance_id": "superpod_0_0",
                    "net_type": "TOPO_FILE_DESC",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "EID",
                            "addr": "000000000000000000100000dfdf0020",  // EID obtained through HCCN TOOL query
                            "ports": ["0/4"],                            // Connection port is port 4 of Die 0
                            "plane_id": "plane0"
                        },
                        {
                            "addr_type": "EID",
                            "addr": "000000000000000000100000dfdf0028",  // EID obtained through HCCN TOOL query.
                            "ports": ["0/5"],                            // Connected to port 5 of Die0.
                            "plane_id": "plane0"
                        },
                        {
                            "addr_type": "EID",
                            "addr": "000000000000000000100000dfdf0030",  // EID obtained through HCCN TOOL query.
                            "ports": ["0/6"],                            // Connected to port 6 of Die0.
                            "plane_id": "plane0"
                        }
                    ]
                },
                {
                    "net_layer": 3,  // Host-side connection.
                    "net_instance_id": "cluster",
                    "net_type": "CLOS",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "IPV4",
                            "addr": "192.168.100.101",  // IPv4 address obtained by running ifconfig -a on the host.
                            "ports": ["d2h"],           // Host NIC port
                            "plane_id": "plane0"
                        }
                    ]
                }
            ]
        },
        {
            "rank_id": 1,            // Unique identifier of the rank.
            "device_id": 1,          // Physical ID of the NPU.
            "local_id": 1,           // Unique identifier of the NPU within the current AI server.
            "level_list": [
                {
                    "net_layer": 0,  // Device-side connection.
                    "net_instance_id": "superpod_0_0",
                    "net_type": "TOPO_FILE_DESC",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "EID",
                            "addr": "000000000000000000100000dfdf0021",  // EID obtained through HCCN TOOL query.
                            "ports": ["0/4"],                            // Connected to port 4 of Die0.
                            "plane_id": "plane0"
                        },
                        {
                            "addr_type": "EID",
                            "addr": "000000000000000000100000dfdf0029",  // EID obtained through HCCN TOOL query.
                            "ports": ["0/5"],                            // Connected to port 5 of Die0.
                            "plane_id": "plane0"
                        },
                        {
                            "addr_type": "EID",
                            "addr": "000000000000000000100000dfdf0031",  // EID obtained through HCCN TOOL query
                            "ports": ["0/6"],                            // Connected to port 6 of Die 0
                            "plane_id": "plane0"
                        }
                    ]
                },
                {
                    "net_layer": 3,  // Host-side connection
                    "net_instance_id": "cluster",
                    "net_type": "CLOS",
                    "net_attr": "",
                    "rank_addr_list": [
                        {
                            "addr_type": "IPV4",
                            "addr": "192.168.100.102",  // IPv4 address obtained through ifconfig -a on the host
                            "ports": ["d2h"],           // Host NIC port
                            "plane_id": "plane0"
                        }
                    ]
                }
            ]
        }
    ]
}
```

The following table describes the rank table file configuration:

| Level-1 Configuration Item | Level-2 Configuration Item | Level-3 Configuration Item | Level-4 Configuration Item | Configuration Description |
| --- | --- | --- | --- | --- |
| status |  |  |  | Optional. Rank table availability status.<br>  - `completed`: The rank table is available.<br>  - `initializing`: The rank table is unavailable. |
| version |  |  |  | Mandatory.<br>Rank table template version. Fixed value: `2.0`. |
| rank_count |  |  |  | Mandatory.<br>Number of ranks participating in collective communication, i.e., the number of NPUs. Value range: `[1, 65536]`. |
| rank_list |  |  |  | Mandatory.<br>List of ranks participating in collective communication. |
|  | rank_id |  |  | Mandatory.<br>Rank identifier, an integer starting from `0`. Must be globally unique. Value range: `[0, total NPU count - 1]`.<br>For ease of management, it is recommended to assign `rank_id` in order of physical NPU connectivity—group NPUs that are closer in topology together. |
|  | local_id |  |  | Mandatory.<br>Unique identifier of the NPU within the current AI server, starting from `0`. Value range: `[0, 64]`. |
|  | device_id |  |  | Mandatory.<br>Physical ID of the NPU. Value range: `[0, 64]`. |
|  | device_port |  |  | Optional.<br>Communication port of the device NIC. Value range: `[1, 65535]`. Ensure the specified port is not occupied by other processes. Ports in the range `[1, 1023]` are system-reserved and should be avoided.<br>In single-card multi-process scenarios, it is recommended to configure this field with distinct port numbers for different processes. Otherwise, port conflicts may cause service startup failures. |
|  | host_port |  |  | Optional.<br>Communication port of the host NIC. Value range: `[1, 65535]`. Ensure the specified port is not occupied by other processes. Ports in the range `[1, 1023]` are system-reserved and should be avoided.<br>In host NIC collective communication scenarios, it is recommended to configure this field with distinct port numbers for different processes. Otherwise, port conflicts may cause service startup failures. |
|  | level_list |  |  | Mandatory.<br>Resource information of the rank at each network layer.<br>The array length must not exceed 8. |
|  |  | net_layer |  | Mandatory.<br>Network layer. Value range: `[0, 7]`.<br>`net_layer` must be numbered consecutively starting from `0`. |
|  |  | net_instance_id |  | Mandatory.<br>Instance ID at this network layer. User-defined and must be unique within the same `net_layer`. Length must not exceed 1024 characters. |
|  |  | net_type |  | Mandatory.<br>Network type of this layer.<br>When `net_layer` is set to `0`, this parameter can only be set to `TOPO_FILE_DESC`, indicating that the network is described via a topology file.<br>When `net_layer` is set to a non-zero value, this parameter supports the following values:<br>  - `CLOS`: All nodes are reachable, e.g., a fat-tree structure connected through switches.<br>  - `TOPO_FILE_DESC`: The network type is described via a topology file. |
|  |  | net_attr |  | Optional.<br>Reserved field for additional information about this network layer. |
|  |  | rank_addr_list |  | Mandatory.<br>Network address information used by the current rank at this network layer.<br>The array length must not exceed 24. Each Die must be configured separately. |
|  |  |  | addr_type | Mandatory.<br>Address type of the current rank. Supported values:<br>  - `EID`: Obtain the EID using HCCN TOOL.<br>  - `IPv4`: Host NIC only. Obtain it on the host using `ifconfig -a`. Configure the NIC that has a one-hop PCIe-SW connection to the NPU.<br>  - `IPv6`: Host NIC only. Obtain it on the host using `ifconfig -a`. Configure the NIC that has a one-hop PCIe-SW connection to the NPU. |
|  |  |  | addr | Mandatory.<br>Network address of the current rank. String type, 1–256 characters. Must conform to the format specified by `addr_type`. |
|  |  |  | ports | Mandatory.<br>List of ports bound to `addr`. One address can be bound to multiple ports. Each port is formatted as `Die ID/Port ID`, and multiple ports are separated by commas.<br>Note: A port can be mapped to only one address within the same network layer. A maximum of 16 ports can be configured, and the string length must not exceed 32 characters. |
|  |  |  | plane_id | Optional.<br>Network plane ID. Default value: `0`. |
