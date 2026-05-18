# rank table配置资源信息（Atlas A3 训练系列产品/Atlas A3 推理系列产品）

针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，集群训练支持超节点模式组网与典型组网。需要注意，针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，每个NPU包含2个Device（即两个Die），每个Device为一个rank。

> [!NOTE]说明
> rank table文件为JSON格式，本节所示JSON文件示例中的注释仅为方便理解，实际使用时，请删除JSON文件中的注释。

## 超节点模式组网

以包含两个超节点，每个超节点内包含两个AI Server，每个AI Server内四个Device的资源配置文件为例，配置示例如下：

```json
{
    "status": "completed",         // rank table可用标识，completed为可用
    "version": "1.2",              // rank table模板版本信息，针对超节点模式组网，配置为：1.2
    "server_count":"4",            // 参与训练的AI Server数目
    "server_list": [
        {
            "server_id": "node_0",     // AI Server标识，string类型，请确保全局唯一
            "host_ip":"172.16.0.100",  // AI Server的Host IP地址
            "device": [
                {"device_id": "0","super_device_id":"0","device_ip": "192.168.1.6","device_port":"16666","backup_device_ip":"192.168.1.7","backup_device_port":"16667","host_port":"16665","rank_id": "0"}, // device_id为处理器的物理ID；super_device_id为处理器在超节点系统中的物理ID；device_ip为处理器真实网卡IP；device_port为处理器的网卡通信端口；backup_device_ip为开启超节点间算子重执行特性时的备用IP；host_port为Host网卡的通信端口；rank_id为rank标识，从0开始配置
                {"device_id": "1","super_device_id":"1","device_ip": "192.168.1.7","device_port":"16666","backup_device_ip":"192.168.1.6","backup_device_port":"16667","host_port":"16666","rank_id": "1"},
                {"device_id": "2","super_device_id":"2","device_ip": "192.168.1.8","device_port":"16668","backup_device_ip":"192.168.1.9","backup_device_port":"16670","host_port":"16667","rank_id": "2"},
                {"device_id": "3","super_device_id":"3","device_ip": "192.168.1.9","device_port":"16669","backup_device_ip":"192.168.1.8","backup_device_port":"16667","host_port":"16668","rank_id": "3"}]
        },
        {
            "server_id": "node_1",
            "host_ip":"172.16.0.101",
            "device": [
                {"device_id": "0","super_device_id":"4","device_ip": "192.168.2.6","device_port":"16666","backup_device_ip":"192.168.2.7","backup_device_port":"16667","host_port":"16665","rank_id": "4"},
                {"device_id": "1","super_device_id":"5","device_ip": "192.168.2.7","device_port":"16666","backup_device_ip":"192.168.2.6","backup_device_port":"16667","host_port":"16666","rank_id": "5"},
                {"device_id": "2","super_device_id":"6","device_ip": "192.168.2.8","device_port":"16668","backup_device_ip":"192.168.2.9","backup_device_port":"16670","host_port":"16667","rank_id": "6"},
                {"device_id": "3","super_device_id":"7","device_ip": "192.168.2.9","device_port":"16669","backup_device_ip":"192.168.2.8","backup_device_port":"16667","host_port":"16668","rank_id": "7"}]
        },
        {
            "server_id": "node_2",
            "host_ip":"172.16.0.102",
            "device": [
                {"device_id":"0","super_device_id":"0","device_ip":"192.168.3.6","device_port":"16666","backup_device_ip":"192.168.3.7","backup_device_port":"16667","host_port":"16665","rank_id":"8"},
                {"device_id":"1","super_device_id":"1","device_ip":"192.168.3.7","device_port":"16666","backup_device_ip":"192.168.3.6","backup_device_port":"16667","host_port":"16666","rank_id":"9"},
                {"device_id":"2","super_device_id":"2","device_ip":"192.168.3.8","device_port":"16668","backup_device_ip":"192.168.3.9","backup_device_port":"16670","host_port":"16667","rank_id":"10"},
                {"device_id":"3","super_device_id":"3","device_ip":"192.168.3.9","device_port":"16669","backup_device_ip":"192.168.3.8","backup_device_port":"16667","host_port":"16668","rank_id":"11"}]
        },
        {
            "server_id": "node_3",
            "host_ip":"172.16.0.103",
            "device": [
                {"device_id":"0","super_device_id":"4","device_ip":"192.168.4.6","device_port":"16666","backup_device_ip":"192.168.4.7","backup_device_port":"16667","host_port":"16665","rank_id":"12"},
                {"device_id":"1","super_device_id":"5","device_ip":"192.168.4.7","device_port":"16666","backup_device_ip":"192.168.4.6","backup_device_port":"16667","host_port":"16666","rank_id":"13"},
                {"device_id":"2","super_device_id":"6","device_ip":"192.168.4.8","device_port":"16668","backup_device_ip":"192.168.4.9","backup_device_port":"16670","host_port":"16667","rank_id":"14"},
                {"device_id":"3","super_device_id":"7","device_ip":"192.168.4.9","device_port":"16669","backup_device_ip":"192.168.4.8","backup_device_port":"16667","host_port":"16668","rank_id":"15"}]
        }
    ],
    "super_pod_list": [
        {
            "super_pod_id": "0",          // 超节点唯一标识
            "server_list": [              // 超节点内的AI Server列表
                {"server_id": "node_0"},  // server_id为Server标识，与"server_list"中的server_id对应 
                {"server_id": "node_1"}]
        },
        {
            "super_pod_id": "1",
            "server_list": [
                {"server_id":"node_2"},
                {"server_id":"node_3"}]
        }
    ]
}
```

rank table配置文件说明如下所示：

| 一级配置项 | 二级配置项 | 三级配置项 | 配置说明 |
| --- | --- | --- | --- |
| status |  |  | 必选。<br>rank table可用标识。<br>  - completed：表示rank table可用。<br>  - initializing：表示rank table不可用。 |
| version |  |  | 必选。<br>rank table模板版本信息。<br>针对超节点模式组网，配置为：1.2。 |
| server_count |  |  | 可选。<br>参与集合通信的AI Server个数。 |
| server_list |  |  | 必选。<br>参与集合通信的AI Server列表。 |
|  | server_id |  | 必选。<br>AI Server标识，字符串类型，长度小于等于64，请确保全局唯一。<br>配置示例：node_0。 |
|  | host_ip |  | 可选。<br>AI Server的Host IP地址，要求为常规IPv4格式。<br>开启HCCL重执行特性的场景下，此字段必须配置，否则重执行失效，走无重执行流程。<br>重执行特性默认关闭，可参见环境变量[HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md)。 |
|  | device |  | 必选。<br>Device列表。 |
|  |  | device_id | 必选。<br>AI处理器的物理ID，即Device在AI Server上的序列号。<br>可通过执行“ls /dev/davinci*”命令获取AI处理器的物理ID。<br>例如：显示/dev/davinci0，表示AI处理器的物理ID为0。<br>取值范围：\[0，实际Device数量-1]。<br>注意：“device_id”配置项的优先级高于环境变量“ASCEND_DEVICE_ID”。 |
|  |  | super_device_id | 可选（若不配置该字段，会采用“AI Server模式”）。<br>AI处理器在超节点系统中的物理ID，为超节点系统中NPU唯一标识。<br>开发者可通过npu-smi命令查询，命令示例如下：<br>npu-smi info -t spod-info -i id -c chip_id<br><br>  - id：设备id。通过npu-smi info -l命令查出的NPU ID即为设备id。<br>  - chip_id：芯片id。通过npu-smi info -m命令查出的Chip ID即为芯片id。<br><br>回显中的“SDID”即为超节点系统中的NPU唯一标识。 |
|  |  | device_ip | 可选。<br>AI处理器集成网卡IP，全局唯一，要求为常规IPv4或IPv6格式。<br>需要注意：<br>  1. 当组网中包含多个超节点时，device_ip必须配置。<br>  2. 若组网中仅包含一个超节点，device_ip在以下场景必须配置，其他场景可不配置。当超节点内使用RDMA通信时（即将环境变量HCCL_INTER_HCCS_DISABLE配置为TRUE，禁用了HCCS功能的场景），device_ip必须配置。<br>可以在当前AI Server执行命令cat /etc/hccn.conf获取网卡IP，例如：<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>查询到的address_xx即为网卡IP，address后的序号为AI处理器的物理ID，即device_id，后面的ip地址即为需要用户填入的该device对应的网卡IP。 |
|  |  | device_port | 可选。<br>Device网卡的通信端口，取值范围为\[1,65535]，需要确保指定的端口未被其他进程占用。需要注意，\[1,1023]为系统保留端口，应避免使用这些端口。<br>单卡多进程的业务场景下，建议配置此字段，并且不同的业务进程需要设置不同的端口号，否则业务可能会因为端口冲突运行失败。 |
|  |  | backup_device_ip | 可选。<br>当超节点间通信开启了算子重执行特性时，如果Device网卡故障（RDMA链路故障），可通过此参数将同一NPU中的另一个Die网卡指定为备用Device网卡，提升算子重执行的成功率，此种启用备用Device网卡的通信方式称之为借轨通信。<br>“backup_device_ip”为常规IPv4或IPv6格式，查询方法可参见“device_ip”的配置说明。<br>需要注意：<br>  1. “backup_device_ip”与“device_ip”对应的Device需要属于同一NPU，即同一NPU下Die0与Die1对应的Device网卡才能互为备用。<br>  2. 仅通信算子展开模式为AI_CPU，且开启了超节点间通信的算子重执行特性时，此配置才生效，即：export HCCL_OP_EXPANSION_MODE="AI_CPU"<br>export HCCL_OP_RETRY_ENABLE="L1:1,L2:1"<br>L2代表通信域物理范围为超节点间通信域，取值为1代表开启通信算子重执行特性。<br>  3. 为了确保借轨功能正常执行，需要满足如下条件：<br>  - 备用网卡的通信链路正常。<br>  - 互为主备的Device都在业务可见范围内。例如，NPU1中包含Device0与Device1两个Die，互为主备，假设通过环境变量ASCEND_RT_VISIBLE_DEVICES指定了业务可见Device为Device0，Device1不对业务可见，则借轨功能无法执行。<br>  4. 如果通信过程中发生了借轨（假设某个NPU的Die0网卡故障，启用了备用的Die1网卡），原Die0的网卡流量也会通过Die1网卡收发，导致Die1的流量增大，总体性能会由于物理带宽减半、端口冲突导致下降。<br>  5. 开启借轨场景下，若NPU0的Die0网卡故障，会切换到其备用网卡Die1，由于两个NPU之间的通信要求本端与对端同时切换为备用网卡，因此NPU1也会从Die0切换为Die1，即“[图1](#figure1)”所示。但如果Die0与Die1之间本身就存在通信任务，此时借轨功能无法执行。<br>  6. 开启借轨通信功能时，建议一个NPU的两个Die分配给同一个训练或推理任务。如果同一个NPU的两个Die分给两个不同的训练或推理任务，一个任务发生故障，会借用另一个任务的网卡，两个任务均会发生一定程度的性能下降。<br>  7. 同一NPU仅支持发生一次借轨，且不支持回切。如[图2](#figure2)所示，“图示一”中NPU0 -> NPU1间通信链路故障，启用了备用链路，发生借轨，通信正常进行；若再发生“图示二”所示故障，则不再支持借轨，会报错退出。 |
|  |  | backup_device_port | 可选。<br>备用Device网卡的通信端口，取值范围为\[1,65535]，需要确保指定的端口未被其他进程占用。需要注意，\[1,1023]为系统保留端口，应避免使用这些端口。<br>若启用了借轨通信功能，且业务为单卡多进程的场景，建议配置此字段，并且不同的业务进程需要设置不同的端口号，否则业务可能会因为端口冲突运行失败。<br>注意：同一个Device网卡作为主网卡与备网卡时配置的通信端口号不能相同。 |
|  |  | host_port | 可选。<br>Host网卡的通信端口，取值范围为\[1,65535]，同一AI Server内每个Device对应的host_port应各不相同，且需要确保指定的端口未被其他进程占用。需要注意，\[1,1023]为系统保留端口，应避免使用这些端口。<br>若通过环境变量[HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md)开启了HCCL重执行特性，且业务为单卡多进程场景（即多个业务进程同时共用一个NPU），建议配置此字段，并且不同的业务进程需要设置不同的端口号，否则业务可能会因为端口冲突运行失败。 |
|  |  | rank_id | 必选。<br>rank唯一标识，请配置为整数，从0开始配置，且全局唯一，取值范围：\[0, 总Device数量-1]。<br>- 建议rank_id按照Device物理连接顺序进行排序，即将物理连接上较近的Device编排在一起，否则可能会对性能造成影响。<br>&nbsp;&nbsp;  例如，若device_ip按照物理连接从小到大设置，则rank_id也建议按照从小到大的顺序设置。<br>- 不同AI Server中的rank_id不支持交叉配置。<br> &nbsp;&nbsp; 正例：server 1中的rank_id集合为{0,1,2,3}，server 2中的rank_id集合为{4,5,6,7}。<br> &nbsp;&nbsp; 反例：server 1中的rank_id集合为{0,1,2,7}，server 2中的rank_id集合为{3,4,5,6}。 |
| super_pod_list |  |  | 可选（若不配置该字段，会采用“AI Server模式”）。<br>参与集合通信的超节点列表。 |
|  | super_pod_id |  | 若配置了“super_pod_list”，该字段必选。<br>超节点唯一标识，全局唯一，支持如下两种配置方式：<br>  - 配置为超节点物理ID，可通过npu-smi工具查询，命令示例如下：npu-smi info -t spod-info -i id -c chip_id<br>id：设备id。通过npu-smi info -l命令查出的NPU ID即为设备id。chip_id：芯片id。通过npu-smi info -m命令查出的Chip ID即为芯片id。<br>回显中的“Super Pod ID”即为超节点的物理ID。<br>  - id：设备id。通过npu-smi info -l命令查出的NPU ID即为设备id。<br>  - chip_id：芯片id。通过npu-smi info -m命令查出的Chip ID即为芯片id。<br>  - 用户自定义编号，字符串格式，需全局唯一。自定义配置ID的场景下，用户可以把一个物理超节点划分为多个小的逻辑超节点使用，比如一个物理超节点有8个AI Server节点，用户可以把前4个AI Server节点作为一个小的超节点，编号为super_pod_1，后4个AI Server节点作为一个小的超节点，编号为super_pod_2。 |
|  | server_list |  | 必选。<br>超节点内的AI Server列表。 |
|  |  | server_id | 必选。<br>Server标识，字符串类型，与"server_list"中的server_id对应。<br>配置示例：node_0。 |

> [!NOTE]说明
> 如果组网中存在多个超节点，请将属于同一超节点内的AI Server信息配置在一起。假设有两个超节点，标识分别为“0”和“1”，请先配置“0”中的AI Server信息，再配置“1”中的AI Server信息，不支持“0”中的AI Server信息与“1”中的AI Server信息交叉配置。

**图 1**  借轨通信切换示例<a id="figure1"></a>  
![](figures/borrow_comm_switch_example.png "借轨通信切换示例")

**图 2**  同一NPU仅支持一次借轨示例<a id="figure2"></a>  
![](figures/npu_single_borrow_example.png "同一NPU仅支持一次借轨示例")

## 典型集群组网（即AI Server模式）

以包含两个AI Server，每个AI Server内2个Device为例，rank table文件配置示例如下：

```json
{
    "status":"completed",   // rank table可用标识，completed为可用
    "version":"1.0",        // rank table模板版本信息，典型集群组网，配置为：1.0
    "server_count":"2",     //参与训练的AI Server数目，此例中，有两个AI Server
    "server_list":
    [
        {
            "server_id":"node_0",       // AI Server标识，String类型，请确保全局唯一
            "host_ip":"172.16.0.110",   // AI Server的Host IP地址
            "device":[   // AI Server中的Device列表
                {
                    "device_id":"0",              // 处理器的物理ID
                    "device_ip":"192.168.1.8",    // 处理器真实网卡IP
                    "device_port":"16667",        // 处理器的网卡通信端口
                    "host_port":"16666",          // Host网卡的通信端口
                    "rank_id":"0"                 // rank的标识，从0开始配置
                },
                {
                    "device_id":"1",
                    "device_ip":"192.168.1.9", 
                    "device_port":"16667",
                    "host_port":"16667", 
                    "rank_id":"1"
                }
            ]
        },
        {
            "server_id":"node_1",
            "host_ip":"172.16.0.111",
            "device":[
                {
                    "device_id":"0",
                    "device_ip":"192.168.2.8",
                    "device_port":"16667",
                    "host_port":"16666", 
                    "rank_id":"2"
                },
                {
                    "device_id":"1",
                    "device_ip":"192.168.2.9", 
                    "device_port":"16667",
                    "host_port":"16667", 
                    "rank_id":"3"
                }
            ]
        }
    ]
}
```

rank table配置文件说明如下表所示：

| 一级配置项 | 二级配置项 | 三级配置项 | 配置说明 |
| --- | --- | --- | --- |
| status |  |  | 必选。<br>rank table可用标识。<br>  - completed：表示rank table可用。<br>  - initializing：表示rank table不可用。 |
| version |  |  | 必选。<br>rank table模板版本信息。<br>针对典型集群组网，配置为：1.0。 |
| server_count |  |  | 必选。<br>参与集合通信的AI Server个数。 |
| server_list |  |  | 必选。<br>参与集合通信的AI Server列表。 |
|  | server_id |  | 必选。<br>AI Server标识，字符串类型，长度小于等于64，请确保全局唯一。<br>配置示例：node_0。 |
|  | host_ip |  | 可选。<br>AI Server的Host IP地址，要求为常规IPv4格式。<br>开启HCCL重执行特性的场景下，此字段必须配置，否则重执行失效，走无重执行流程。<br>重执行特性默认关闭，可参见环境变量[HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md)。 |
|  | device |  | 必选。<br>AI Server中的Device列表。 |
|  |  | device_id | 必选。<br>AI处理器的物理ID，即Device在AI Server上的序列号。<br>可通过执行“ls /dev/davinci*”命令获取AI处理器的物理ID。<br>例如：显示/dev/davinci0，表示AI处理器的物理ID为0。<br>取值范围：\[0，实际Device数量-1]。<br>注意：“device_id”配置项的优先级高于环境变量“ASCEND_DEVICE_ID”。 |
|  |  | device_ip | 可选。<br>AI处理器集成网卡IP，全局唯一，要求为常规IPv4或IPv6格式。<br>需要注意：<br>  - 多机场景下，device_ip必须配置。<br>  - 单机场景下，device_ip可不配置。<br>可以在当前AI Server执行指令cat /etc/hccn.conf获取网卡IP，例如：<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>查询到的address_xx即为网卡IP，address后的序号为AI处理器的物理ID，即device_id，后面的ip地址即为需要用户填入的该device对应的网卡IP。 |
|  |  | device_port | 可选。<br>Device网卡的通信端口，取值范围为\[1,65535]，需要确保指定的端口未被其他进程占用。需要注意，\[1,1023]为系统保留端口，应避免使用这些端口。<br>单卡多进程的业务场景下，建议配置此字段，并且不同的业务进程需要设置不同的端口号，否则业务可能会因为端口冲突运行失败。 |
|  |  | host_port | 可选。<br>Host网卡的通信端口，取值范围为\[1,65535]，同一AI Server内每个Device对应的host_port应各不相同，且需要确保指定的端口未被其他进程占用。需要注意，\[1,1023]为系统保留端口，应避免使用这些端口。<br>若通过环境变量[HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md)开启了HCCL重执行特性，且业务为单卡多进程场景（即多个业务进程同时共用一个NPU），建议配置此字段，并且不同的业务进程需要设置不同的端口号，否则业务可能会因为端口冲突运行失败。 |
|  |  | rank_id | 必选。<br>Rank唯一标识，请配置为整数，从0开始配置，且全局唯一，取值范围：\[0, 总Device数量-1]。<br>- 建议rank_id按照Device物理连接顺序进行排序，即将物理连接上较近的Device编排在一起，否则可能会对性能造成影响。<br>&nbsp;&nbsp;  例如，若device_ip按照物理连接从小到大设置，则rank_id也建议按照从小到大的顺序设置。<br>- 不同AI Server中的rank_id不支持交叉配置。<br> &nbsp;&nbsp; 正例：server 1中的rank_id集合为{0,1,2,3}，server 2中的rank_id集合为{4,5,6,7}。<br> &nbsp;&nbsp; 反例：server 1中的rank_id集合为{0,1,2,7}，server 2中的rank_id集合为{3,4,5,6}。 |
