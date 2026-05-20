# rank table配置资源信息（Atlas 训练系列产品）

针对Atlas 训练系列产品，在rank table文件中配置参与训练的AI处理器信息支持两种配置模板，全新场景推荐使用模板一，模板二用于兼容部分已有场景。

> [!NOTE]说明
> rank table文件为JSON格式，本节所示JSON文件示例中的注释仅为方便理解，实际使用时，请删除JSON文件中的注释。

## 模板一（推荐使用）

以包含两个AI Server，每个AI Server内2个Device为例，rank table文件配置示例如下：

```json
{
    "status":"completed",   // rank table可用标识，completed为可用
    "version":"1.0",        // rank table模板版本信息，配置为：1.0
    "server_count":"2",     // 参与训练的AI Server数目，此例中，有两个AI Server
    "server_list":
    [
        {
            "server_id":"node_0",  //AI Server标识，String类型，请确保全局唯一
            "device":[             // AI Server中的Device列表
                {
                    "device_id":"0",   // 处理器的物理ID
                    "device_ip":"192.168.1.8",   // 处理器真实网卡IP
                    "rank_id":"0"                // rank的标识，从0开始配置，请确保全局唯一
                },
                {
                    "device_id":"1",
                    "device_ip":"192.168.1.9", 
                    "rank_id":"1"
                }
            ]
        },
        {
            "server_id":"node_1",
            "device":[
                {
                    "device_id":"0",
                    "device_ip":"192.168.2.8",
                    "rank_id":"2"
                },
                {
                    "device_id":"1",
                    "device_ip":"192.168.2.9", 
                    "rank_id":"3"
                }
            ]
        }
    ]
}
```

rank table配置文件说明如下所示：

| 一级配置项 | 二级配置项 | 三级配置项 | 配置说明 |
| --- | --- | --- | --- |
| status |  |  | 必选。<br>rank table可用标识。<br>  - completed：表示rank table可用。<br>  - initializing：表示rank table不可用。 |
| version |  |  | 必选。<br>rank table模板版本信息。<br>配置为：1.0。 |
| server_count |  |  | 必选。<br>参与集合通信的AI Server个数。 |
| server_list |  |  | 必选。<br>参与集合通信的AI Server列表。 |
|  | server_id |  | 必选。<br>AI Server标识，字符串类型，长度小于等于64，请确保全局唯一。<br>配置示例：node_0。 |
|  | device |  | 必选。<br>AI Server中的Device列表。 |
|  |  | device_id | 必选。<br>AI处理器的物理ID，即Device在AI Server上的序列号。<br>可通过执行“ls /dev/davinci*”命令获取AI处理器的物理ID。<br>例如：显示/dev/davinci0，表示AI处理器的物理ID为0。<br>取值范围：\[0，实际Device数量-1]。<br>注意：“device_id”配置项的优先级高于环境变量“ASCEND_DEVICE_ID”。 |
|  |  | device_ip | 必选。<br>AI处理器集成网卡IP，全局唯一，要求为常规IPv4或IPv6格式。<br>可以在当前AI Server执行指令“cat /etc/hccn.conf”获取网卡IP，例如：<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>查询到的address_xx即为网卡IP，address后的序号为AI处理器的物理ID，即device_id，后面的ip地址即为需要用户填入的该device对应的网卡IP。 |
|  |  | rank_id | 必选。<br>rank唯一标识，请配置为整数，从0开始配置，且全局唯一，取值范围：\[0, 总Device数量-1]。 <br>- 建议rank_id按照Device物理连接顺序进行排序，即将物理连接上较近的Device编排在一起，否则可能会对性能造成影响。<br>&nbsp;&nbsp;  例如，若device_ip按照物理连接从小到大设置，则rank_id也建议按照从小到大的顺序设置。<br>- 不同AI Server中的rank_id不支持交叉配置。<br> &nbsp;&nbsp; 正例：server 1中的rank_id集合为{0,1,2,3}，server 2中的rank_id集合为{4,5,6,7}。<br> &nbsp;&nbsp; 反例：server 1中的rank_id集合为{0,1,2,7}，server 2中的rank_id集合为{3,4,5,6}。 |

## 模板二（兼容部分已有场景，新版本不推荐使用）

```json
{
    "status":"completed",  // rank table可用标识，completed为可用
    "group_count":"1",     // group数量，建议为1
    "group_list":          // group列表
    [
        {
            "group_name":"hccl_world_group",  //group名称，建议hccl_world_group
            "instance_count":"2",             // instance实例个数，容器场景下可理解为容器个数
            "device_count":"2",               // group中的所有device数目
            "instance_list":[                 // instance实例信息列表
                {
                    "pod_name":"tf-bae41",     //instance实例名称，一般为容器名称
                    "server_id":"node_0",      //AI Server标识，String类型，请确保全局唯一
                    "devices":[                //instance实例的device列表
                        {
                            "device_id":"0",           // 处理器的物理ID
                            "device_ip":"192.168.1.8"  // 处理器的真实网卡IP
                        }
                    ]
                },
                {
                    "pod_name":"tf-tbdf1",
                    "server_id":"node_1",
                    "devices":[
                        {
                            "device_id":"1",
                            "device_ip":"192.168.1.9"  
                        }
                    ]
                }
            ]
        }
    ]
}
```

rank table配置文件说明如下所示：

| 一级配置项 | 二级配置项 | 三级配置项 | 四级配置项 | 配置说明 |
| --- | --- | --- | --- | --- |
| status |  |  |  | 必选。<br>rank table可用标识。<br><br>  - completed：表示rank table可用。<br>  - initializing：表示rank table不可用。 |
| group_count |  |  |  | 必选。<br>用户申请的group数量，建议配置为1。 |
| group_list |  |  |  | 必选。<br>Group列表。 |
|  | group_name |  |  | 可选。<br>Group名称，当group_count为1时，建议配置为hccl_world_group或者空。因为当前版本无论定义为任何值，都会创建名称为hccl_world_group的group。<br>如果通过该配置文件创建了多个group，则系统会自动将多个group合并为一个名称为“hccl_world_group”的group资源。 |
|  | instance_count |  |  | 必选。<br>和instance_list中pod_name个数保持一致，例如：容器场景下为容器实际数量。 |
|  | device_count |  |  | 必选。<br>group中设备数量。 |
|  | instance_list |  |  | 必选。<br>instance实例信息列表。 |
|  |  | pod_name |  | 必选。<br>用户自定义配置，保持instance_list内全局唯一。 |
|  |  | server_id |  | 必选。<br>AI Server标识，字符串类型，长度小于等于64，请确保全局唯一。<br>配置示例：node_0。 |
|  |  | devices |  | 必选。<br>devices信息列表。 |
|  |  |  | device_id | 必选。<br>AI处理器的物理ID，即Device在Server上的序列号。<br>可通过执行“ls /dev/davinci*”命令获取AI处理器的物理ID。<br>例如：显示/dev/davinci0，表示AI处理器的物理ID为0。<br>取值范围：\[0，实际Device数量-1]。<br>注意：“device_id”配置项的优先级高于环境变量“ASCEND_DEVICE_ID”。 |
|  |  |  | device_ip | 必选。<br>AI处理器集成网卡IP，全局唯一，要求为常规IPv4或IPv6格式。<br>可以在当前Server执行指令cat /etc/hccn.conf获取网卡IP。 |
