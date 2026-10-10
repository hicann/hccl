# Rank Table Configuration Resource Information (Atlas Training Products)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-02T07:33:03.954Z pushedAt=2026-09-03T08:56:33.612Z -->

For Atlas training products, two configuration templates are supported for configuring AI processor information for training in the rank table file. Template 1 is recommended for new scenarios, and template 2 is used for compatibility with certain existing scenarios.

> [!NOTE]Note
> The rank table file is in JSON format. The comments in the JSON file examples shown in this section are provided only for ease of understanding. In actual use, delete the comments from the JSON file.

## Template 1 (Recommended)

The following is a rank table file configuration example with two AI servers, each containing two Devices:

```json
{
    "status":"completed",   // Rank table availability flag. "completed" indicates that the rank table is available.
    "version":"1.0",        // Rank table template version. Set to 1.0.
    "server_count":"2",     // Number of AI servers participating in training. In this example, there are two AI servers.
    "server_list":
    [
        {
            "server_id":"node_0",  //AI server identifier, of the String type. Ensure that it is globally unique.
            "device":[             // List of devices in the AI server.
                {
                    "device_id":"0",   // Physical ID of the processor.
                    "device_ip":"192.168.1.8",   // Real NIC IP address of the processor.
                    "rank_id":"0"                // Rank identifier. Configure it starting from 0. Ensure that it is globally unique.
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

The rank table configuration file is described as follows:

| First-Level Configuration Item | Second-Level Configuration Item | Third-Level Configuration Item | Configuration Description |
| --- | --- | --- | --- |
| status |  |  | Mandatory.<br>Rank table availability flag.<br>  - completed: indicates that the rank table is available.<br>  - initializing: indicates that the rank table is unavailable. |
| version |  |  | Mandatory.<br>Rank table template version.<br>Set to 1.0. |
| server_count |  |  | Mandatory.<br>Number of AI servers participating in collective communication. |
| server_list |  |  | Mandatory.<br>List of AI servers participating in collective communication. |
|  | server_id |  | Mandatory.<br>AI server identifier, of the string type, with a length less than or equal to 64. Ensure that it is globally unique.<br>Example: node_0. |
|  | device |  | Mandatory.<br>List of devices in the AI server. |
|  |  | device_id | Mandatory.<br>Physical ID of the AI processor, that is, the serial number of the device on the AI server.<br>You can obtain the physical ID of the AI processor by running the "ls /dev/davinci*" command.<br>For example, if /dev/davinci0 is displayed, the physical ID of the AI processor is 0.<br>Value range: \[0, actual number of devices - 1].<br>Note: The "device_id" configuration item takes precedence over the "ASCEND_DEVICE_ID" environment variable. |
|  |  | device_ip | Mandatory.<br>IP address of the integrated NIC of the AI processor. It must be globally unique and in the standard IPv4 or IPv6 format.<br>You can run the "cat /etc/hccn.conf" command on the current AI server to obtain the NIC IP address. For example:<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>The queried address_xx is the NIC IP address. The number after "address" is the physical ID of the AI processor, that is, device_id, and the IP address after it is the NIC IP address of the device that you need to fill in. |
|  |  | rank_id | Mandatory.<br>Unique rank identifier. Configure it as an integer starting from 0, and ensure that it is globally unique. Value range: \[0, total number of devices - 1]. <br>- It is recommended that rank_id be sorted by the physical connection order of devices, that is, devices that are physically closer are arranged together. Otherwise, performance may be affected.<br>&nbsp;&nbsp;  For example, if device_ip is set in ascending order of physical connection, it is recommended that rank_id also be set in ascending order.<br>- Cross configuration of rank_id across different AI servers is not supported.<br> &nbsp;&nbsp; Positive example: the rank_id set in server 1 is set to {0,1,2,3}, and the rank_id set in server 2 is set to {4,5,6,7}.<br> &nbsp;&nbsp; Negative example: the rank_id set in server 1 is set to {0,1,2,7}, and the rank_id set in server 2 is set to {3,4,5,6}. |

## Template 2 (Compatible With Some Existing Scenarios, Not Recommended for New Versions)

```json
{
    "status":"completed",  // Rank table availability flag. "completed" indicates that the rank table is available.
    "group_count":"1",     // Number of groups. The recommended value is 1.
    "group_list":          // Group list.
    [
        {
            "group_name":"hccl_world_group",  //Group name. The recommended value is hccl_world_group.
            "instance_count":"2",             // Number of instances. In container scenarios, this can be understood as the number of containers.
            "device_count":"2",               // Number of all devices in the group.
            "instance_list":[                 // List of instance information.
                {
                    "pod_name":"tf-bae41",     //Instance name, generally the container name.
                    "server_id":"node_0",      //AI server identifier, of the String type. Ensure that it is globally unique.
                    "devices":[                //List of devices in the instance.
                        {
                            "device_id":"0",           // Physical ID of the processor.
                            "device_ip":"192.168.1.8"  // Actual NIC IP address of the processor
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

The rank table configuration file is described as follows:

| Level-1 Configuration Item | Level-2 Configuration Item | Level-3 Configuration Item | Level-4 Configuration Item | Configuration Instructions |
| --- | --- | --- | --- | --- |
| status |  |  |  | Mandatory.<br>Rank table availability status.<br><br>  - completed: The rank table is available.<br>  - initializing: The rank table is unavailable. |
| group_count |  |  |  | Mandatory.<br>Number of groups requested by the user. Recommended value: 1. |
| group_list |  |  |  | Mandatory.<br>List of groups. |
|  | group_name |  |  | Optional.<br>Group name. When group_count is 1, the recommended value is `hccl_world_group` or leave it empty. Regardless of the specified value, a group named `hccl_world_group` is created in the current version.<br>If multiple groups are defined in this configuration file, the system automatically merges them into a single group resource named `hccl_world_group`. |
|  | instance_count |  |  | Mandatory.<br>Must match the number of pod_name entries in instance_list. For example, in container scenarios, this is the actual number of containers. |
|  | device_count |  |  | Mandatory.<br>Number of devices in the group. |
|  | instance_list |  |  | Mandatory.<br>List of instance information. |
|  |  | pod_name |  | Mandatory.<br>User-defined configuration. Must be globally unique within instance_list. |
|  |  | server_id |  | Mandatory.<br>AI server identifier (string, ≤ 64 characters). Must be globally unique.<br>Example: node_0. |
|  |  | devices |  | Mandatory.<br>List of device information. |
|  |  |  | device_id | Mandatory.<br>Physical ID of the AI processor (device serial number on the server).<br>Run `ls /dev/davinci*` to obtain the physical ID. For example, `/dev/davinci0` indicates physical ID 0.<br>Value range: [0, actual number of devices - 1].<br>Note: This setting overrides the `ASCEND_DEVICE_ID` environment variable. |
|  |  |  | device_ip | Mandatory.<br>IP address of the AI processor's integrated NIC, globally unique. Supports both IPv4 and IPv6.<br>Run `cat /etc/hccn.conf` on the current server to obtain the NIC IP address. |
