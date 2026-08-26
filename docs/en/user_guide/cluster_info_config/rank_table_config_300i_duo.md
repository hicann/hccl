# Rank Table Configuration Resource Information (Atlas 300I Duo Inference Card)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-30T03:50:26.350Z pushedAt=2026-07-31T08:41:39.604Z -->

For the Atlas 300I Duo inference card, taking two AI servers with two devices in each AI server as an example, the rank table file configuration example is as follows:

> [!NOTE]
> The rank table file is in JSON format. The comments in the JSON file examples shown in this section are provided only for better understanding. When using the file in practice, please remove the comments from the JSON file.

```json
{
    "status":"completed",   // Rank table availability flag. 'completed' indicates available.
    "version":"1.0",        // Rank table template version. Set to: 1.0
    "server_count":"2",     // Number of AI servers participating in training. In this example, there are two AI servers.
    "server_list":
    [
        {
            "server_id":"node_0",  //AI server identifier, of type String. Ensure that it is globally unique.
            "device":[             // List of devices in the AI server.
                {
                    "device_id":"0",   // Physical ID of the processor.
                    "device_ip":"192.168.1.8",   // Actual NIC IP address of the processor.
                    "rank_id":"0"                // Rank identifier, starting from 0. Ensure that it is globally unique.
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

| Level-1 | Level-2 | Level-3 | Description |
|---------|---------|---------|-------------|
| status | | | Mandatory. Rank table availability status. <br> - `completed`: The rank table is available. <br> - `initializing`: The rank table is unavailable. |
| version | | | Mandatory. Rank table template version. Set to `1.0`. |
| server_count | | | Mandatory. Number of AI servers participating in collective communication. |
| server_list | | | Mandatory. List of AI servers participating in collective communication. |
| | server_id | | Mandatory. AI server identifier (string, ≤ 64 characters). Must be globally unique. Example: `node_0`. |
| | device | | Mandatory. List of devices on the AI server. |
| | | device_id | Mandatory. Physical ID of the AI processor (device serial number on the server). <br> Run `ls /dev/davinci*` to obtain the physical ID. Example: `/dev/davinci0` indicates physical ID `0`. <br> Value range: `[0, number_of_devices - 1]`. <br> **Note**: This setting overrides the `ASCEND_DEVICE_ID` environment variable. |
| | | device_ip | Mandatory. IP address of the AI processor's integrated NIC, globally unique. Supports both IPv4 and IPv6. <br> Run `cat /etc/hccn.conf` on the AI server to obtain the NIC IP address, e.g.: <br> `address_0=xx.xx.xx.xx` <br> `netmask_0=xx.xx.xx.xx` <br> `netdetect_0=xx.xx.xx.xx` <br> The `address_xx` entry gives the NIC IP. The number following `address` (e.g., `0`) corresponds to `device_id`. Fill in the IP address for each device accordingly. |
| | | rank_id | Mandatory. Rank identifier, an integer starting from `0`, globally unique. <br> Value range: `[0, total_devices - 1]`. <br> - It is recommended to assign `rank_id` in order of physical device connectivity—group devices that are closer in topology to reduce potential performance impact. <br> &nbsp;&nbsp; For example, if `device_ip` is assigned in ascending order of physical connections, `rank_id` should follow the same order. <br> - Cross-server rank_id interleaving is not allowed. <br> &nbsp;&nbsp; **Correct**: Server 1 uses rank IDs `{0,1,2,3}`, Server 2 uses `{4,5,6,7}`. <br> &nbsp;&nbsp; **Incorrect**: Server 1 uses `{0,1,2,7}`, Server 2 uses `{3,4,5,6}`. |
