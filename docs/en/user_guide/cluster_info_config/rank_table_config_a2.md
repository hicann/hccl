# Rank Table Configuration Resource Information (Atlas A2 Training Products/Atlas A2 Inference Products)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:52:37.054Z pushedAt=2026-07-31T08:41:04.484Z -->

For Atlas A2 training products/Atlas A2 inference products, taking two AI servers with two devices in each AI server as an example, the rank table file configuration example is as follows:

> [!NOTE]Note
> The rank table file is in JSON format. The comments in the JSON file example shown in this section are only for better understanding. When using it in practice, delete the comments from the JSON file.

```json
{
    "status": "completed",  // Rank table availability status. "completed" indicates that the rank table is available.
    "version": "1.0",       // Rank table template version. Set to "1.0".
    "server_count": "2",    // Number of AI servers participating in training. In this example, there are two servers.
    "server_list": [
        {
            "server_id": "node_0",  // AI server identifier (string). Must be globally unique.
            "device": [             // List of devices on the AI server.
                {
                    "device_id": "0",            // Physical ID of the AI processor.
                    "device_ip": "192.168.1.8",  // Actual NIC IP address of the processor.
                    "device_port": "16667",      // Listening port of the processor's NIC.
                    "rank_id": "0"               // Rank identifier, starting from 0. Must be globally unique.
                },
                {
                    "device_id": "1",
                    "device_ip": "192.168.1.9",
                    "device_port": "16667",
                    "rank_id": "1"
                }
            ]
        },
        {
            "server_id": "node_1",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "192.168.2.8",
                    "device_port": "16667",
                    "rank_id": "2"
                },
                {
                    "device_id": "1",
                    "device_ip": "192.168.2.9",
                    "device_port": "16667",
                    "rank_id": "3"
                }
            ]
        }
    ]
}
```

The rank table configuration file is described as follows:

| Level-1 Configuration Item | Level-2 Configuration Item | Level-3 Configuration Item | Description |
| --- | --- | --- | --- |
| status |  |  | Mandatory.<br>Rank table availability status.<br>  - completed: The rank table is available.<br>  - initializing: The rank table is unavailable. |
| version |  |  | Mandatory.<br>Rank table template version. Set to 1.0. |
| server_count |  |  | Mandatory.<br>Number of AI servers participating in collective communication. |
| server_list |  |  | Mandatory.<br>List of AI servers participating in collective communication. |
|  | server_id |  | Mandatory.<br>AI server identifier (string, ≤ 64 characters). Must be globally unique.<br>Example: node_0. |
|  | device |  | Mandatory.<br>List of devices on the AI server. |
|  |  | device_id | Mandatory.<br>Physical ID of the AI processor (device serial number on the server).<br>Run `ls /dev/davinci*` to obtain the physical ID. For example, `/dev/davinci0` indicates physical ID 0.<br>Value range: [0, actual number of devices - 1].<br>Note: This setting overrides the `ASCEND_DEVICE_ID` environment variable. |
|  |  | device_ip | Optional.<br>IP address of the AI processor's integrated NIC, globally unique, in standard IPv4 or IPv6 format.<br>Note the following:<br>  - In multi-server scenarios, this field is mandatory.<br>  - In single-server scenarios, this field can be left empty.<br>Run `cat /etc/hccn.conf` on the current AI server to obtain the NIC IP address. Example:<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>The `address_xx` entry is the NIC IP address. The number following `address` corresponds to `device_id`; fill in the IP address for each device accordingly. |
|  |  | device_port | Optional.<br>Communication port of the device NIC. Value range: [1, 65535]. Ensure the specified port is not occupied by other processes. Ports in the range [1, 1023] are system-reserved and should be avoided.<br>In single-card multi-process scenarios (where multiple service processes share one NPU), it is recommended to configure this field with distinct port numbers for different processes. Otherwise, port conflicts may cause service startup failures. |
|  |  | rank_id | Mandatory.<br>Rank identifier, an integer starting from 0. Must be globally unique.<br>Value range: [0, total number of devices - 1].<br>- It is recommended to assign `rank_id` in order of physical device connectivity—group devices that are closer in topology to reduce potential performance impact.<br>&nbsp;&nbsp;  For example, if `device_ip` is set in ascending order of physical connections, `rank_id` should also be set in ascending order.<br>- Cross-server `rank_id` interleaving is not supported.<br>&nbsp;&nbsp;  Correct example: Server 1 uses {0, 1, 2, 3}, Server 2 uses {4, 5, 6, 7}.<br>&nbsp;&nbsp;  Incorrect example: Server 1 uses {0, 1, 2, 7}, Server 2 uses {3, 4, 5, 6}. |
