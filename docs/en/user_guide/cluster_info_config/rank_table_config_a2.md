# Rank Table Configuration Resource Information (Atlas A2 Training Products/Atlas A2 Inference Products)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-02T07:33:04.663Z pushedAt=2026-09-03T09:05:07.298Z -->

For Atlas A2 training products/Atlas A2 inference products, taking two AI servers with two devices in each AI server as an example, the rank table file configuration example is as follows:

> [!NOTE]Note
> The rank table file is in JSON format. The comments in the JSON file examples shown in this section are provided only for ease of understanding. In actual use, delete the comments from the JSON file.

```json
{
    "status":"completed",  // Rank table availability flag. completed indicates that the rank table is available.
    "version":"1.0",       // Rank table template version. Set this parameter to 1.0.
    "server_count":"2",    // Number of AI servers participating in training. In this example, there are two AI servers.
    "server_list":
    [
        {
            "server_id":"node_0",  //AI server ID, of the String type. Ensure that it is globally unique.
            "device":[             // List of devices in the AI server
                {
                    "device_id":"0",            // Physical ID of the processor
                    "device_ip":"192.168.1.8",  // Actual NIC IP address of the processor
                    "device_port":"16667",      // NIC listening port of the processor
                    "rank_id":"0"               // Rank ID, configured starting from 0. Ensure that it is globally unique.
                },
                {
                    "device_id":"1",
                    "device_ip":"192.168.1.9", 
                    "device_port":"16667",
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
                    "device_port":"16667",
                    "rank_id":"2"
                },
                {
                    "device_id":"1",
                    "device_ip":"192.168.2.9", 
                    "device_port":"16667",
                    "rank_id":"3"
                }
            ]
        }
    ]
}
```

The rank table configuration file is described as follows:

| Level-1 Configuration Item | Level-2 Configuration Item | Level-3 Configuration Item | Description |
| --- | --- | --- | --- |
| status |  |  | Mandatory.<br>Rank table availability flag.<br>  - completed: indicates that the rank table is available.<br>  - initializing: indicates that the rank table is unavailable. |
| version |  |  | Mandatory.<br>Rank table template version.<br>Set this parameter to 1.0. |
| server_count |  |  | Mandatory.<br>Number of AI servers participating in collective communication. |
| server_list |  |  | Mandatory.<br>List of AI servers participating in collective communication. |
|  | server_id |  | Mandatory.<br>AI server ID, of the string type, with a maximum length of 64. Ensure that it is globally unique.<br>Example: node_0. |
|  | device |  | Mandatory.<br>List of devices in the AI server. |
|  |  | device_id | Mandatory.<br>Physical ID of the AI processor, that is, the serial number of the device on the AI server.<br>You can run the "ls /dev/davinci*" command to obtain the physical ID of the AI processor.<br>For example, if /dev/davinci0 is displayed, the physical ID of the AI processor is 0.<br>Value range: \[0, actual number of devices - 1].<br>Note: The "device_id" configuration item takes precedence over the "ASCEND_DEVICE_ID" environment variable. |
|  |  | device_ip | Optional.<br>IP address of the integrated NIC on the AI processor. It must be globally unique and in standard IPv4 or IPv6 format.<br>Note the following:<br>  - In a multi-server scenario, device_ip must be configured.<br>  - In a single-server scenario, device_ip can be left unconfigured.<br>You can run the cat /etc/hccn.conf command on the current AI server to obtain the NIC IP address. Example:<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>The queried address_xx is the NIC IP address. The number after address is the physical ID of the AI processor, that is, device_id, and the IP address after it is the NIC IP address of the corresponding device that you need to enter. |
|  |  | device_port | Optional.<br>Communication port of the device NIC. The value range is \[1,65535]. Ensure that the specified port is not occupied by other processes. Note that \[1,1023] are system-reserved ports and should be avoided.<br>In a single-card multi-process scenario (that is, multiple service processes share one NPU), you are advised to configure this field and set different port numbers for different service processes. Otherwise, the service may fail to run due to port conflicts. |
|  |  | rank_id | Mandatory.<br>Unique rank ID. Configure it as an integer starting from 0, and ensure that it is globally unique. Value range: \[0, total number of devices - 1].<br>- It is recommended that `rank_id` be assigned according to the order of physical device connections, grouping devices with closer physical connections together. Otherwise, performance may be impacted. <br>&nbsp;&nbsp;  For example, if device_ip is set in ascending order of physical connections, rank_id is also recommended to be set in ascending order.<br>- Cross configuration of rank_id across different AI servers is not supported.<br> &nbsp;&nbsp; Positive example: the rank_id set in server 1 is {0,1,2,3}, and the rank_id set in server 2 is {4,5,6,7}.<br> &nbsp;&nbsp; Negative example: the rank_id set in server 1 is {0,1,2,7}, and the rank_id set in server 2 is {3,4,5,6}. |
