# Rank Table Configuration Resource Information (Atlas A3 Training Products/Atlas A3 Inference Products)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-02T07:34:35.150Z pushedAt=2026-09-03T10:46:25.931Z -->

For the Atlas A3 training products/Atlas A3 inference products, cluster training supports SuperPoD mode networking and typical networking. Note that for the Atlas A3 training products/Atlas A3 inference products, each NPU contains two devices (i.e., two Dies), and each device is a rank.

> [!NOTE]Note
> The rank table file is in JSON format. The comments in the JSON file examples shown in this section are provided only for ease of understanding. In actual use, delete the comments from the JSON file.

## SuperPoD Mode Networking

The following configuration example uses two SuperPoDs, each containing two AI servers, with each AI server having four devices:

```json
{
    "status": "completed",         // rank table availability flag; "completed" indicates available
    "version": "1.2",              // rank table template version information; for SuperPoD mode networking, configure as: 1.2
    "server_count":"4",            // number of AI servers participating in training
    "server_list": [
        {
            "server_id": "node_0",     // AI server identifier, string type; ensure it is globally unique
            "host_ip":"172.16.0.100",  // host IP address of the AI server
            "device": [
                {"device_id": "0","super_device_id":"0","device_ip": "192.168.1.6","device_port":"16666","backup_device_ip":"192.168.1.7","backup_device_port":"16667","host_port":"16665","rank_id": "0"}, // device_id is the physical ID of the processor; super_device_id is the physical ID of the processor in the SuperPoD system; device_ip is the actual NIC IP of the processor; device_port is the NIC communication port of the processor; backup_device_ip is the backup IP used when the operator re-execution feature for inter-SuperPoD communication is enabled; host_port is the communication port of the host NIC; rank_id is the rank identifier, configured starting from 0
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
            "super_pod_id": "0",          // unique identifier of the SuperPoD
            "server_list": [              // list of AI servers in the SuperPoD
                {"server_id": "node_0"},  // server_id is the server identifier, corresponding to the server_id in "server_list"
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

The rank table configuration file is described as follows:

| Level-1 Configuration Item | Level-2 Configuration Item | Level-3 Configuration Item | Description |
| --- | --- | --- | --- |
| status |  |  | Mandatory.<br>rank table availability flag.<br>  - completed: indicates that the rank table is available.<br>  - initializing: indicates that the rank table is unavailable. |
| version |  |  | Mandatory.<br>rank table template version information.<br>For SuperPoD mode networking, configure as: 1.2. |
| server_count |  |  | Optional.<br>Number of AI servers participating in collective communication. |
| server_list |  |  | Mandatory.<br>List of AI servers participating in collective communication. |
|  | server_id |  | Mandatory.<br>AI server identifier, string type, with a length less than or equal to 64. Ensure that it is globally unique.<br>Configuration example: node_0. |
|  | host_ip |  | Optional.<br>Host IP address of the AI server, required to be in standard IPv4 format.<br>In the scenario where the HCCL re-execution feature is enabled, this field must be configured; otherwise, re-execution becomes invalid and the process runs without re-execution.<br>The re-execution feature is disabled by default. For details, see the environment variable [HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md). |
|  | device |  | Mandatory.<br>Device list. |
|  |  | device_id | Mandatory.<br>Physical ID of the AI processor, that is, the serial number of the device on the AI server.<br>You can obtain the physical ID of the AI processor by running the "ls /dev/davinci*" command.<br>For example, if /dev/davinci0 is displayed, the physical ID of the AI processor is 0.<br>Value range: \[0, actual number of devices - 1].<br>Note: The "device_id" configuration item has a higher priority than the environment variable "ASCEND_DEVICE_ID". |
|  |  | super_device_id | Optional (if this field is not configured, "AI server mode" is used).<br>Physical ID of the AI processor in the SuperPoD system, which is the unique identifier of the NPU in the SuperPoD system.<br>Developers can query it using the npu-smi command. The command example is as follows:<br>npu-smi info -t spod-info -i id -c chip_id<br><br>  - id: device ID. The NPU ID queried via the npu-smi info -l command is the device ID.<br>  - chip_id: chip ID. The Chip ID queried via the npu-smi info -m command is the chip ID.<br><br>The "SDID" in the command output is the unique identifier of the NPU in the SuperPoD system. |
|  |  | device_ip | Optional.<br>Integrated NIC IP of the AI processor, globally unique, required to be in standard IPv4 or IPv6 format.<br>Note that:<br>  1. When the networking contains multiple SuperPoDs, device_ip must be configured.<br>  2. If the networking contains only one SuperPoD, device_ip must be configured in the following scenarios and can be left unconfigured in other scenarios. When RDMA communication is used within the SuperPoD (that is, when the environment variable HCCL_INTER_HCCS_DISABLE is configured as TRUE, disabling the HCCS function), device_ip must be configured.<br>You can run the command cat /etc/hccn.conf on the current AI server to obtain the NIC IP. For example:<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>The queried address_xx is the NIC IP. The number after address is the physical ID of the AI processor, that is, device_id, and the IP address after it is the NIC IP corresponding to the device that the user needs to fill in. |
|  |  | device_port | Optional.<br>Communication port of the device NIC. The value range is \[1,65535]. Ensure that the specified port is not occupied by other processes. Note that \[1,1023] are system-reserved ports and should be avoided.<br>In the single-card multi-process service scenario, it is recommended to configure this field, and different service processes need to set different port numbers; otherwise, the service may fail to run due to port conflicts. |
|  |  | backup_device_ip | Optional.<br>When the operator re-execution feature is enabled for inter-SuperPoD communication, if a device NIC fault (RDMA link fault) occurs, this parameter can be used to specify the NIC of another die in the same NPU as the backup device NIC, improving the success rate of operator re-execution. This communication method that uses the backup device NIC is called borrowed-track communication.<br>"backup_device_ip" is in standard IPv4 or IPv6 format. For the query method, see the configuration description of "device_ip".<br>Note that:<br>  1. The devices corresponding to "backup_device_ip" and "device_ip" must belong to the same NPU, that is, only the device NICs corresponding to Die0 and Die1 in the same NPU can serve as backups for each other.<br>  2. This configuration takes effect only when the communication operator expansion mode is AI_CPU and the operator re-execution feature for inter-SuperPoD communication is enabled, that is: export HCCL_OP_EXPANSION_MODE="AI_CPU"<br>export HCCL_OP_RETRY_ENABLE="L1:1,L2:1"<br>L2 indicates that the physical range of the communicator is the inter-SuperPoD communicator. A value of 1 indicates that the communication operator re-execution feature is enabled.<br>  3. To ensure that the borrowed-track function runs properly, the following conditions must be met:<br>  - The communication link of the backup NIC is normal.<br>  - The devices that serve as backups for each other are both within the service-visible range. For example, NPU1 contains two dies, Device0 and Device1, which serve as backups for each other. If the environment variable ASCEND_RT_VISIBLE_DEVICES specifies that only Device0 is visible to the service and Device1 is not, the borrowed-track function cannot be executed.<br>  4. If borrowed-track communication occurs during the communication process (for example, the Die0 NIC of an NPU fails and the backup Die1 NIC is enabled), the traffic of the original Die0 NIC is also sent and received through the Die1 NIC, increasing the traffic of Die1. The overall performance decreases due to the halved physical bandwidth and port conflicts.<br>  5. In the borrowed-track scenario, if the Die0 NIC of NPU0 fails, it switches to its backup NIC Die1. Because communication between two NPUs requires both the local end and the peer end to switch to the backup NIC simultaneously, NPU1 also switches from Die0 to Die1, as shown in [Figure 1](#figure1). However, if a communication task already exists between Die0 and Die1, the borrowed-track function cannot be executed.<br>  6. When the borrowed-track communication function is enabled, it is recommended to assign the two dies of an NPU to the same training or inference task. If the two dies of the same NPU are assigned to two different training or inference tasks, when one task fails, it borrows the NIC of the other task, causing a certain degree of performance degradation for both tasks.<br>  7. The same NPU supports only one borrowed-track switchover, and switching back is not supported. As shown in [Figure 2](#figure2), in "Illustration 1", the communication link between NPU0 and NPU1 fails, the backup link is enabled, borrowed-track communication occurs, and communication proceeds normally. If a fault as shown in "Illustration 2" occurs again, borrowed-track communication is no longer supported and an error is reported to exit. |
|  |  | backup_device_port | Optional.<br>Communication port of the backup device NIC. The value range is \[1,65535]. Ensure that the specified port is not occupied by other processes. Note that \[1,1023] are system-reserved ports and should be avoided.<br>If the borrowed-track communication function is enabled and the service is in the single-card multi-process scenario, it is recommended to configure this field, and different service processes need to set different port numbers; otherwise, the service may fail to run due to port conflicts.<br>Note: The communication port numbers configured for the same device NIC when it serves as the primary NIC and the backup NIC cannot be the same. |
|  |  | host_port | Optional.<br>Communication port of the host NIC. The value range is \[1,65535]. The host_port corresponding to each device in the same AI server should be different, and ensure that the specified port is not occupied by other processes. Note that \[1,1023] are system-reserved ports and should be avoided.<br>If the HCCL re-execution feature is enabled through the environment variable [HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md) and the service is in the single-card multi-process scenario (that is, multiple service processes share one NPU), it is recommended to configure this field, and different service processes need to set different port numbers; otherwise, the service may fail to run due to port conflicts. |
|  |  | rank_id | Mandatory.<br>Unique rank identifier. Configure it as an integer, starting from 0, and ensure it is globally unique. Value range: \[0, total number of devices - 1].<br>- It is recommended to sort rank_id according to the physical connection order of devices, that is, to arrange devices that are physically closer together; otherwise, performance may be affected.<br>&nbsp;&nbsp;  For example, if device_ip is set in ascending order of physical connection, it is recommended to set rank_id in ascending order as well.<br>- Cross configuration of rank_id across different AI servers is not supported.<br> &nbsp;&nbsp; Positive example: the rank_id set of server 1 is {0,1,2,3}, and the rank_id set of server 2 is {4,5,6,7}.<br> &nbsp;&nbsp; Negative example: the rank_id set of server 1 is {0,1,2,7}, and the rank_id set of server 2 is {3,4,5,6}. |
| super_pod_list |  |  | Optional (if this field is not configured, "AI server mode" is used).<br>List of SuperPoDs participating in collective communication. |
|  | super_pod_id |  | If "super_pod_list" is configured, this field is mandatory.<br>Unique identifier of the SuperPoD, globally unique. The following two configuration methods are supported:<br>  - Configure it as the physical ID of the SuperPoD, which can be queried using the npu-smi tool. The command example is as follows: npu-smi info -t spod-info -i id -c chip_id<br>id: device ID. The NPU ID queried via the npu-smi info -l command is the device ID. chip_id: chip ID. The Chip ID queried via the npu-smi info -m command is the chip ID.<br>The "Super Pod ID" in the command output is the physical ID of the SuperPoD.<br>  - id: device ID. The NPU ID queried via the npu-smi info -l command is the device ID.<br>  - chip_id: chip ID. The Chip ID queried via the npu-smi info -m command is the chip ID.<br>  - User-defined number, in string format, which must be globally unique. In the scenario of a user-defined ID, the user can divide one physical SuperPoD into multiple smaller logical SuperPoDs. For example, if a physical SuperPoD has 8 AI server nodes, the user can use the first 4 AI server nodes as a smaller SuperPoD numbered super_pod_1, and the last 4 AI server nodes as another smaller SuperPoD numbered super_pod_2. |
|  | server_list |  | Mandatory.<br>List of AI servers in the SuperPoD. |
|  |  | server_id | Mandatory.<br>Server identifier, string type, corresponding to the server_id in "server_list".<br>Configuration example: node_0. |

> [!NOTE]Note
> If there are multiple SuperPoDs in the network, configure the AI server information belonging to the same SuperPoD together. Assume there are two SuperPoDs with identifiers "0" and "1". Configure the AI server information in "0" first, and then configure the AI server information in "1". Cross-configuration of AI server information between "0" and "1" is not supported.

**Figure 1**  Borrowed-Track Communication Switchover Example<a id="figure1"></a>  
![](figures/borrow_comm_switch_example.png)

**Figure 2** Example of a single NPU supporting only one borrowed-track communication<a id="figure2"></a>  
![](figures/npu_single_borrow_example.png)

## Typical Cluster Networking (AI Server Mode)

The following is a rank table file configuration example with two AI servers, each containing two Devices:

```json
{
    "status":"completed",   // rank table availability flag; "completed" indicates available
    "version":"1.0",        // rank table template version information; for typical cluster networking, set to 1.0
    "server_count":"2",     //number of AI servers participating in training; in this example, there are two AI servers
    "server_list":
    [
        {
            "server_id":"node_0",       // AI server identifier, string type; ensure it is globally unique
            "host_ip":"172.16.0.110",   // Host IP address of the AI server
            "device":[   // list of devices in the AI server
                {
                    "device_id":"0",              // physical ID of the processor
                    "device_ip":"192.168.1.8",    // actual NIC IP of the processor
                    "device_port":"16667",        // NIC communication port of the processor
                    "host_port":"16666",          // communication port of the Host NIC
                    "rank_id":"0"                 // Identifier of the rank, configured starting from 0.
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

The following table describes the rank table configuration file:

| Level-1 Configuration Item | Level-2 Configuration Item | Level-3 Configuration Item | Description |
| --- | --- | --- | --- |
| status |  |  | Mandatory.<br>Availability flag of the rank table.<br>  - completed: indicates that the rank table is available.<br>  - initializing: indicates that the rank table is unavailable. |
| version |  |  | Mandatory.<br>Version information of the rank table template.<br>For typical cluster networking, configure it as 1.0. |
| server_count |  |  | Mandatory.<br>Number of AI servers participating in collective communication. |
| server_list |  |  | Mandatory.<br>List of AI servers participating in collective communication. |
|  | server_id |  | Mandatory.<br>Identifier of the AI server, string type, with a length less than or equal to 64. Ensure that it is globally unique.<br>Configuration example: node_0. |
|  | host_ip |  | Optional.<br>Host IP address of the AI server, required to be in standard IPv4 format.<br>In the scenario where the HCCL re-execution feature is enabled, this field must be configured; otherwise, re-execution becomes invalid and the process runs without re-execution.<br>The re-execution feature is disabled by default. For details, see the environment variable [HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md). |
|  | device |  | Mandatory.<br>List of devices in the AI server. |
|  |  | device_id | Mandatory.<br>Physical ID of the AI processor, that is, the serial number of the device on the AI server.<br>You can obtain the physical ID of the AI processor by running the "ls /dev/davinci*" command.<br>For example, if /dev/davinci0 is displayed, the physical ID of the AI processor is 0.<br>Value range: \[0, actual number of devices - 1].<br>Note: The "device_id" configuration item takes precedence over the environment variable "ASCEND_DEVICE_ID". |
|  |  | device_ip | Optional.<br>IP address of the integrated NIC of the AI processor, globally unique, required to be in standard IPv4 or IPv6 format.<br>Note that:<br>  - In the multi-machine scenario, device_ip must be configured.<br>  - In the single-machine scenario, device_ip can be left unconfigured.<br>You can run the cat /etc/hccn.conf command on the current AI server to obtain the NIC IP, for example:<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>The queried address_xx is the NIC IP. The number after address is the physical ID of the AI processor, that is, device_id, and the IP address after it is the NIC IP corresponding to the device that the user needs to fill in. |
|  |  | device_port | Optional.<br>Communication port of the device NIC, with a value range of \[1,65535]. Ensure that the specified port is not occupied by other processes. Note that \[1,1023] are system-reserved ports, and avoid using these ports.<br>In the single-card multi-process service scenario, it is recommended to configure this field, and different service processes need to set different port numbers; otherwise, the service may fail to run due to port conflicts. |
|  |  | host_port | Optional.<br>Communication port of the host NIC, with a value range of \[1,65535]. The host_port corresponding to each device in the same AI server should be different, and ensure that the specified port is not occupied by other processes. Note that \[1,1023] are system-reserved ports, and avoid using these ports.<br>If the HCCL re-execution feature is enabled through the environment variable [HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md), and the service is in the single-card multi-process scenario (that is, multiple service processes share one NPU at the same time), it is recommended to configure this field, and different service processes need to set different port numbers; otherwise, the service may fail to run due to port conflicts. |
|  |  | rank_id | Mandatory.<br>Unique identifier of the rank. Configure it as an integer, starting from 0, and ensure that it is globally unique. Value range: \[0, total number of devices - 1].<br>- It is recommended to sort rank_id according to the physical connection order of devices, that is, arrange devices that are physically closer together; otherwise, performance may be affected.<br>&nbsp;&nbsp;  For example, if device_ip is set in ascending order of physical connection, it is also recommended to set rank_id in ascending order.<br>- Cross configuration of rank_id in different AI servers is not supported.<br> &nbsp;&nbsp; Positive example: the rank_id set in server 1 is {0,1,2,3}, and the rank_id set in server 2 is {4,5,6,7}.<br> &nbsp;&nbsp; Negative example: the rank_id set in server 1 is {0,1,2,7}, and the rank_id set in server 2 is {3,4,5,6}. |
