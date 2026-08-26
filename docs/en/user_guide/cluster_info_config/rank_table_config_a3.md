# Rank Table Configuration Resource Information (Atlas A3 Training Products/Atlas A3 Inference Products)

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:53:01.298Z pushedAt=2026-08-14T06:32:18.518Z -->

For the Atlas A3 training products/Atlas A3 inference products, cluster training supports SuperPoD mode networking and typical networking. Note that for the Atlas A3 training products/Atlas A3 inference products, each NPU contains two devices (i.e., two Dies), and each device is a rank.

> [!NOTE]Note
> The rank table file is in JSON format. The comments in the JSON file examples shown in this section are provided only for ease of understanding. In actual use, delete the comments from the JSON file.

## SuperPoD Mode Networking

The following configuration example uses two SuperPoDs, each containing two AI servers, with each AI server having four devices:

```json
{
    "status": "completed",         // Rank table availability status. "completed" indicates that the rank table is available.
    "version": "1.2",              // Rank table template version. For SuperPoD mode networking, set to "1.2".
    "server_count": "4",           // Number of AI servers participating in training.
    "server_list": [
        {
            "server_id": "node_0",     // AI server identifier (string). Must be globally unique.
            "host_ip": "172.16.0.100", // Host IP address of the AI server.
            "device": [
                {"device_id": "0", "super_device_id": "0", "device_ip": "192.168.1.6", "device_port": "16666", "backup_device_ip": "192.168.1.7", "backup_device_port": "16667", "host_port": "16665", "rank_id": "0"}, // device_id: physical ID of the processor; super_device_id: physical ID of the processor in the SuperPoD system; device_ip: actual NIC IP of the processor; device_port: NIC communication port of the processor; backup_device_ip: backup IP used when the inter-super-node operator re-execution feature is enabled; host_port: communication port of the host NIC; rank_id: rank identifier, starting from 0.
                {"device_id": "1", "super_device_id": "1", "device_ip": "192.168.1.7", "device_port": "16666", "backup_device_ip": "192.168.1.6", "backup_device_port": "16667", "host_port": "16666", "rank_id": "1"},
                {"device_id": "2", "super_device_id": "2", "device_ip": "192.168.1.8", "device_port": "16668", "backup_device_ip": "192.168.1.9", "backup_device_port": "16670", "host_port": "16667", "rank_id": "2"},
                {"device_id": "3", "super_device_id": "3", "device_ip": "192.168.1.9", "device_port": "16669", "backup_device_ip": "192.168.1.8", "backup_device_port": "16667", "host_port": "16668", "rank_id": "3"}]
        },
        {
            "server_id": "node_1",
            "host_ip": "172.16.0.101",
            "device": [
                {"device_id": "0", "super_device_id": "4", "device_ip": "192.168.2.6", "device_port": "16666", "backup_device_ip": "192.168.2.7", "backup_device_port": "16667", "host_port": "16665", "rank_id": "4"},
                {"device_id": "1", "super_device_id": "5", "device_ip": "192.168.2.7", "device_port": "16666", "backup_device_ip": "192.168.2.6", "backup_device_port": "16667", "host_port": "16666", "rank_id": "5"},
                {"device_id": "2", "super_device_id": "6", "device_ip": "192.168.2.8", "device_port": "16668", "backup_device_ip": "192.168.2.9", "backup_device_port": "16670", "host_port": "16667", "rank_id": "6"},
                {"device_id": "3", "super_device_id": "7", "device_ip": "192.168.2.9", "device_port": "16669", "backup_device_ip": "192.168.2.8", "backup_device_port": "16667", "host_port": "16668", "rank_id": "7"}]
        },
        {
            "server_id": "node_2",
            "host_ip": "172.16.0.102",
            "device": [
                {"device_id": "0", "super_device_id": "0", "device_ip": "192.168.3.6", "device_port": "16666", "backup_device_ip": "192.168.3.7", "backup_device_port": "16667", "host_port": "16665", "rank_id": "8"},
                {"device_id": "1", "super_device_id": "1", "device_ip": "192.168.3.7", "device_port": "16666", "backup_device_ip": "192.168.3.6", "backup_device_port": "16667", "host_port": "16666", "rank_id": "9"},
                {"device_id": "2", "super_device_id": "2", "device_ip": "192.168.3.8", "device_port": "16668", "backup_device_ip": "192.168.3.9", "backup_device_port": "16670", "host_port": "16667", "rank_id": "10"},
                {"device_id": "3", "super_device_id": "3", "device_ip": "192.168.3.9", "device_port": "16669", "backup_device_ip": "192.168.3.8", "backup_device_port": "16667", "host_port": "16668", "rank_id": "11"}]
        },
        {
            "server_id": "node_3",
            "host_ip": "172.16.0.103",
            "device": [
                {"device_id": "0", "super_device_id": "4", "device_ip": "192.168.4.6", "device_port": "16666", "backup_device_ip": "192.168.4.7", "backup_device_port": "16667", "host_port": "16665", "rank_id": "12"},
                {"device_id": "1", "super_device_id": "5", "device_ip": "192.168.4.7", "device_port": "16666", "backup_device_ip": "192.168.4.6", "backup_device_port": "16667", "host_port": "16666", "rank_id": "13"},
                {"device_id": "2", "super_device_id": "6", "device_ip": "192.168.4.8", "device_port": "16668", "backup_device_ip": "192.168.4.9", "backup_device_port": "16670", "host_port": "16667", "rank_id": "14"},
                {"device_id": "3", "super_device_id": "7", "device_ip": "192.168.4.9", "device_port": "16669", "backup_device_ip": "192.168.4.8", "backup_device_port": "16667", "host_port": "16668", "rank_id": "15"}]
        }
    ],
    "super_pod_list": [
        {
            "super_pod_id": "0",          // Unique identifier of the SuperPoD.
            "server_list": [              // List of AI servers in the SuperPoD.
                {"server_id": "node_0"},  // server_id corresponds to the server_id in "server_list".
                {"server_id": "node_1"}]
        },
        {
            "super_pod_id": "1",
            "server_list": [
                {"server_id": "node_2"},
                {"server_id": "node_3"}]
        }
    ]
}
```

The rank table configuration file is described as follows:

| Level-1 Configuration Item | Level-2 Configuration Item | Level-3 Configuration Item | Description |
| --- | --- | --- | --- |
| status |  |  | Mandatory.<br>Rank table availability status.<br>  - completed: The rank table is available.<br>  - initializing: The rank table is unavailable. |
| version |  |  | Mandatory.<br>Rank table template version.<br>For SuperPoD mode networking, set to 1.2. |
| server_count |  |  | Optional.<br>Number of AI servers participating in collective communication. |
| server_list |  |  | Mandatory.<br>List of AI servers participating in collective communication. |
|  | server_id |  | Mandatory.<br>AI server identifier (string, ≤ 64 characters). Must be globally unique.<br>Example: node_0. |
|  | host_ip |  | Optional.<br>Host IP address of the AI server, in standard IPv4 format.<br>This field must be configured when the HCCL re-execution feature is enabled; otherwise, re-execution will not take effect and the process will continue without re-execution.<br>The re-execution feature is disabled by default. See the environment variable [HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md). |
|  | device |  | Mandatory.<br>List of devices on the AI server. |
|  |  | device_id | Mandatory.<br>Physical ID of the AI processor (device serial number on the server).<br>Run `ls /dev/davinci*` to obtain the physical ID. For example, `/dev/davinci0` indicates physical ID 0.<br>Value range: [0, actual number of devices - 1].<br>Note: This setting overrides the `ASCEND_DEVICE_ID` environment variable. |
|  |  | super_device_id | Optional (if not configured, "AI server mode" is used).<br>Physical ID of the AI processor in the SuperPoD system, serving as the unique identifier of the NPU within the SuperPoD.<br>Developers can query this value using the `npu-smi` command. Example:<br>`npu-smi info -t spod-info -i id -c chip_id`<br><br>  - `id`: Device ID. The NPU ID returned by `npu-smi info -l` is the device ID.<br>  - `chip_id`: Chip ID. The Chip ID returned by `npu-smi info -m` is the chip ID.<br><br>The `SDID` field in the command output is the unique identifier of the NPU in the SuperPoD system. |
|  |  | device_ip | Optional.<br>IP address of the AI processor's integrated NIC, globally unique, in standard IPv4 or IPv6 format.<br>Notes:<br>  1. When the network contains multiple SuperPoDs, this field must be configured.<br>  2. If the network contains only one SuperPoD, this field must be configured in the following scenarios and may be omitted otherwise: when RDMA communication is used within the SuperPoD (i.e., when the environment variable `HCCL_INTER_HCCS_DISABLE` is set to `TRUE`, disabling the HCCS function).<br>Run `cat /etc/hccn.conf` on the current AI server to obtain the NIC IP address. Example output:<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>The `address_xx` entry is the NIC IP address. The number following `address` corresponds to `device_id`; fill in the IP address for each device accordingly. |
|  |  | device_port | Optional.<br>Communication port of the device NIC. Value range: [1, 65535]. Ensure the specified port is not occupied by other processes. Ports in the range [1, 1023] are system-reserved and should be avoided.<br>In single-card multi-process scenarios (where multiple service processes share one NPU), it is recommended to configure this field with distinct port numbers for different processes. Otherwise, port conflicts may cause service startup failures. |
|  |  | backup_device_ip | Optional.<br>When the operator re-execution feature is enabled for inter-SuperPoD communication, if a device NIC failure (RDMA link failure) occurs, this parameter specifies another Die NIC within the same NPU as the backup device NIC, improving the success rate of operator re-execution. This communication mode is called Borrowed-Track Communication.<br>`backup_device_ip` must be in standard IPv4 or IPv6 format. For the query method, see the configuration description of `device_ip`.<br>Notes:<br>  1. The devices corresponding to `backup_device_ip` and `device_ip` must belong to the same NPU—only the NICs of Die0 and Die1 within the same NPU can serve as mutual backups.<br>  2. This configuration takes effect only when the communication operator expansion mode is set to `AI_CPU` and the inter-SuperPoD operator re-execution feature is enabled, i.e.:<br>     `export HCCL_OP_EXPANSION_MODE="AI_CPU"`<br>     `export HCCL_OP_RETRY_ENABLE="L1:1,L2:1"`<br>     `L2` indicates that the physical scope of the communication domain is inter-SuperPoD; a value of `1` enables the operator re-execution feature.<br>  3. To ensure proper operation of Borrowed-Track, the following conditions must be met:<br>     - The backup NIC's communication link is normal.<br>     - The devices serving as mutual backups must both be within the service visibility scope. For example, NPU1 contains two Dies (Device0 and Device1) that serve as mutual backups. If the environment variable `ASCEND_RT_VISIBLE_DEVICES` specifies that only Device0 is visible to the service and Device1 is not, Borrowed-Track cannot be executed.<br>  4. When Borrowed-Track occurs during communication (e.g., the Die0 NIC of a certain NPU fails and the backup Die1 NIC is activated), the original Die0 NIC traffic will also be sent and received through Die1, increasing the traffic on Die1. Overall performance will degrade due to halved physical bandwidth and potential port conflicts.<br>  5. In the Borrowed-Track scenario, if the Die0 NIC of NPU0 fails, it switches to its backup NIC Die1. Since communication between two NPUs requires both the local and peer ends to switch simultaneously, NPU1 will also switch from Die0 to Die1, as shown in "[Figure 1](#figure1)". However, if a communication task already exists between Die0 and Die1, Borrowed-Track cannot be executed.<br>  6. When Borrowed-Track communication is enabled, it is recommended that both Dies of an NPU be assigned to the same training or inference task. If the two Dies of the same NPU are assigned to two different tasks and one task fails, it will borrow the NIC of the other task, causing performance degradation for both tasks.<br>  7. The same NPU supports only one Borrowed-Track operation and does not support switchback. As shown in [Figure 2](#figure2), in "Diagram 1", the communication link between NPU0 and NPU1 fails, the backup link is enabled, Borrowed-Track occurs, and communication proceeds normally. If the fault shown in "Diagram 2" occurs again, Borrowed-Track is no longer supported, and the process exits with an error. |
|  |  | backup_device_port | Optional.<br>Communication port of the backup device NIC. Value range: [1, 65535]. Ensure the specified port is not occupied by other processes. Ports in the range [1, 1023] are system-reserved and should be avoided.<br>If Borrowed-Track communication is enabled and the service is in a single-card multi-process scenario, it is recommended to configure this field with distinct port numbers for different processes. Otherwise, port conflicts may cause service startup failures.<br>Note: The port number configured for the same device NIC when used as the primary NIC and as the backup NIC must be different. |
|  |  | host_port | Optional.<br>Communication port of the host NIC. Value range: [1, 65535]. The `host_port` for each device on the same AI server must be unique. Ensure the specified port is not occupied by other processes. Ports in the range [1, 1023] are system-reserved and should be avoided.<br>If the HCCL re-execution feature is enabled via the environment variable [hccl_op_retry_enable](../hccl_env/HCCL_OP_RETRY_ENABLE.md) and the service is in a single-card multi-process scenario (i.e., multiple service processes sharing one NPU), it is recommended to configure this field with distinct port numbers for different processes. Otherwise, port conflicts may cause service startup failures. |
|  |  | rank_id | Mandatory.<br>Rank identifier, an integer starting from 0. Must be globally unique.<br>Value range: [0, total number of devices - 1].<br>- It is recommended to assign `rank_id` in order of physical device connectivity—group devices that are closer in topology to reduce potential performance impact.<br>&nbsp;&nbsp;For example, if `device_ip` is set in ascending order of physical connections, `rank_id` should also be set in ascending order.<br>- Cross-server `rank_id` interleaving is not supported.<br>&nbsp;&nbsp;Correct example: Server 1 uses {0, 1, 2, 3}, Server 2 uses {4, 5, 6, 7}.<br>&nbsp;&nbsp;Incorrect example: Server 1 uses {0, 1, 2, 7}, Server 2 uses {3, 4, 5, 6}. |
| super_pod_list |  |  | Optional (if not configured, "AI server mode" is used).<br>List of SuperPoDs participating in collective communication. |
|  | super_pod_id |  | Mandatory if `super_pod_list` is configured.<br>Unique identifier of the SuperPoD, globally unique. The following two configuration methods are supported:<br>  - Configure as the physical ID of the SuperPoD, which can be queried using the `npu-smi` tool. Example command:<br>    `npu-smi info -t spod-info -i id -c chip_id`<br>    - `id`: Device ID. The NPU ID returned by `npu-smi info -l` is the device ID.<br>    - `chip_id`: Chip ID. The Chip ID returned by `npu-smi info -m` is the chip ID.<br>    The `Super Pod ID` field in the command output is the physical ID of the SuperPoD.<br>  - User-defined number, in string format, which must be globally unique. In the user-defined ID scenario, users can divide one physical SuperPoD into multiple smaller logical SuperPoDs. For example, if a physical SuperPoD has eight AI server nodes, the user can treat the first four nodes as one small SuperPoD named `super_pod_1` and the last four nodes as another small SuperPoD named `super_pod_2`. |
|  | server_list |  | Mandatory.<br>List of AI servers within the SuperPoD. |
|  |  | server_id | Mandatory.<br>Server identifier (string), corresponding to the `server_id` in `server_list`.<br>Example: node_0. |

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
    "status": "completed",   // Rank table availability status. "completed" indicates that the rank table is available.
    "version": "1.0",        // Rank table template version. For typical cluster networking, set to "1.0".
    "server_count": "2",     // Number of AI servers participating in training. In this example, there are two servers.
    "server_list": [
        {
            "server_id": "node_0",       // AI server identifier (string). Must be globally unique.
            "host_ip": "172.16.0.110",   // Host IP address of the AI server.
            "device": [                  // List of devices on the AI server.
                {
                    "device_id": "0",              // Physical ID of the processor.
                    "device_ip": "192.168.1.8",    // IP address of the processor's physical NIC.
                    "device_port": "16667",        // Communication port of the processor NIC.
                    "host_port": "16666",          // Communication port of the host NIC.
                    "rank_id": "0"                 // Rank identifier, starting from 0.
                },
                {
                    "device_id": "1",
                    "device_ip": "192.168.1.9",
                    "device_port": "16667",
                    "host_port": "16667",
                    "rank_id": "1"
                }
            ]
        },
        {
            "server_id": "node_1",
            "host_ip": "172.16.0.111",
            "device": [
                {
                    "device_id": "0",
                    "device_ip": "192.168.2.8",
                    "device_port": "16667",
                    "host_port": "16666",
                    "rank_id": "2"
                },
                {
                    "device_id": "1",
                    "device_ip": "192.168.2.9",
                    "device_port": "16667",
                    "host_port": "16667",
                    "rank_id": "3"
                }
            ]
        }
    ]
}
```

The following table describes the rank table configuration file:

| Level-1 Configuration Item | Level-2 Configuration Item | Level-3 Configuration Item | Configuration Description |
| --- | --- | --- | --- |
| status |  |  | Mandatory.<br>Rank table availability status.<br>  - completed: The rank table is available.<br>  - initializing: The rank table is unavailable. |
| version |  |  | Mandatory.<br>Rank table template version.<br>For typical cluster networking, set to 1.0. |
| server_count |  |  | Mandatory.<br>Number of AI servers participating in collective communication. |
| server_list |  |  | Mandatory.<br>List of AI servers participating in collective communication. |
|  | server_id |  | Mandatory.<br>AI server identifier (string, ≤ 64 characters). Must be globally unique.<br>Example: node_0. |
|  | host_ip |  | Optional.<br>Host IP address of the AI server, in standard IPv4 format.<br>This field must be configured when the HCCL re-execution feature is enabled; otherwise, re-execution will not take effect and the process will continue without it.<br>The re-execution feature is disabled by default. See the environment variable [HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md). |
|  | device |  | Mandatory.<br>List of devices on the AI server. |
|  |  | device_id | Mandatory.<br>Physical ID of the AI processor (device serial number on the server).<br>Run `ls /dev/davinci*` to obtain the physical ID. For example, `/dev/davinci0` indicates physical ID 0.<br>Value range: [0, actual number of devices - 1].<br>Note: This setting overrides the `ASCEND_DEVICE_ID` environment variable. |
|  |  | device_ip | Optional.<br>IP address of the AI processor's integrated NIC, globally unique, in standard IPv4 or IPv6 format.<br>Notes:<br>  - In multi-node scenarios, this field is mandatory.<br>  - In single-node scenarios, this field can be left empty.<br>Run `cat /etc/hccn.conf` on the current AI server to obtain the NIC IP address. Example output:<br>address_0=xx.xx.xx.xx<br>netmask_0=xx.xx.xx.xx<br>netdetect_0=xx.xx.xx.xx<br>The `address_xx` entry is the NIC IP address. The number following `address` corresponds to `device_id`; fill in the IP address for each device accordingly. |
|  |  | device_port | Optional.<br>Communication port of the device NIC. Value range: [1, 65535]. Ensure the specified port is not occupied by other processes. Ports in the range [1, 1023] are system-reserved and should be avoided.<br>In single-card multi-process scenarios (where multiple service processes share one NPU), it is recommended to configure this field with distinct port numbers for different processes. Otherwise, port conflicts may cause service startup failures. |
|  |  | host_port | Optional.<br>Communication port of the host NIC. Value range: [1, 65535]. The `host_port` for each device on the same AI server must be unique. Ensure the specified port is not occupied by other processes. Ports in the range [1, 1023] are system-reserved and should be avoided.<br>If the HCCL re-execution feature is enabled via the environment variable [HCCL_OP_RETRY_ENABLE](../hccl_env/HCCL_OP_RETRY_ENABLE.md) and the service is in a single-card multi-process scenario (i.e., multiple service processes sharing one NPU), it is recommended to configure this field with distinct port numbers for different processes. Otherwise, port conflicts may cause service startup failures. |
|  |  | rank_id | Mandatory.<br>Rank identifier, an integer starting from 0. Must be globally unique.<br>Value range: [0, total number of devices - 1].<br>- It is recommended to assign `rank_id` in order of physical device connectivity—group devices that are closer in topology to reduce potential performance impact.<br>&nbsp;&nbsp;For example, if `device_ip` is set in ascending order of physical connections, `rank_id` should also be set in ascending order.<br>- Cross-server `rank_id` interleaving is not supported.<br>&nbsp;&nbsp;Correct example: Server 1 uses {0, 1, 2, 3}, Server 2 uses {4, 5, 6, 7}.<br>&nbsp;&nbsp;Incorrect example: Server 1 uses {0, 1, 2, 7}, Server 2 uses {3, 4, 5, 6}. |
