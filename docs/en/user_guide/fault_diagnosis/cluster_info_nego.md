# Cluster Information Negotiation

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T03:56:58.431Z pushedAt=2026-08-13T06:57:20.558Z -->

## Service Flow and Troubleshooting Approach

### Service Flow

During cluster information negotiation (the scenario where a communicator is created based on root node information), HCCL establishes socket connections between the server node and each rank node to exchange local information, thereby obtaining the cluster information of the entire communicator and completing communicator initialization.

![Cluster Information Negotiation Service Flow](figures/cluster_info_nego_flow.png)

1. The server node calls the HcclGetRootInfo API to start a listening thread.

    1. A server node is selected from the cluster, typically the rank0 node within the communicator. This node calls the HcclGetRootInfo API.

    2. The IP address and port information of the host network interface card are obtained to generate the rootInfo value. The host network interface card can be specified through the HCCL_SOCKET_IFNAME environment variable, and the port can be specified through the HCCL_IF_BASE_PORT and HCCL_HOST_SOCKET_PORT_RANGE environment variables.

    3. The server node performs binding and listening based on the IP address and port, starts a background thread to wait for connections from all agents in the communicator, and directly returns rootInfo to complete the interface call.

2. Each rank node calls the HcclCommInitRootInfo interface to establish a connection to the server node.

    1. The upper-layer service or framework broadcasts rootInfo to each rank in the communicator. Each rank node calls the HcclCommInitRootInfo interface with rootInfo as an input parameter.

    2. A socket connection is established with the server through the host network interface card, and the rank's own rankInfo is sent to the server. After the transmission is complete, the rank enters the receiving state and waits for the server to return the complete cluster information. The process of establishing the socket connection and waiting for the cluster information to be returned must be completed within a specified timeout period, which can be controlled through the HCCL_CONNECT_TIMEOUT environment variable.

3. The server node collects the complete cluster information and sends it to each rank.

    1. After collecting the rankInfo of all ranks, the background thread on the server node generates the complete cluster information and sends it to each agent thread of the ranks. In this way, each rank obtains the full rank information of the entire communicator.

    2. Meanwhile, the server must complete waiting for socket connections from all ranks within a certain timeout. The timeout can be controlled by the HCCL_CONNECT_TIMEOUT environment variable.

### Cause Analysis

Since the communicator creation method based on root node information requires socket connections to be established between rank nodes, the host-side network interface cards are used, and all ranks within the communicator must execute synchronously within the timeout period. Therefore, network connectivity issues with the host NICs or the failure of some ranks to correctly establish socket connections are common causes of negotiation failures. The primary task in fault location is to identify the abnormal rank.

Common causes of cluster information negotiation phase failures and key logs:

- Port binding failure on the server node. Run the following command to check for port binding failures. For details, see [Server Node Port Binding Failure (EI0019)](#server-node-port-binding-failure-ei0019).

    ```bash
    grep -r "socket type\[2\].*Please check the port status and whether the port is being used by other process"
    ```

- The server node does not receive socket connections from all ranks in the communicator. Run the following command for a quick query. For details, see [Some Ranks Fail to Connect to the Server Node (EI0015)](#some-ranks-fail-to-connect-to-the-server-node-ei0015).

    ```bash
    grep -r "topo exchange server get socket timeout"
    ```

- A socket establishment timeout occurs between the rank node and the server node. This can be quickly queried using the following command. For details, see [Rank and Server Node Socket Establishment Timeout (EI0015)](#rank-and-server-node-socket-establishment-timeout-ei0015).

    ```bash
    grep -r "topo exchange agent get socket timeout"
    ```

### Key Log Information

- During communicator creation, HCCL records key log information when the communicator creation interface is invoked. The log records are stored in the plog file under the run directory of the CANN log. Whether a process has invoked the corresponding communicator creation interface can be determined based on the corresponding logs. For detailed log information, see [HCCL-Related Log Description](./debug_thinking.md#hccl-related-log-description).

- During the communicator negotiation phase, if a rank fails to establish a socket connection with the root node, the root node prints the information of connected ranks before timeout exit and broadcasts the information of missing ranks to the ranks that have successfully established connections. The unconnected rank can be identified based on this information to further confirm the root cause. For detailed log information, see [Some Ranks Fail to Connect to the Server Node (EI0015)](#some-ranks-fail-to-connect-to-the-server-node-ei0015).

## Server Node Port Binding Failure (EI0019)

### Symptom

The log contains the EI0019 error. The error message is as follows:

```text
[PID: 2267203] 2025-11-21-11:38:29.575.404 Communication_Error_Bind_IP_Port(EI0019): Failed to enable listening for the host network adapter socket.Reason: The IP address 192.168.1.100 and port 50001 have already been bound.
```

The CANN log contains the keyword "socket type\[2\], \*\*\* Please check the port status and whether the port is being used by other process.", as shown below. **In addition, note that port binding failures may also occur during the parameter plane link establishment stage when communication operators are dispatched. The "socket type" in the error log can be used to determine the cause: if the type is 2, the failure is a host-side NIC port binding failure during communicator cluster negotiation. For Atlas A3 training products/Atlas A3 inference products and Atlas A2 training products/Atlas A2 inference products, if the type is 0 or 1, the failure is a parameter plane port binding failure. For details, see** [Parameter Plane Port Binding Failure (EI0019)](./param_link_stage.md#parameter-plane-port-binding-failure-ei0019).

```text
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.860 [hccl_socket.cc:110] [3626636][InitChannelStage][RanktableDetect] socket type[2], listen on ip[192.168.1.100%enp53s0f2] and specific port[60000] fail. Please check the port status and whether the port is being used by other process.
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.869 [topoinfo_detect.cc:744] [3626636][InitGroupStage][RanktableDetect]StartRootNetwork failed, ret[7]
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.874 [topoinfo_detect.cc:233] [3626636][InitGroupStage][RanktableDetect]SetupServer failed, hostIP[192.168.1.100%enp53s0f2] and hostPort[60000] ret[7]
[ERROR] HCCL(3626636,all_reduce_test):2025-11-21-13:18:47.639.882 [op_base.cc:1071] [3626636][InitGroupStage][RanktableDetect]HcclGetRootInfo failed, ret[7]
```

### Possible Cause

HCCL port binding fails. During the communicator creation phase, HCCL needs to bind ports 60000–60031 by default. If these ports are already bound at that time, the HCCL port binding fails, which in turn causes the communicator creation to fail.

### Solution

The port range can be configured as follows:

- Specify the starting port number and port reservation range for the host network interface card through the [HCCL_IF_BASE_PORT](../hccl_env/HCCL_IF_BASE_PORT.md) environment variable.

- For Atlas A3 training products/Atlas A3 inference products and Atlas A2 training products/Atlas A2 inference products, if multiple processes need to be executed simultaneously on a single NPU, the communication port range used by HCCL on the host side must be configured through [HCCL_HOST_SOCKET_PORT_RANGE](../hccl_env/HCCL_HOST_SOCKET_PORT_RANGE.md) to avoid port conflicts between multiple processes.

## Some Ranks Fail to Connect to the Server Node (EI0015)

### Symptom

The CANN log contains the keyword "topo exchange server get socket timeout!" or "Failed to connect agent".

- Symptom on the server node:

    ```text
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.624.966 [topoinfo_exchange_server.cc:314] [1041362][InitGroupStage][RanktableDetect]topo exchange server get socket timeout! timeout[120 s]
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.625.103 [topoinfo_exchange_server.cc:501] [1041362][InitGroupStage][DisplayConnectionedRank]total connected num is [4],line num is [1]
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.625.112 [topoinfo_exchange_server.cc:503] [1041362][InitGroupStage][DisplayConnectionedRank]need connect rankNum is [8]
    [ERROR] HCCL(1041081,all_reduce_test):2025-11-21-01:20:01.625.145 [topoinfo_exchange_server.cc:517] [1041362][InitGroupStage][DisplayConnectionedRank]connected rankinfo[LINE 0]: [0000000000000000],[0000000000000002],[0000000000000004],[0000000000000006];
    ```

- Symptom on the agent node:

    ```text
    [ERROR] HCCL(1041085,all_reduce_test):2025-11-21-01:20:01.630.122 [topoinfo_exchange_base.cc:145] [1041085][InitGroupStage][RanktableDetect] TopoDetect ERROR occur !!! fault_type[1], fault_info["Failed to connect agent[1,3,5,7,]"]
    [ERROR] HCCL(1041085,all_reduce_test):2025-11-21-01:20:01.630.557 [topoinfo_exchange_agent.cc:552] [1041085][InitGroupStage][RanktableDetect]rank num[8]is different with rank list size[4] in total topo rank info.
    ```

### Possible Cause

After the server node calls the `HcclGetRootinfo` interface, a background thread is started to wait for all ranks to connect until the timeout expires. Therefore, if not all ranks in the communicator successfully connect to the server thread within the timeout period, the server thread waits until timeout and reports an error. In addition, after the timeout error occurs, the server thread prints the list of currently connected ranks. Based on this information, the ranks that failed to connect can be identified, and the cause of the connection failure for those ranks can be further investigated.

### Solution

The following figure shows the troubleshooting approach for partial rank disconnection:

![Troubleshooting approach for partial rank disconnection](figures/rank_disconnect_debug.png)

1. Identify the disconnected ranks from the error message.

    - **Server node**: When the server node times out, it prints the information of connected ranks. Since the rankId sequence within the communicator is [0 ~ rankSize-1], the disconnected ranks can be derived from the connected rank information. For example, in the log case above, the connection of rank9 is missing, so further investigation into why rank9 failed to connect is required.

    - **Agent node**: For agents that have successfully connected, after the server node times out, they receive the cluster unconnected rank error message disseminated by the server node. Therefore, the unconnected rank can be directly identified based on "Failed to connect agent" in the error message, and the cause of the connection failure for that rank can be further investigated. For example, the log case above indicates that the connection of rank9 is missing, so the cause of rank9's failure to connect needs to be further investigated.

2. Confirm whether the unconnected rank has issued the communicator creation interface.

    Since HCCL records default logs in the run directory of CANN logs during communicator creation, you can filter the **entire cluster's logs** to check whether the corresponding rank has issued the communicator creation. For example, run the following command to filter whether rank9 has a Communicator creation issuance record. If there are multiple communicators, multiple log lines may appear. You can confirm whether they belong to the same communicator based on the "identifier" communicator name in the logs:

    ```bash
    grep -r "Entry-HcclCommInitRootInfoInner" | grep "rank\[9\]"
    ```

    - If the corresponding rank has issued the communicator creation interface, obtain the node and process ID information of the missing rank based on the search results, and then further investigate the related error log information based on the CANN logs of the missing rank.

    - If no communicator creation interface issuance log is found for the corresponding rank in the cluster logs, investigate the cause of the rank not issuing the corresponding Communicator creation interface from the service side.

## Rank and Server Node Socket Establishment Timeout (EI0015)

### Symptom

The keyword "topo exchange agent get socket timeout!" is found in the CANN log, as shown below:

```text
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.979 [topoinfo_exchange_agent.cc:190] [7988][InitGroupStage][RanktableDetect]topo exchange agent get socket timeout! timeout[120] 
[ERROR] HCCL(7988,all_reduce_test):2025-03-19-04:16:13.978.995 [topoinfo_exchange_agent.cc:41] [7988][TopoInfoExchangeAgent][Setup]TopoExchangeAgent: connect server[127.10.0.1 : 60000] failed
```

### Possible Cause

During the communicator creation phase through cluster negotiation, the current rank creates a socket with the IP address and port of the server node based on the rootInfo information. However, the socket establishment times out due to network connectivity issues.

### Solution

1. Check the connectivity of the host-side network and the corresponding port.

    Since the host side often has multiple network interface cards, HCCL selects a host NIC in lexicographical order for socket connection by default. As a result, an unreachable host NIC may be selected. You can specify the NIC to be used through the [HCCL_SOCKET_IFNAME](../hccl_env/HCCL_SOCKET_IFNAME.md) environment variable. If the NIC is correctly selected, further check whether the specified port is reachable. For example, based on the following log, check whether the faulty node can reach port 60000 of 127.10.0.1. You can run the following command to query the host NIC information obtained by the current process:

    ```bash
    grep -r "get host ip success\|find nic.*success"
    ```

2. Check whether the interval between the communicator creation delivery time of each agent and the server node within the communicator exceeds the timeout.

    In the run directory of the CANN log, run `grep -r "Entry-"` to confirm the delivery time of the communicator creation interface, or directly calculate the interval between the communicator creation delivery time of each agent and the server node based on the timestamp in the error log. If the interval exceeds the timeout, configure the connection establishment timeout through [HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md), which defaults to 120 seconds.

    The time when each communicator creation interface is delivered can be queried using the following command:

    ```bash
    grep -r "Entry-HcclGetRootInfo\|Entry-HcclCommInitRootInfoInner" run/plog
    ```

    Based on the following query results, it can be seen that the communicator interface delivery time for rank\[3\] is approximately 200 seconds slower than that of other ranks, while the connection establishment timeout is 120 seconds. Therefore, the entire communicator creation process eventually failed due to a timeout. For this scenario, if there are preceding differences in service processes on different ranks, for example, some processes need to load more data and therefore start more slowly, this error can be resolved by increasing the timeout through the [HCCL_CONNECT_TIMEOUT](../hccl_env/HCCL_CONNECT_TIMEOUT.md) environment variable.

    ```text
    [INFO] HCCL(3079955,all_reduce_test):2025-11-20-11:59:56.716.583 [op_base.cc:1293] [3079955]Entry-HcclCommInitRootInfoInner:ranks[4], rank[3], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[3]
    [INFO] HCCL(3079952,all_reduce_test):2025-11-20-11:56:36.704.523 [op_base.cc:858] [3079952]Entry-HcclGetRootInfo:rootInfo[0xaaaae85c79a0], deviceLogicId[0]
    [INFO] HCCL(3079952,all_reduce_test):2025-11-20-11:56:36.711.546 [op_base.cc:1293] [3079952]Entry-HcclCommInitRootInfoInner:ranks[4], rank[0], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[0]
    [INFO] HCCL(3079953,all_reduce_test):2025-11-20-11:56:36.712.024 [op_base.cc:1293] [3079953]Entry-HcclCommInitRootInfoInner:ranks[4], rank[1], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[1]
    [INFO] HCCL(3079954,all_reduce_test):2025-11-20-11:56:36.712.065 [op_base.cc:1293] [3079954]Entry-HcclCommInitRootInfoInner:ranks[4], rank[2], rootinfo: host ip[127.10.0.1] port[60000] nicDeploy[1] identifier[127.10.0.1%enp_60000_0_1763610996711234], deviceLogicId[2]
    ```

## Typical Multi-Node Scenario Communicator Initialization Failure (EI0015)

### Symptom

The following figure shows a communicator creation failure case in a three-node scenario:

![Communicator Creation Failure Case in a Three-Node Scenario](figures/3node_comm_create_fail.png)

This symptom is a typical error log of a communicator creation negotiation timeout in a three-node, 24-card scenario, where node 0 is the root node of the communicator. The error symptoms of each node are analyzed as follows:

- **Node 0**: Node 0 is the root node. The error message indicates that the server thread timed out while waiting for all ranks in the communicator to connect. The successfully connected ranks can be obtained from the error message, and the unconnected ranks (rank16 to rank23) can be deduced by reverse inference.

- **Node 1**: This node successfully created a socket connection with the root node. After waiting for the root node timeout, it received the unconnected rank information broadcast by the root node. The unconnected ranks, rank16 to rank23, can be directly obtained from the error log.

- **Node 2**: The error log of Node 2 indicates a socket timeout when establishing a connection with the server node. The root cause is a Host-side network configuration error between Node 2 and the root node, which prevented the connection. The problem was resolved after the configuration was corrected.

### Troubleshooting Approach

As can be seen from this typical scenario, when a link establishment timeout occurs during communicator creation in the cluster, the unconnected rank -- that is, the root node that reported the error -- can be quickly identified from the error log, regardless of whether the log originates from the server node or a node that has been successfully connected. At this point, it is sufficient to focus on troubleshooting the failure cause of the unconnected rank. For example, a common cause of connection timeout is that the [HCCL_SOCKET_IFNAME](../hccl_env/HCCL_SOCKET_IFNAME.md) environment variable is not configured, resulting in the use of an unreachable Host network interface card.
