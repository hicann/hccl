# Configuring Resource Information via Environment Variables

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-30T03:50:07.819Z pushedAt=2026-07-31T07:26:13.177Z -->

In addition to configuring resource information through a rank table file, developers can also configure resource information using the combination of environment variables described in this section.

The method of configuring resource information via environment variables is applicable only to the communication domain initialization of TensorFlow framework networks and supports only the following products:

Atlas A2 training products / Atlas A2 inference products

Atlas training products

## Configuration Description

The following environment variables must be configured on each AI server node that executes training to set up resource information. An example is shown below:

```bash
export CM_CHIEF_IP=192.168.1.1
export CM_CHIEF_PORT=6000
export CM_CHIEF_DEVICE=0
export CM_WORKER_SIZE=8
export CM_WORKER_IP=192.168.0.1
export HCCL_SOCKET_FAMILY=AF_INET
```

- CM_CHIEF_IP: The Host listening IP address of the Master node, which is the IP address used for communication with other nodes. It must be in standard IPv4 or IPv6 format.

- CM_CHIEF_PORT: The listening port of the Master node. It must be configured as an integer with a value range of 0–65520. Ensure that the port is not occupied by other processes.

- CM_CHIEF_DEVICE: The Device logical ID used by the Master node to collect server-side cluster information.

This environment variable must be configured as an integer. Value range: [0, maximum number of devices in the server - 1].

- CM_WORKER_SIZE: Specifies the total number of devices participating in cluster training within the network. It must be configured as an integer, with a value range of 0 to 32768.

- CM_WORKER_IP: Specifies the NIC IP address used by the current node to communicate with the master. It must be in standard IPv4 or IPv6 format.

- HCCL_SOCKET_FAMILY: **This environment variable is optional.** It controls the IP protocol version used by the device-side communication NIC. AF_INET indicates the IPv4 protocol, and AF_INET6 indicates the IPv6 protocol. **When not configured, the IPv4 protocol is used by default.**

**NOTE**

- If the IP protocol specified by the environment variable HCCL_SOCKET_FAMILY does not match the actually obtained NIC information, the NIC information in the actual environment prevails.

    For example, if the environment variable HCCL_SOCKET_FAMILY is set to AF_INET6, but only IPv4 NICs exist on the device side, IPv4 NICs will actually be used.

- When configuring cluster information using the environment variables described above, the environment variables RANK_TABLE_FILE, RANK_ID, and RANK_SIZE must not exist in the environment.

- For Atlas A2 training products/Atlas A2 inference products, if the service involves a single-card multi-process scenario, it is recommended to configure the communication port used by HCCL on the NPU side through the environment variable [HCCL_NPU_SOCKET_PORT_RANGE](../hccl_env/HCCL_NPU_SOCKET_PORT_RANGE.md). Otherwise, port conflicts may occur. However, note that multiple processes will have a certain impact on resource overhead and communication performance. Configuration example:

    ```bash
    export HCCL_NPU_SOCKET_PORT_RANGE="auto"
    ```

## Configuration Example

Taking the scenario where the number of AI Server nodes for distributed training is 2 and the number of Devices is 16 as an example, each AI Server node has 8 Devices. Before starting the training process on each Device, configure the following environment variables in the corresponding shell window to set up the resource information.

- Node 0: This node is the Master node, responsible for cluster information management, resource allocation, and scheduling.

    ```bash
    export CM_CHIEF_IP=192.168.1.1
    export CM_CHIEF_PORT=6000
    export CM_CHIEF_DEVICE=0
    export CM_WORKER_SIZE=16
    export CM_WORKER_IP=192.168.1.1
    ```

- Node 1

    ```bash
    export CM_CHIEF_IP=192.168.1.1
    export CM_CHIEF_PORT=6000
    export CM_CHIEF_DEVICE=0
    export CM_WORKER_SIZE=16
    export CM_WORKER_IP=192.168.2.1
    ```
