# Using the Communication Library API to Implement Communication Functions

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-31T04:00:52.172Z pushedAt=2026-08-17T02:02:50.895Z -->

## Overview

HCCL provides development interfaces in both C and Python languages for implementing distributed capabilities.

- The C language interface is used for framework adaptation in single-operator mode to implement distributed capabilities.

- The Python language interface is used for framework adaptation in graph mode. Currently, it is only used for implementing distributed optimization of TensorFlow networks on AI processors.

**This chapter describes how to call the HCCL C language interface to develop collective communication functions.**

The main development process for developers to implement collective communication functions by calling HCCL C language interfaces is as follows.

![Collective Communication Operation Process](figures/hccl_operation_flow.png "Collective Communication Operation Process")

1. First, configure the cluster information, create a communicator handle, and initialize the HCCL communicator.

2. Implement communication operations. HCCL communication operations include two categories: point-to-point communication and collective communication.

    - Point-to-point communication refers to the process of directly transmitting data between two NPUs in a multi-NPU environment. It is commonly used for sending and receiving activation values in pipeline parallelism scenarios. HCCL provides point-to-point communication at different granularities, including single-send and single-receive interfaces between a single rank and another single rank, as well as batch send and receive interfaces across multiple ranks.

    - Collective communication refers to data transmission operations involving multiple NPUs, such as AllReduce, AllGather, and Broadcast. It is commonly used for gradient synchronization and parameter update across different NPUs in large-scale clusters. Collective communication operations enable all compute nodes to exchange data in a parallel, efficient, and orderly manner, thereby improving data transmission efficiency.

3. After collective communication operations are complete, the communicator must be destroyed to release the associated memory and stream resources.

## Communicator

A communicator is the context in which collective communication operators are executed. It manages the corresponding communication objects (for example, an NPU is a communication object) and the resources required for communication. Each communication object in a communicator is called a rank, and each rank is assigned a unique identifier ranging from 0 to n-1, where n is the number of NPUs.

Communicators can be created in the following ways, depending on the user scenario:

- Multi-machine collective communication scenario

  - If a complete rank table file that describes the cluster information is available, a communicator can be created by using the `HcclCommInitClusterInfo` API, or a communicator with specific configurations can be created by using the `HcclCommInitClusterInfoConfig` API.

  - If a complete rank table file is not available, the HcclGetRootInfo interface can be used together with the HcclCommInitRootInfo/HcclCommInitRootInfoConfig interface to create a communicator based on root node information.

- In a single-machine collective communication scenario, the HcclCommInitAll interface can be used to create communicators in batches within a single machine.

- Based on an existing communicator, the HcclCreateSubCommConfig interface can be used to split a sub-communicator with a specific configuration.

> [!NOTE]Note
>
> - All communication operators under multiple communicators must be delivered serially on each Device. Out-of-order delivery, concurrent multi-threaded delivery, and thread reentry are not supported.
> - On the same Device, all communication operators within the same communicator must use the same Context for delivery.
> - Mixed execution of graph mode communication and single-operator communication is not supported within the same communicator.
> - Operators within the same communicator must be executed serially by the user.
> - Multiple communicators must be created serially on the same NPU.
> - For Atlas A3 training products/Atlas A3 inference products, when initializing a communicator, if multiple SuperPoDs exist in the network, configure the AI Server information belonging to the same SuperPoD together. Assuming there are two SuperPoDs with identifiers "0" and "1", configure the AI Server information in "0" first, and then configure the AI Server information in "1". Cross-configuration of AI Server information between "0" and "1" is not supported.

### Creating a Communicator Based on a Rank Table

In multi-machine collective communication scenarios where a communicator is created based on a cluster information configuration file (rank table file), each card requires a separate process. Follow the procedure below to create the communicator:

1. Construct a rank table file. (For rank table file configuration, see [Cluster Information Configuration](./cluster_info_config/README.md).)

2. Each card calls the [HcclCommInitClusterInfo](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCommInitClusterInfo.md) API to create a communicator, or calls the [HcclCommInitClusterInfoConfig](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCommInitClusterInfoConfig.md) API to create a communicator with specific configurations.

A simple code sample snippet is as follows:

```c
    int devId = 0;
    // Configure the rank table file path.
    char* rankTableFile = "/home/rank_table.json";
    // Define the communicator handle.
    HcclComm hcclComm;
    // Initialize the HCCL communicator.
    HcclCommInitClusterInfo(rankTableFile, devId, &hcclComm);
   
    /*Collective Communication Operations*/

    // Destroy the HCCL communicator.
    HcclCommDestroy(hcclComm);
```

> [!NOTE]Note
> For Ascend 950PR/Ascend 950DT, Atlas A3 training products/Atlas A3 inference products, and Atlas A2 training products/Atlas A2 inference products, if the service is in a single-card multi-process scenario, it is recommended to configure the "device_port" field in the rank table configuration file, and different business processes must be set with different port numbers. Otherwise, the service may fail due to port conflicts. However, note that multiple processes will have a certain impact on resource overhead and communication performance.

### Creating a Communicator Based on Root Node Information

In multi-machine collective communication scenarios, if a complete cluster information configuration file (rank table file) is not available, HCCL provides a method for creating a communicator based on root node information. **The following two typical usage scenarios are mainly involved**:

- Scenario where each Device corresponds to one business process. The implementation process is as follows:

    1. For Ascend 950PR/Ascend 950DT, check whether the rootinfo file exists. Skip this step for other products.

        Before creating a communicator based on root node information, check whether the `/etc/hccl_rootInfo.json` file exists. This file records the EID (Entity ID, the identifier of the initiator or receiver in communication) information for inter-NPU communication and is automatically generated after environment deployment. If this file does not exist, submit an issue in the current source repository.

    2. Specify the communication IP address or communication NIC used by the Host node during HCCL initialization (optional).<a id="set_host_nic"></a>

        - Method 1: Configure the communication IP address on each Host node through the environment variable [HCCL_IF_IP](./hccl_env/HCCL_IF_IP.md). This IP address is used to communicate with the root node and can be in IPv4 or IPv6 format. Only one IP address can be configured. The following is a configuration example:

            ```bash
            export HCCL_IF_IP=10.10.10.1
            ```

        - Method 2: Configure the communication NIC name on each Host node through the environment variable [HCCL_SOCKET_IFNAME](./hccl_env/HCCL_SOCKET_IFNAME.md), and configure the communication protocol used by the NIC through [HCCL_SOCKET_FAMILY](./hccl_env/HCCL_SOCKET_FAMILY.md). HCCL will obtain the Host IP through this NIC name to communicate with the root node. The following is a configuration example:

            ```bash
            # Configure the IP protocol version used by the communication NIC during HCCL initialization. AF_INET: IPv4; AF_INET6: IPv6.
            export HCCL_SOCKET_FAMILY=AF_INET
            
            # The following NIC name configuration formats are supported (choose one of the four specifications. Multiple NICs can be configured in the environment variable, separated by commas. The first matched NIC is used as the communication NIC).
            # Exact NIC matching
            export HCCL_SOCKET_IFNAME==eth0,enp0   # Use the specified eth0 or enp0 NIC
            export HCCL_SOCKET_IFNAME=^=eth0,enp0     # Do not use the eth0 or enp0 NIC
            
            # Fuzzy NIC matching
            export HCCL_SOCKET_IFNAME=eth,enp       # Use all NICs prefixed with eth or enp
            export HCCL_SOCKET_IFNAME=^eth,enp      # Do not use any NIC whose name starts with eth or enp.
            ```

        The environment variable HCCL_IF_IP takes precedence over HCCL_SOCKET_IFNAME. If neither HCCL_IF_IP nor HCCL_SOCKET_IFNAME is configured, the system automatically selects the NIC based on the following priority. If the NIC selected on the current node cannot establish a link with the NIC selected on the root node, HCCL connection setup will fail.

        ```text
        NIC selection priority: physical NICs other than Docker and loopback (sorted by NIC name in ascending lexicographic order) > Docker NICs > loopback NICs.
        ```

    3. On the root node, call the [HcclGetRootInfo](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclGetRootInfo.md) API to generate the root node rank identification information "rootInfo", which includes the device IP address, device ID, and other details.

    4. Broadcast the rank information of the root node to all ranks within the communicator.

    5. On all nodes in the communicator, call the [HcclCommInitRootInfo](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCommInitRootInfo.md) or [HcclCommInitRootInfoConfig](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCommInitRootInfoConfig.md) API (which creates a communicator with specific configuration) to initialize the communicator based on the received "rootInfo" and the rank ID of the local rank.

- Each AI Server corresponds to one business process, and each thread corresponds to one Device. In the scenario where multiple communicators are created through multi-threading, the implementation process is as follows:

    1. For Ascend 950PR/Ascend 950DT, check whether the rootinfo file exists. For other products, skip this step.

        Before creating a communicator based on root node information, check whether the "/etc/hccl_rootInfo.json" file exists. This file records the EID (Entity ID, the identifier of the initiating or receiving object in communication) information for inter-NPU communication and is automatically generated after environment deployment is complete. If this file does not exist, file an issue in the current code repository.

    2. Refer to [Step 2 of the "Each Device Corresponds to One Business Process" scenario](#set_host_nic) to specify the communication IP address or communication NIC used by the host node during HCCL initialization (optional).

    3. In the main process, loop through "specifying different Devices + calling the HcclGetRootInfo API" to obtain multiple "rootInfo" entries.

    4. Each Device is matched with a thread, and based on different "rootInfo" information, the [HcclCommInitRootInfo](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCommInitRootInfo.md) or [HcclCommInitRootInfoConfig](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCommInitRootInfoConfig.md) interface is concurrently called to initialize the communicator.

> [!NOTE]Note
>
> For Ascend 950PR/Ascend 950DT, Atlas A3 training products/Atlas A3 inference products, and Atlas A2 training products/Atlas A2 inference products, if the service is in a single-card multi-process scenario, it is recommended to configure the communication ports used by HCCL on the Host side and the NPU side respectively through the environment variables "[HCCL_HOST_SOCKET_PORT_RANGE](./hccl_env/HCCL_HOST_SOCKET_PORT_RANGE.md)" and "[HCCL_NPU_SOCKET_PORT_RANGE](./hccl_env/HCCL_NPU_SOCKET_PORT_RANGE.md)". Otherwise, port conflicts may occur. A configuration example is shown below. However, note that multiple processes may have a certain impact on resource overhead and communication performance.
>
> ```bash
> export HCCL_HOST_SOCKET_PORT_RANGE="auto"
> export HCCL_NPU_SOCKET_PORT_RANGE="auto"
> ```

### Batch Creation of Communicators on a Single Machine

In a single-machine communication scenario, developers can create communicators for multiple cards through a single process, where each card corresponds to one thread. The creation process is as follows:

1. Construct the Device list in the communicator, for example: \{0, 1, 2, 3, 4, 5, 6, 7\}, where the Device IDs in the list are logical IDs (which can be queried using the **npu-smi info -m** command). HCCL creates the communicator in the order specified in the list.

2. Call the [HcclCommInitAll](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCommInitAll.md) API in the process to create the communicator.

```c
    uint32_t ndev = 8;
    // Construct the logical ID list of devices.
    int32_t devices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    // Define the communicator handle.
    HcclComm comms[ndev];
    // Initialize the HCCL communicator.
    HcclCommInitAll(ndev, devices, comms);

    // Start threads to execute collective communication operations.
    std::vector<std::unique_ptr<std::thread> > threads(ndev);
    struct ThreadContext args[ndev];
    for (uint32_t i = 0; i < ndev; i++) {
        args[i].device = i;
        args[i].comm = comms[i];
       /*  Collective communication operation   */      
    }

    // Destroy the HCCL communicator.
    for (uint32_t i = 0; i < ndev; i++) {
        HcclCommDestroy(comms[i]);
    }
```

Note that when calling collective communication operation APIs (such as HcclAllReduce) in multiple threads, it should be ensured that the time difference between the calls to collective communication operation APIs in different threads does not exceed the link establishment timeout period of collective communication (which can be set via the environment variable [HCCL_CONNECT_TIMEOUT](./hccl_env/HCCL_CONNECT_TIMEOUT.md), with a default value of 120s), to avoid link establishment timeout.

### Splitting a Sub-Communicator From an Existing Communicator

HCCL provides the [HcclCreateSubCommConfig](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCreateSubCommConfig.md) API, which splits a sub-communicator with a specific configuration from an existing communicator. This method of creating a sub-communicator does not require socket connection establishment or rank information exchange, and can be used for rapid communicator creation in the event of a service fault.

```c
// Initialize the global communicator.
HcclComm globalHcclComm;
HcclCommInitClusterInfo(rankTableFile, devId, &globalHcclComm);
// Communicator configuration.
HcclCommConfig config;
HcclCommConfigInit(&config);
config.hcclBufferSize = 50;
strcpy(config.hcclCommName, "comm_1");
// Initialize the sub-communicator.
HcclComm hcclComm;
uint32_t rankIds[4] = {0, 1, 2, 3};  // Rank list of the sub-communicator
HcclCreateSubCommConfig(&globalHcclComm, 4, rankIds, 1, devId, &config, &hcclComm);
```

> [!NOTE]Note
> This interface does not support nested splitting of communicators, meaning that further splitting of sub-communicators within a sub-communicator is not supported.

### Destroying a Communicator

After collective communication operations are complete, the [HcclCommDestroy](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCommDestroy.md) interface must be called to destroy the specified communicator, and the runtime management interface must be called to release the memory, stream, and device resources used for communication.

## Collective Communication

Collective communication is a communication mode in which multiple NPUs participate in data transmission to form a collective operation. It is commonly used in scenarios such as gradient synchronization and parameter update among different NPUs in large-scale clusters.

HCCL supports communication operators such as AllReduce, Broadcast, AllGather, Scatter, ReduceScatter, Reduce, AlltoAll, and AlltoAllV, and provides corresponding APIs for developers to invoke, enabling rapid implementation of collective communication capabilities.

### Broadcast

The Broadcast operation broadcasts data from the root node within the communicator to other ranks.

![Broadcast](figures/broadcast.png)

Note: Only one root node can exist within the communicator.

Related interface: [HcclBroadcast](../api_ref/comm_op_interface/HcclBroadcast.md).

### Scatter

The Scatter operation evenly distributes data from the root node within the communicator to other ranks.

![Scatter](figures/scatter.png)

Note:Only one root node is allowed within the communicator.

Related API: [HcclScatter](../api_ref/comm_op_interface/HcclScatter.md).

### AllGather

The AllGather operation reorders the inputs of all nodes within the communicator by rank ID (in ascending order), concatenates them, and then sends the result to the output buffer of all nodes.

![AllGather](figures/allgather.png)

> [!NOTE]Note
> For the AllGather operation, each node receives the data set reordered by rank ID, meaning that the AllGather output is the same for every node.

Related API: [HcclAllGather](../api_ref/comm_op_interface/HcclAllGather.md).

### AllGatherV

The AllGatherV operation reorders the inputs of all nodes within the communicator by rank ID (in ascending order), concatenates them, and then sends the result to the outputs of all nodes. Unlike the AllGather operation, AllGatherV supports configuring different data sizes for the inputs of different nodes within the communicator.

![AllGatherV](figures/allgatherv.png)

> [!NOTE]Note
> For the AllGatherV operation, each node receives the data set reordered by rank ID, meaning that the AllGatherV output of each node is identical.

Related API: [HcclAllGatherV](../api_ref/comm_op_interface/HcclAllGatherV.md).

### Reduce

The Reduce operation performs a reduction operation (supporting sum, prod, max, and min) on the input data of all ranks within the communicator, and then sends the result to the output buffer of the root node.

![Reduce](figures/reduce.png)

NOTE
Only one root node is allowed within the communicator.

Related API: [HcclReduce](../api_ref/comm_op_interface/HcclReduce.md).

### AllReduce

The AllReduce operation performs a reduction operation (supporting sum, prod, max, and min) on the input data of all nodes within the communicator, and then sends the result to the output buffer of all nodes.

![AllReduce](figures/allreduce.png)

Note: Each rank can have only one input.

Related interface: [HcclAllReduce](../api_ref/comm_op_interface/HcclAllReduce.md).

### ReduceScatter

The ReduceScatter operation divides the input data of all ranks within the communicator into rank size equal portions, and then performs a reduction operation (such as sum, prod, max, or min) on one portion taken from each rank. Finally, the results are scattered to the output buffer of each rank according to the rank number.

![ReduceScatter](figures/reduce_scatter.png)

Related interface: [HcclReduceScatter](../api_ref/comm_op_interface/HcclReduceScatter.md).

### ReduceScatterV

The ReduceScatterV operation is similar to the ReduceScatter operation, with the difference that it supports configuring different data volumes for different nodes within the communicator (the data size for different indices on the same rank can be set, but the data size for the same index across different ranks must remain consistent). After performing a reduction operation (such as sum, prod, max, or min) on the data corresponding to each index from each rank, the results are scattered to the output buffer of each rank according to the index.

![ReduceScatterV](figures/reduce_scatterv.png)

Related interface: [HcclReduceScatterV](../api_ref/comm_op_interface/HcclReduceScatterV.md).

### AlltoAll

The AlltoAll operation sends data of the same size to all ranks within the communicator and receives data of the same size from all ranks.

![AlltoAll](figures/alltoall.png)

The AlltoAll operation splits the input data into a specific number of chunks along a specific dimension, sends them to other ranks in order, receives input data from other ranks, and concatenates the data along a specific dimension in order.

Related API: [HcclAlltoAll](../api_ref/comm_op_interface/HcclAlltoAll.md).

### AlltoAllV

The AlltoAllV operation sends data (with customizable data volumes) to all ranks within the communicator and receives data from all ranks.

![AlltoAllV](figures/alltoallv.png)

Related API: [HcclAlltoAllV](../api_ref/comm_op_interface/HcclAlltoAllV.md).

### AlltoAllVC

The AlltoAllVC operation sends data (with customizable data volumes) to all ranks within the communicator and receives data from all ranks. Compared with AlltoAllV, AlltoAllVC passes the send and receive parameters of all ranks through the input parameter sendCountMatrix.

![AlltoAllVC](figures/alltoallvc.png)

Related interface: [HcclAlltoAllVC](../api_ref/comm_op_interface/HcclAlltoAllVC.md).

### Calling the API

The following uses the [HcclAllReduce](../api_ref/comm_op_interface/HcclAllReduce.md) interface as an example to describe its usage. The prototype definition of the HcclAllReduce interface is as follows:

```c
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

HcclAllReduce performs a reduction operation on the inputs of all nodes within the communicator and then sends the result to the output of all nodes. The op parameter specifies the type of reduction operation. HcclAllReduce allows each node to have only one input.

As shown in the following code snippet, a sum operation is performed on the data in all input memory within the communicator in the float32 data format (in the example, only one data element in each rank participates), and the sum result is then sent to the output memory of all nodes.

```c
void* hostBuf = nullptr;
void* sendBuf = nullptr;
void* recvBuf = nullptr;
uint64_t count = 1;
int malloc_kSize = count * sizeof(float);
aclrtStream stream;
aclrtCreateStream(&stream);

//Allocate memory for the collective communication operation.
aclrtMalloc((void**)&sendBuf, malloc_kSize, ACL_MEM_MALLOC_HUGE_ONLY); 
aclrtMalloc((void**)&recvBuf, malloc_kSize, ACL_MEM_MALLOC_HUGE_ONLY);

//Initialize the input memory.
aclrtMallocHost((void**)&hostBuf, malloc_kSize);
aclrtMemcpy((void*)sendBuf, malloc_kSize, (void*)hostBuf, malloc_kSize, ACL_MEMCPY_HOST_TO_DEVICE);

//Execute the collective communication operation.
HcclAllReduce((void *)sendBuf, (void*)recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
```

For a complete example of the HcclAllReduce API call, see the "HcclAllReduce Operation Code Sample" in the [Sample Code](#sample-code) section under different communicator initialization methods.

## Point-to-Point Communication

Point-to-point communication refers to a communication mode in which data is directly transmitted between two NPUs in a multi-NPU environment. It is commonly used for sending and receiving activation values in pipeline parallelism scenarios.

HCCL provides point-to-point communication operators at different granularities, including single-send/single-receive operators (Send/Receive) between a single rank and another single rank, and batch send/receive operators (BatchSendRecv) among multiple ranks. HCCL provides corresponding APIs for developers to invoke.

### Send/Receive (Single-Send/Single-Receive)

- Send: Sends data from one rank to another rank.

- Receive: Receives data sent from another rank.

HCCL provides the corresponding interfaces [HcclSend](../api_ref/comm_op_interface/HcclSend.md) and [HcclRecv](../api_ref/comm_op_interface/HcclRecv.md) for single-send/single-receive scenarios. They must be issued in strict order and used in pairs. The send and receive ends must complete synchronization before data transmission can proceed, and subsequent operator tasks can be executed only after data transmission is complete.

![SendRecv](figures/send_recv.png)

A simple code example is as follows:

```c
if(rankId == 0){
    uint32_t destRank = 1;
    uint32_t srcRank = 1;
    HcclSend(sendBuf, count, dataType, destRank, hcclComm, stream);
    HcclRecv(recvBuf, count, dataType, srcRank, hcclComm, stream);
}
if(rankId == 1){
    uint32_t srcRank = 0;
    uint32_t destRank = 0;
    HcclRecv(recvBuf, count, dataType, srcRank, hcclComm, stream);
    HcclSend(sendBuf, count, dataType, destRank, hcclComm, stream);
}
```

### BatchSendRecv (Batch Send and Receive)

HCCL provides the [HcclBatchSendRecv](../api_ref/comm_op_interface/HcclBatchSendRecv.md) interface for data sending and receiving among multiple ranks within the communicator. This interface has two characteristics:

- The interface internally reorders the sequence of batch data sending and receiving operations. Therefore, the order of batch send and receive tasks in a single interface call is not strictly required, but the number of data send operations and data receive operations in a single interface call must be exactly matched.

- The send and receive processes are scheduled and executed independently, and they do not block each other, thereby achieving full-duplex link concurrency.

**Note the following when using this interface**: In a single interface call, only one block of memory data can be transmitted in a unidirectional data flow between two ranks, to avoid ambiguity in the send and receive addresses of multiple blocks of memory data during the process.

A simple code snippet is as follows:

```c
HcclSendRecvItem sendRecvInfo[itemNum];
HcclSendRecvType currType;
for (size_t i = 0; i < op_type.size(); ++i) {
    if (op_type[i] == "isend") {
        currType = HcclSendRecvType::HCCL_SEND;
    } else if (op_type[i] == "irecv") {
        currType = HcclSendRecvType::HCCL_RECV;
    } 
    sendRecvInfo[i] = HcclSendRecvItem{currType,
                                       tensor_ptr_list[i],
                                       count_list[i],
                                       type_list[i],
                                       remote_rank_list[i]
                                       };
}
HcclBatchSendRecv(sendRecvInfo, itemNum, hcclComm, stream);
```

## Sample Code

HCCL provides sample code for implementing collective communication functions using the communication library API in different scenarios. Developers can select references based on actual requirements.

### Communicator Code Examples

- [Managing One NPU Device per Process (Initializing Communicator Based on Root Node Information)](https://gitcode.com/cann/hcomm/tree/9.1.0/examples/01_communicators/01_one_device_per_process/)

- [Managing One NPU Device per Process (Initializing Communicator Based on Rank Table)](https://gitcode.com/cann/hcomm/tree/9.1.0/examples/01_communicators/02_one_device_per_process_rank_table/)

- [Managing One NPU Device per Thread](https://gitcode.com/cann/hcomm/tree/9.1.0/examples/01_communicators/03_one_device_per_pthread/)

### Point-to-Point Communication Sample Code

- [HcclSend/HcclRecv (Basic Send and Receive)](https://gitcode.com/cann/hccl/tree/9.1.0/examples/01_point_to_point/01_send_recv/)

- [HcclBatchSendRecv (Implementing Ring Communication)](https://gitcode.com/cann/hccl/tree/9.1.0/examples/01_point_to_point/02_batch_send_recv_ring)

### Collective Communication Sample Code

- [AllReduce](../../../examples/02_collectives/01_allreduce)

- [Broadcast](../../../examples/02_collectives/02_broadcast)

- [AllGather](../../../examples/02_collectives/03_allgather)

- [ReduceScatter](../../../examples/02_collectives/04_reduce_scatter)

- [Reduce](../../../examples/02_collectives/05_reduce)

- [AlltoAll](../../../examples/02_collectives/06_alltoall)

- [AlltoAllV](../../../examples/02_collectives/07_alltoallv)

- [AlltoAllVC](../../../examples/02_collectives/08_alltoallvc)

- [Scatter](../../../examples/02_collectives/09_scatter)
