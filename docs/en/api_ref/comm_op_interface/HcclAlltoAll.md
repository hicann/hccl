# HcclAlltoAll

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-04T09:50:00.159Z pushedAt=2026-08-11T03:39:32.566Z -->

## Applicable Products

<!-- npu="950" id1 -->

- Ascend 950PR/Ascend 950DT: Supported

<!-- end id1 -->
<!-- npu="A3" id2 -->

- Atlas A3 training products/Atlas A3 inference products: Supported

<!-- end id2 -->
<!-- npu="910b" id3 -->

- Atlas A2 training products/Atlas A2 inference products: Supported

<!-- end id3 -->
<!-- npu="310p" id4 -->

- Atlas inference products: Supported

<!-- end id4 -->
<!-- npu="910" id5 -->

- Atlas training products: Supported

<!-- end id5 -->

## Function

The operation API for the collective communication operator AlltoAll, which sends data of the same size to all ranks in the communicator and receives data of the same size from all ranks.

![alltoall](figures/alltoall.png)

The AlltoAll operation splits the input data into a specific number of blocks along a specific dimension, sends them to other ranks in order, and simultaneously receives input data from other ranks, concatenating the data along the specific dimension in order.

## Function Prototype

```c
HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType, const void *recvBuf, uint64_t recvCount, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| sendCount | Input | Amount of data sent to each rank. |
| sendType | Input | Data type of the send data, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types). |
| recvBuf | Output | Destination data buffer address, where the collective communication result is output.<br>The addresses configured for recvBuf and sendBuf cannot be the same. |
| recvCount | Input | Amount of data received from each rank, which must be the same as the sendCount value. |
| recvType | Input | Data type of the receive data, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type, which must be the same as the sendType value. |
| comm | Input | Communicator where the collective communication operation takes place. |
| stream | Input | Stream used by this rank. |

### Supported Data Types

- For Ascend 950PR/Ascend 950DT, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.

- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas training products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.

- For Atlas 300I Duo, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The sendCount, sendType, recvCount, and recvType of all ranks must be the same.

- The performance of the AlltoAll operation depends on the buffer size for data sharing between NPUs. When the communication data volume exceeds the buffer size, performance will degrade significantly. If the AlltoAll communication data volume in your service is large, you are advised to configure the environment variable [HCCL_BUFFSIZE](../../user_guide/hccl_env/HCCL_BUFFSIZE.md) to appropriately increase the buffer size to improve communication performance.

<!-- npu="910" id13 -->

- For Atlas training products, the AlltoAll communicator must meet the following constraints:

    For a single server, 1p and 2p communicators must be within the same cluster (devices 0-3 and devices 4-7 in a server each form a cluster). For single-server 4p and 8p and multi-server communicators, ranks must be organized by cluster as the basic unit, and the cluster selection across servers must be consistent.

- For Atlas training products, in single-server use cases, the NIC status must be "up"; otherwise, this API will fail to execute.

<!-- end id13 -->
<!-- npu="310p" id14 -->

- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of two Atlas 300I Duo inference cards (i.e., four NPUs) per server.

<!-- end id14 -->

## Example

```c
// Apply for device memory for the collective communication operation.
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t count = 8;
size_t mallocSize = count * sizeof(float);
aclrtMalloc((void **)&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc((void **)&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Execute AlltoAll to send data of the same size to all ranks in the communicator and receive data of the same size from all ranks.
size_t perCount = count / rankSize;
HcclAlltoAll(sendBuf, perCount, HCCL_DATA_TYPE_FP32, recvBuf, perCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Block and wait for the collective communication task in the task stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Release device memory.
aclrtFree(recvBuf);          // Release device memory.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
