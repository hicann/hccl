# HcclAllGather

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-04T08:28:31.997Z pushedAt=2026-08-04T09:45:55.266Z -->

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

The operation interface of the collective communication operator AllGather, which reorders the inputs of all nodes in the communicator by rank ID, concatenates them, and then sends the result to the outputs of all nodes.

![allgather](figures/allgather.png)

> [!NOTE] Note
> For AllGather operations, each node receives the dataset reordered by rank ID, meaning that the AllGather output is the same for every node.

## Function Prototype

```c
HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Name | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| recvBuf | Output | Destination data buffer address, where the collective communication result is output. |
| sendCount | Input | The data size of sendBuf participating in the AllGather operation. The data size of recvBuf equals sendCount x rank size. |
| dataType | Input | The data type of the AllGather operation, of type [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md).<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types). |
| comm | Input | The communicator where the collective communication operation takes place. |
| stream | Input | The stream used by this rank. |

### Supported Data Types

- For Ascend 950PR/Ascend 950DT, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.

- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas training products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.

- For Atlas 300I Duo, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The sendCount and dataType must be the same across all ranks.

- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of 16 Atlas 300I Duo inference cards (i.e., 32 NPUs) per server.

## Example

```c
// Allocate device memory for the collective communication operation.
void *sendBuf = nullptr, *recvBuf = nullptr;
uint32_t rankSize = 8;
uint64_t sendCount = 1;  // Number of data elements sent by each node.
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = rankSize * sendCount * sizeof(float);
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator and stream.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, devId, &hcclComm);

// Create a task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Execute AllGather to concatenate sendBuf of all ranks in the communicator in rank_id order, and then send the result to recvBuf of all ranks.
HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Block and wait for the collective communication task in the task stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Release device memory.
aclrtFree(recvBuf);          // Release device memory.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
