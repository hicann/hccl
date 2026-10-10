# HcclAllGatherV

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T07:37:40.484Z pushedAt=2026-09-18T08:03:58.109Z -->

## Applicable Products

<!-- npu="950" id1 -->
- 950PR/950DT: Supported
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
- Atlas training products: Not supported
<!-- end id5 -->

## Description

The operation API of the collective communication operator AllGatherV, which reorders the inputs of all nodes in the communicator by rank ID, concatenates them, and then sends the result to the outputs of all nodes.

Unlike the AllGather operator, the AllGatherV operator supports configuring different data sizes for the inputs of different nodes in the communicator.

![allgatherv](figures/allgatherv.png)

> [!NOTE]
> For AllGatherV operations, each node receives the dataset reordered by rank ID, meaning that the AllGatherV output is the same for every node.

## Prototype

```c
HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| sendCount | Input | Data size of sendBuf participating in the AllGatherV operation. |
| recvBuf | Output | Address of the destination data buffer where the collective communication result is output.<br>The address configured for recvBuf cannot be the same as that for sendBuf. |
| recvCounts | Input | Data size of each rank in recvBuf participating in the AllGatherV operation, as an array of the uint64 type.<br>The i-th element of the array indicates the amount of data to be received from rank i, which must be the same as the sendCount value of rank i. |
| recvDispls | Input | Offset (in units of dataType) of the data of each rank in recvBuf participating in the AllGatherV operation, as an array of the uint64 type.<br>The i-th element of the array indicates the start offset in recvBuf where the data received from rank i is placed. |
| dataType | Input | Data type of the AllGatherV operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types).|
| comm | Input | Communicator in which the collective communication operation is performed. |
| stream | Input | Stream used by the current rank. |

### Supported Data Types

<!-- npu="950" id11 -->
- For 950PR/950DT, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.
<!-- end id11 -->
<!-- npu="A3" id12 -->
- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id12 -->
<!-- npu="910b" id13 -->
- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id13 -->
<!-- npu="310p" id6 -->
- For Atlas 300I Duo, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id6 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | A passed pointer parameter is nullptr, such as comm, recvCounts, recvDispls, or stream (sendBuf cannot be nullptr when sendCount is greater than 0, and recvBuf cannot be nullptr when recvCounts are not all zeros). |
| HCCL_E_PARA | A passed parameter is invalid, for example, `count` exceeds the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, `dataType` is invalid or not supported by the current model. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The recvCounts, recvDispls, and dataType must be identical across all ranks.
<!-- npu="A3" id15 -->
- For Atlas A3 training products and Atlas A3 inference products, only single-server use cases are supported.
<!-- end id15 -->
<!-- npu="910b" id16 -->
- For Atlas A2 training products and Atlas A2 inference products, only multi-server symmetric deployments are supported. Asymmetric deployments (i.e., asymmetric device counts) are not supported.
<!-- end id16 -->
<!-- npu="310p" id10 -->
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of two Atlas 300I Duo inference cards (i.e., four NPUs) per server.
<!-- end id10 -->
- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.

## Example

```c
// Allocate device memory for the collective communication operation.
uint32_t rankSize = 8;
uint64_t sendCount = 1;  // Number of data elements sent by each rank
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = rankSize * sendCount * sizeof(float);

void *sendBuf = nullptr;
void *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Set recvCounts and recvDispls. Each rank receives the same amount of data.
std::vector<uint64_t> recvCounts(rankSize, sendCount);
std::vector<uint64_t> recvDispls(rankSize);
for (uint32_t i = 0; i < rankSize; ++i) {
    recvDispls[i] = i * sendCount;
}

// Initialize the communicator.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Run AllGatherV. Concatenate sendBuf of all ranks in the communicator after reordering by rank ID, and send the result to recvBuf of all ranks.
HcclAllGatherV(sendBuf, sendCount, recvBuf, recvCounts.data(), recvDispls.data(), HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Block and wait for the collective communication tasks in the stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free the memory on the device.
aclrtFree(recvBuf);          // Free the memory on the device.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
