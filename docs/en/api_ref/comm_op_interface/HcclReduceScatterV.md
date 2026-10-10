# HcclReduceScatterV

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T08:20:46.571Z pushedAt=2026-09-18T08:03:58.136Z -->

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

The operation API of the collective communication operator ReduceScatterV, which is similar to ReduceScatter, except that it allows different nodes within the communicator to be configured with different data sizes (the data size for different indices on the same rank can be set, but the data size for the same index across different ranks must be consistent). It performs a reduction operation (supporting sum, prod, max, and min) on the data corresponding to each index across all ranks, and then scatters the results to the output buffer of each rank by index.

![reducescatterv](figures/reducescatterv.png)

## Prototype

```c
HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| sendCounts | Input | Size of the data in sendBuf of each rank participating in the ReduceScatterV operation, in an array of the uint64 type.<br>The i-th element of this array indicates the amount of data to send to rank i. |
| sendDispls | Input | Offset of the data of each rank participating in the ReduceScatterV operation in sendBuf (in the unit of dataType), in an array of the uint64 type.<br>The i-th element of this array indicates the offset of the data sent to rank i in sendBuf. |
| recvBuf | Output | Address of the destination data buffer. The collective communication result is output to this buffer.<br>recvBuf and sendBuf cannot be set to the same address. |
| recvCount | Input | Size of the data in recvBuf corresponding to the rank participating in the ReduceScatterV operation.<br>Assume that the ID of the current rank is i. The value of recvCount must be the same as that of the element with subscript i in the sendCounts array. |
| dataType | Input | Data type of the ReduceScatterV operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [dataType Description](#datatype-description).|
| op | Input | Type of the Reduce operation.<br>Different models support different operation types. For details, see [op Description](#description).|
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by the current rank. |

### dataType Description

<!-- npu="950" id10 -->
- For 950PR/950DT, the supported data types are int8, int16, int32, int64, float16, float32, and bfp16.
<!-- end id10 -->
<!-- npu="A3" id11 -->
- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, int16, int32, int64, float16, float32, bfp16.
<!-- end id11 -->
<!-- npu="910b" id12 -->
- For Atlas A2 training products/Atlas A2 inference products, the supported data types are int8, int16, int32, float16, float32, and bfp16.
<!-- end id12 -->
<!-- npu="310p" id6 -->
- For Atlas 300I Duo inference card, the supported data types are int16, float16, and float32.
<!-- end id6 -->

### Operation Types

<!-- npu="950" id14 -->
- For 950PR/950DT, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.
<!-- end id14 -->
<!-- npu="A3" id15 -->
- For Atlas A3 training products/Atlas A3 inference products, the supported operation types are sum, max, and min.
<!-- end id15 -->
<!-- npu="910b" id16 -->
- For Atlas A2 training products/Atlas A2 inference products, the supported operation types are sum, max, and min.
<!-- end id16 -->
<!-- npu="310p" id7 -->
- For Atlas 300I Duo, only the sum operation type is supported.
<!-- end id7 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | A passed pointer parameter is null, for example, comm, sendCounts, sendDispls, or stream is nullptr (recvBuf cannot be nullptr when recvCount is greater than 0). |
| HCCL_E_PARA | A passed parameter is invalid, for example, count exceeds the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, dataType is invalid or not supported by the current model, or the prod operation does not support the int16/bfp16 data type. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The sendCounts, sendDispls, dataType, and op must be the same across all ranks.
<!-- npu="A3" id18 -->
- For Atlas A3 training products and Atlas A3 inference products, only single-server use cases are supported.
<!-- end id18 -->
<!-- npu="910b" id19 -->
- For Atlas A2 training products and Atlas A2 inference products, only multi-server symmetric deployments are supported. Asymmetric deployments (i.e., asymmetric device counts) are not supported.
<!-- end id19 -->
<!-- npu="310p" id8 -->
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of two Atlas 300I Duo inference cards (i.e., four NPUs) per server.
<!-- end id8 -->
- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:
  - int8: 1-byte address alignment.
  - int16, float16, and bfp16: 2-byte address alignment.
  - int32 and float32: 4-byte address alignment.
  - int64: 8-byte address alignment.
- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.

## Example

```c
// Allocate device memory for the collective communication operation.
uint32_t rankSize = 8;
uint64_t recvCount = 1;  // Number of data units received by each rank
size_t recvSize = recvCount * sizeof(float);
size_t totalSendCount = rankSize * recvCount;
size_t sendSize = totalSendCount * sizeof(float);

void *sendBuf = nullptr;
void *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Set sendCounts and sendDispls. Each rank sends the same amount of data.
std::vector<uint64_t> sendCounts(rankSize, recvCount);
std::vector<uint64_t> sendDispls(rankSize);
for (uint32_t i = 0; i < rankSize; ++i) {
    sendDispls[i] = i * recvCount;
}

// Initialize the communicator.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Execute ReduceScatterV: Sum the sendBuf of all ranks and then scatter the result to the recvBuf of each rank according to the rank ID.
HcclReduceScatterV(sendBuf, sendCounts.data(), sendDispls.data(), recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// Block until the collective communication task in the stream is complete.
aclrtSynchronizeStream(stream);

// Release resources
aclrtFree(sendBuf);          // Free device-side memory
aclrtFree(recvBuf);          // Free device-side memory.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
