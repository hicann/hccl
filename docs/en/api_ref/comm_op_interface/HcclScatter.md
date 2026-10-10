# HcclScatter

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T08:24:49.325Z pushedAt=2026-09-18T08:03:58.138Z -->

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
- Atlas inference products: Not supported
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas training products: Supported
<!-- end id5 -->

## Description

The operation API of the collective communication operator Scatter, which evenly scatters data from the root node to other ranks.

## Prototype

```c
HcclResult HcclScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| recvBuf | Output | Address of the destination data buffer. The collective communication result is output to this buffer. |
| recvCount | Input | Number of data units in recvBuf that participate in the scatter operation. For example, `recvcount` is `1` if only one int32 element participates. |
| dataType | Input | Data type for the scatter operation, of type [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md).<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types).|
| root | Input | Rank ID that serves as the scatter root. |
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by this rank. |

### Supported Data Types

<!-- npu="950" id10 -->
- For 950PR/950DT, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.
<!-- end id10 -->
<!-- npu="A3" id11 -->
- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id11 -->
<!-- npu="910b" id12 -->
- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.
<!-- end id12 -->
<!-- npu="910" id13 -->
- For Atlas training products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id13 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | A passed pointer parameter is null. For example, comm or recvBuf is nullptr (the root node's sendBuf cannot be nullptr). |
| HCCL_E_PARA | A passed parameter is invalid. For example, count exceeds the upper limit, or root is out of range. |
| HCCL_E_NOT_SUPPORT | The operation is not supported. For example, dataType is invalid or not supported by the current model, or Scatter is not supported in hybrid networking. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- recvCount, dataType, and root must be the same across all ranks.
- There can be only one root node globally.
- The sendBuf of a non-root node can be null. The sendBuf of the root node cannot be null.
- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.

## Example

```c
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t sendCount = 8;
uint64_t recvCount = 1;
size_t sendSize = sendCount * sizeof(float);
size_t recvSize = recvCount * sizeof(float);

// Allocate device memory for receiving the Scatter result.
ACLCHECK(aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY));
// On the root node, allocate device memory for storing the data to send.
if (device == rootRank) {
    ACLCHECK(aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY));
}

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, device, &hcclComm);

// Create a stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Perform Scatter to evenly divide the root node's data in the communicator and scatter it to other ranks.
HcclScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, rootRank, hcclComm, stream);
// Block until the collective communication tasks in the stream complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device-side memory.
aclrtFree(recvBuf);          // Free device-side memory.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator
```
