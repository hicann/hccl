# HcclAllGather

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T07:34:14.410Z pushedAt=2026-09-18T08:03:58.107Z -->

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
- Atlas training products: Supported
<!-- end id5 -->

## Description

The operation API of the collective communication operator AllGather, which reorders the inputs of all nodes in the communicator by rank ID, concatenates them, and then sends the result to the outputs of all nodes.

![allgather](figures/allgather.png)

> [!NOTE]
> For AllGather operations, each node receives the dataset reordered by rank ID, meaning that the AllGather output is the same for every node.

## Prototype

```c
HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| recvBuf | Output | Address of the destination data buffer. The collective communication result is output to this buffer. |
| sendCount | Input | Data size of sendBuf participating in the AllGather operation. The data size of recvBuf equals sendCount x rank size. |
| dataType | Input | The data type of the AllGather operation, of type [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md).<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types).|
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by the current rank. |

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
<!-- npu="310p" id6 -->
- For Atlas 300I Duo, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id6 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | The passed pointer parameter is null, e.g., comm, sendBuf, recvBuf, or stream is nullptr. |
| HCCL_E_PARA | The passed parameter is invalid, e.g., `count` exceeds the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, e.g., `dataType` is invalid or not supported by the current model. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The sendCount and dataType must be the same across all ranks.
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of 16 Atlas 300I Duo inference cards (i.e., 32 NPUs) per server.
- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.

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

// Create the task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Perform AllGather to concatenate sendBuf of all ranks in the communicator in rank_id order, and send the result to recvBuf of all ranks.
HcclAllGather(sendBuf, recvBuf, sendCount, HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Block and wait for the collective communication tasks in the task stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free memory on the device.
aclrtFree(recvBuf);          // Free memory on the device.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
