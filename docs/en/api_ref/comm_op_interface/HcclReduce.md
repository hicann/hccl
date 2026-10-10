# HcclReduce

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T08:11:16.061Z pushedAt=2026-09-18T08:03:58.130Z -->

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

The operation API for the collective communication operator Reduce, which adds (or performs other reduce operations on) data from all ranks and then sends the result to the specified position on the root node.

![reduce](figures/reduce.png)

## Prototype

```c
HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| recvBuf | Output | Address of the destination data buffer. The collective communication result is output to this buffer. |
| count | Input | Number of data units participating in the reduce operation. For example, `count` is `1` if only one int32 data unit participates. |
| dataType | Input | Data type of the reduce operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [dataType Description](#datatype-description).|
| op | Input | Operation type of the reduce operation.<br>Different models support different operation types. For details, see [op Description](#description).|
| root | Input | Rank ID serving as the reduce root. |
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by the current rank. |

### dataType Description

<!-- npu="950" id10 -->
- For 950PR/950DT, supported data types: int8, int16, int32, int64, uint64, float16, float32, float64, and bfp16.
<!-- end id10 -->
<!-- npu="A3" id11 -->
- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, int16, int32, int64, float16, float32, bfp16.
<!-- end id11 -->
<!-- npu="910b" id12 -->
- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, int16, int32, int64, float16, float32, bfp16. Note that for int64, performance may degrade to certain extent.
<!-- end id12 -->
<!-- npu="910" id13 -->
- For Atlas training products, supported data types: int8, int32, int64, float16, float32.
<!-- end id13 -->

### Operation Types

<!-- npu="950" id14 -->
- For 950PR/950DT, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.
<!-- end id14 -->
<!-- npu="A3" id15 -->
- For Atlas A3 training products/Atlas A3 inference products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.
<!-- end id15 -->
<!-- npu="910b" id16 -->
- For Atlas A2 training products/Atlas A2 inference products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.
<!-- end id16 -->
<!-- npu="910" id17 -->
- For Atlas training products, the supported operation types are sum, prod, max, and min.
<!-- end id17 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | The passed pointer parameter is null, for example, comm, sendBuf, recvBuf, or stream is nullptr. |
| HCCL_E_PARA | The passed parameter is invalid, for example, count exceeds the upper limit or root is out of range. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, dataType is invalid or not supported by the current model, or the prod operation does not support the int16/bfp16 data type. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The count, dataType, and op must be the same across all ranks.
- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:
  - int8: 1-byte address alignment.
  - int16, float16, and bfp16: 2-byte address alignment.
  - int32 and float32: 4-byte address alignment.
  - int64, uint64, and float64: 8-byte address alignment.
- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.

## Example

```c
// Allocate device memory for collective communication operations.
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

// Execute Reduce. Add the sendBuf data of all ranks at corresponding positions, and send the result to the recvBuf of the root node.
HcclReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, rootRank, hcclComm, stream);
// Block and wait for the collective communication tasks in the task stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device-side memory.
aclrtFree(recvBuf);          // Free device-side memory.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
