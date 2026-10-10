# HcclAllReduce

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T07:41:36.044Z pushedAt=2026-09-18T08:03:58.111Z -->

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

The operation API of the collective communication operator AllReduce, which adds (or performs other reduction operations on) the input data of all nodes in the communicator, and then sends the result to the output buffer of all nodes. The reduction operation type is specified by the `op` parameter.

![allreduce](figures/allreduce.png)

## Prototype

```c
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Pointer to the source data buffer. |
| recvBuf | Output | Pointer to the destination data buffer. The collective communication result is output to this buffer. |
| count | Input | Number of data units involved in the AllReduce operation. For example, `count` is `1` if only one int32 data unit is involved. |
| dataType | Input | Data type of the AllReduce operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [dataType Description](#datatype-description).|
| op | Input | Operation type of the Reduce operation.<br>Different models support different operation types. For details, see [op Description](#description).|
| comm | Input | Communicator in which the collective communication operation is performed. |
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
<!-- npu="310p" id6 -->
- For Atlas 300I Duo, supported data types: int8, int16, int32, float16, float32.
<!-- end id6 -->

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
<!-- npu="310p" id7 -->
- For Atlas 300I Duo, the supported operation types are sum, prod, max, and min. The prod, max, and min operations do not support the int16 data type.
<!-- end id7 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | A passed pointer parameter is nullptr, such as comm, sendBuf, recvBuf, or stream. |
| HCCL_E_PARA | A passed parameter is invalid, for example, `count` exceeds the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, `dataType` is invalid or not supported by the current model, or the prod operation does not support the int16/bfp16 data type. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The count, dataType, and op must be the same across all ranks.
- Each rank has only one input.
- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:
  - int8: 1-byte address alignment.
  - int16, float16, and bfp16: 2-byte address alignment.
  - int32 and float32: 4-byte address alignment.
  - int64, uint64, and float64: 8-byte address alignment.
- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.
- When symmetric memory is registered in a communicator, the input buffer pointed to by sendBuf is used as a symmetric window for directly reading from and writing into remote ranks. During ReduceScatter, some algorithms perform in-place reduction (read+reduce) on sendBuf, causing the data in sendBuf to be modified (corrupted). Therefore, ensure that:
  - After HcclAllReduce is called, the data in sendBuf is no longer used as the original input;
  - To retain the original input data, back up sendBuf before the call.

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

// Perform AllReduce: Sum the input data of all nodes in the communicator, and send the result to the output buffer of all nodes.
HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// Block and wait for the collective communication tasks in the task stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free memory on the device.
aclrtFree(recvBuf);          // Free memory on the device.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
