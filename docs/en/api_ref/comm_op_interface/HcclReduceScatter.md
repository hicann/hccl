# HcclReduceScatter

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T08:15:47.281Z pushedAt=2026-09-18T08:03:58.133Z -->

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

The operation API of the collective communication operator ReduceScatter, which divides the input data of all ranks in the communicator into *{ranksize}* portions, then takes one of the *{ranksize}* portions from each rank for a reduction operation (such as sum, prod, max, min). Finally, the results are scattered to the output buffer of each rank according to the rank indices.

![reducescatter](figures/reducescatter.png)

## Prototype

```c
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| recvBuf | Output | Address of the destination data buffer. The collective communication result is output to this buffer. |
| recvCount | Input | Data size of recvBuf participating in the ReduceScatter operation. The data size of sendBuf equals recvCount x rank size. |
| dataType | Input | Data type of the ReduceScatter operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [dataType Description](#datatype-description).|
| op | Input | Operation type of the Reduce operation.<br>Different models support different operation types. For details, see [op Description](#description).|
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by this rank. |

### dataType Description

<!-- npu="950" id10 -->
- For 950PR/950DT, supported data types: int8, int16, int32, int64, uint64, float16, float32, float64, bfp16.
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
<!-- npu="310p" id7 -->
- For Atlas 300I Duo, the supported operation types are sum, prod, max, and min. The prod, max, and min operations do not support the int16 data type.
<!-- end id7 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | A passed pointer parameter is null, for example, comm, sendBuf, recvBuf, or stream is nullptr. |
| HCCL_E_PARA | An invalid parameter is passed, for example, count exceeds the upper limit. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, dataType is invalid or not supported by the current model, or the prod operation does not support the int16/bfp16 data type. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

- The recvCount, dataType, and op must be the same across all ranks.
<!-- npu="310p" id8 -->
- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of 16 Atlas 300I Duo inference cards (i.e., 32 NPUs) per server.
<!-- end id8 -->
- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:

  - int8: 1-byte address alignment.
  - int16, float16, and bfp16: 2-byte address alignment.
  - int32 and float32: 4-byte address alignment.
  - int64, uint64, and float64: 8-byte address alignment.
- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.
- When symmetric memory is registered in a communicator, the input buffer pointed to by sendBuf is used as a symmetric window for directly reading from and writing into remote ranks. During ReduceScatter, some algorithms perform in-place reduction (read+reduce) on sendBuf, causing the data in sendBuf to be modified (corrupted). Therefore, ensure that:
  - After HcclReduceScatter is called, the data in sendBuf is no longer used as the original input;
  - If the original input data needs to be retained, back up sendBuf before the call.

## Example

```c
uint32_t rankSize = 8;
uint64_t recvCount = 1;  // Number of data elements received by each node
uint64_t sendSize = rankSize * recvCount * sizeof(float);
uint64_t recvSize = recvCount * sizeof(float);

// Allocate device memory for the collective communication operation.
void *sendBuf = nullptr, *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator and stream.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Execute ReduceScatter: Sum the sendBuf of all ranks, and evenly scatter the result to the recvBuf of each rank in rank_id order.
HcclReduceScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// Block and wait for the collective communication task in the task stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free memory on the device.
aclrtFree(recvBuf);          // Free memory on the device.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
