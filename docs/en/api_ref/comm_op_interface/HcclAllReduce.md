# HcclAllReduce

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-04T09:50:08.176Z pushedAt=2026-08-05T02:40:26.134Z -->

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

The operation API of the collective communication operator AllReduce, which adds (or performs other reduction operations on) the input data of all nodes in the communicator, and then sends the result to the output buffer of all nodes. The reduction operation type is specified by the `op` parameter.

![allreduce](figures/allreduce.png)

## Function Prototype

```c
HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameters

| Name | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| recvBuf | Output | Destination data buffer address, where the collective communication result is output. |
| count | Input | Number of data elements participating in the allreduce operation. For example, if only one int32 data element participates, count=1. |
| dataType | Input | Data type of the AllReduce operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [dataType Description](#datatype-description). |
| op | Input | Type of the reduce operation.<br>Different models support different operation types. For details, see [Operation Description](#operation-description). |
| comm | Input | Communicator where the collective communication operation takes place. |
| stream | Input | Stream used by the current rank. |

### dataType Description

- For Ascend 950PR/Ascend 950DT, supported data types: int8, int16, int32, int64, uint64, float16, float32, float64, bfp16. For int64, uint64, and float64, only intra-node communication is currently supported.

- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, int16, int32, int64, float16, float32, bfp16.

- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, int16, int32, int64, float16, float32, bfp16. Note that the int64 data type may experience some performance degradation.

- For Atlas training products, supported data types: int8, int32, int64, float16, float32.

- For Atlas 300I Duo inference card, supported data types: int8, int16, int32, float16, float32.

### Operation Description

- For Ascend 950PR/Ascend 950DT, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.

- For Atlas A3 training products/Atlas A3 inference products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.

- For Atlas A2 training products/Atlas A2 inference products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.

- For Atlas training products, the supported operation types are sum, prod, max, and min.

- For Atlas 300I Duo, the supported operation types are sum, prod, max, and min, where prod, max, and min operations do not support the int16 data type.

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The count, dataType, and op must be identical across all ranks.

- Each rank has only one input.

- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:

  - int8: 1-byte address alignment.

  - int16, float16, and bfp16: 2-byte address alignment.

  - int32 and float32: 4-byte address alignment.

  - int64, uint64, and float64: 8-byte address alignment.

## Example

```c
// Apply for device memory for collective communication operations.
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

// Execute AllReduce to sum the input data of all nodes in the communicator, and then send the result to the output buffer of all nodes.
HcclAllReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// Block and wait for the collective communication task in the task stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Release the memory on the device.
aclrtFree(recvBuf);          // Release the memory on the device.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
