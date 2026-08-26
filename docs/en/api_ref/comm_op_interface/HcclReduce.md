# HcclReduce

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-04T09:50:40.834Z pushedAt=2026-08-05T02:40:26.141Z -->

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

- Atlas inference products: Not supported

<!-- end id4 -->
<!-- npu="910" id5 -->

- Atlas training products: Supported

<!-- end id5 -->

## Function

The operation API for the collective communication operator Reduce, which adds (or performs other reduce operations on) data from all ranks and then sends the result to the specified position on the root node.

![reduce](figures/reduce.png)

## Function Prototype

```c
HcclResult HcclReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| recvBuf | Output | Destination data buffer address, where the collective communication result is output. |
| count | Input | Number of data units involved in the reduce operation. For example, if only one int32 data unit is involved, count=1. |
| dataType | Input | Data type of the reduce operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [dataType Description](#datatype-description). |
| op | Input | Reduce operation type.<br>Different models support different operation types. For details, see [op Description](#op-description). |
| root | Input | Rank ID for the reduce root. |
| comm | Input | Communicator where the collective communication operation takes place. |
| stream | Input | Stream used by the current rank. |

### dataType Description

- For Ascend 950PR/Ascend 950DT, supported data types: int8, int16, int32, int64, uint64, float16, float32, float64, bfp16. For int64, uint64, and float64, only intra-node communication is currently supported.

- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, int16, int32, int64, float16, float32, bfp16.

- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, int16, int32, int64, float16, float32, bfp16. Note that performance may degrade to some extent for the int64 data type.

- For Atlas training products, supported data types: int8, int32, int64, float16, float32.

### op Description

- For Ascend 950PR/Ascend 950DT, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 or bfp16 data types.

- For Atlas A3 training products/Atlas A3 inference products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 or bfp16 data types.

- For Atlas A2 training products/Atlas A2 inference products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 or bfp16 data types.

- For Atlas training products, the supported operation types are sum, prod, max, and min.

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The count, dataType, and op must be the same across all ranks.

- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:

  - int8: 1-byte address alignment.

  - int16, float16, bfp16: 2-byte address alignment.

  - int32, float32: 4-byte address alignment.

  - int64, uint64, and float64: 8-byte address alignment.

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

// Execute Reduce, add the sendBuf values at corresponding positions across all ranks, and then send the result to the recvBuf of the root node.
HcclReduce(sendBuf, recvBuf, count, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, rootRank, hcclComm, stream);
// Block and wait for the collective communication task in the stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Release memory on the device.
aclrtFree(recvBuf);          // Release memory on the device.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
