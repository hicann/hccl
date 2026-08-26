# HcclReduceScatter

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-04T09:50:45.923Z pushedAt=2026-08-11T03:40:58.103Z -->

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

The operation API of the collective communication operator ReduceScatter, which divides the input data of all ranks in the communication domain into *{ranksize}* portions, then takes one of the *{ranksize}* portions from each rank for a reduction operation (such as sum, prod, max, min). Finally, the results are scattered to the output buffer of each rank according to the rank indices.

![reducescatter](figures/reducescatter.png)

## Function Prototype

```c
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameters

| Name | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| recvBuf | Output | Destination data buffer address, where the collective communication result is output. |
| recvCount | Input | Data size of recvBuf participating in the ReduceScatter operation. The data size of sendBuf equals recvCount x rank size. |
| dataType | Input | Data type of the ReduceScatter operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [dataType Description](#datatype-description). |
| op | Input | Operation type of Reduce.<br>Different models support different operation types. For details, see [Operation Types](#operation-types). |
| comm | Input | Communicator where the collective communication operation resides. |
| stream | Input | Stream used by this rank. |

### dataType Description

- For Ascend 950PR/Ascend 950DT, supported data types: int8, int16, int32, int64, uint64, float16, float32, float64, bfp16. For int64, uint64, and float64, only intra-node communication is supported.

- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, int16, int32, int64, float16, float32, bfp16.

- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, int16, int32, int64, float16, float32, bfp16. Note that for int64, performance may degrade to certain extent.

- For Atlas training products, supported data types: int8, int32, int64, float16, float32.

- For Atlas 300I Duo, supported data types: int8, int16, int32, float16, float32.

### Operation Types

- For Ascend 950PR/Ascend 950DT, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.

- For Atlas A3 training products/Atlas A3 inference products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.

- For Atlas A2 training products/Atlas A2 inference products, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 and bfp16 data types.

- For Atlas 300I Duo, the supported operation types are sum, prod, max, and min. The prod, max, and min operations do not support the int16 data type.

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The recvCount, dataType, and op must be the same across all ranks.

<!-- npu="310p" id12 -->

- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of 16 Atlas 300I Duo inference cards (i.e., 32 NPUs) per server.

<!-- end id12 -->

- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:

  - int8: 1Byte address alignment.

  - int16, float16, and bfp16: 2Byte address alignment.

  - int32 and float32: 4Byte address alignment.

  - int64, uint64, and float64: 8Byte address alignment.

## Example

```c
uint32_t rankSize = 8;
uint64_t recvCount = 1;  // Amount of data received by each node.
uint64_t sendSize = rankSize * recvCount * sizeof(float);
uint64_t recvSize = recvCount * sizeof(float);

// Apply for device memory for the collective communication operation.
void *sendBuf = nullptr, *recvBuf = nullptr;
aclrtMalloc(&sendBuf, sendSize, ACL_MEM_MALLOC_HUGE_ONLY);
aclrtMalloc(&recvBuf, recvSize, ACL_MEM_MALLOC_HUGE_ONLY);

// Initialize the communicator and stream.
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Execute ReduceScatter, sum the sendBuf of all ranks, and then evenly distribute the result to the recvBuf of each rank in rank_id order.
HcclReduceScatter(sendBuf, recvBuf, recvCount, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, hcclComm, stream);
// Block and wait for the collective communication task in the stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Release memory on the device.
aclrtFree(recvBuf);          // Release memory on the device.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
