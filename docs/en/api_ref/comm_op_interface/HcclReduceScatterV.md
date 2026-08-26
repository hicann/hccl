# HcclReduceScatterV

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-04T09:51:16.596Z pushedAt=2026-08-05T02:40:26.159Z -->

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

- Atlas training products: Not supported

<!-- end id5 -->

## Function

The operation API of the collective communication operator ReduceScatterV, which is similar to ReduceScatter, except that it allows different nodes within the communicator to be configured with different data sizes (the data size for different indices on the same rank can be set, but the data size for the same index across different ranks must be consistent). It performs a reduction operation (supporting sum, prod, max, and min) on the data corresponding to each index across all ranks, and then scatters the results to the output buffer of each rank by index.

![reducescatterv](figures/reducescatterv.png)

## Function Prototype

```c
HcclResult HcclReduceScatterV(void *sendBuf, const void *sendCounts, const void *sendDispls, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```

## Parameters

| Name | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| sendCounts | Input | Data size in sendBuf for each rank participating in the ReduceScatterV operation, as an array of the uint64 type.<br>The i-th element of this array indicates the amount of data to send to rank i. |
| sendDispls | Input | Offset (in units of dataType) of the data for each rank participating in the ReduceScatterV operation within sendBuf, as an array of the uint64 type.<br>The i-th element of this array indicates the offset of the data to send to rank i within sendBuf. |
| recvBuf | Output | Destination data buffer address, to which the collective communication result is output.<br>The addresses configured for recvBuf and sendBuf must differ. |
| recvCount | Input | Data size in recvBuf for the rank participating in the ReduceScatterV operation.<br>Assuming the current rank number is *i*, the value of recvCount must be the same as the value of the element at index *i* in the sendCounts array. |
| dataType | Input | Data type for the ReduceScatterV operation, of the [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [dataType Description](#datatype-description). |
| op | Input | Reduction operation type.<br>Different models support different operation types. For details, see [op Description](#op-description). |
| comm | Input | Communicator in which the collective communication operation resides. |
| stream | Input | Stream used by the current rank. |

### dataType Description

- For Ascend 950PR/Ascend 950DT, the supported data types are int8, int16, int32, int64, float16, float32, and bfp16.

- For Atlas A3 training products/Atlas A3 inference products, the supported data types are int8, int16, int32, int64, float16, float32, and bfp16.

- For Atlas A2 training products/Atlas A2 inference products, the supported data types are int8, int16, int32, float16, float32, and bfp16.

- For Atlas 300I Duo inference card, the supported data types are int16, float16, and float32.

### op Description

- For Ascend 950PR/Ascend 950DT, the supported operation types are sum, prod, max, and min. The prod operation does not support int16 or bfp16 data types.

- For Atlas A3 training products/Atlas A3 inference products, the supported operation types are sum, max, and min.

- For Atlas A2 training products/Atlas A2 inference products, the supported operation types are sum, max, and min.

- For Atlas 300I Duo, only the sum operation type is supported.

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The sendCounts, sendDispls, dataType, and op must be the same across all ranks.

- For Atlas A3 training products and Atlas A3 inference products, only single-server use cases are supported.

- For Atlas A2 training products and Atlas A2 inference products, only multi-server symmetric deployments are supported. Asymmetric deployments (i.e., asymmetric device counts) are not supported.

- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of two Atlas 300I Duo inference cards (i.e., 4 NPUs) per server.

- The input and output addresses (sendBuf and recvBuf) of the operator must meet the following alignment requirements based on the data type:

  - int8: 1Byte address alignment.

  - int16, float16, and bfp16: 2Byte address alignment.

  - int32 and float32: 4Byte address alignment.

  - int64: 8Byte address alignment.
  