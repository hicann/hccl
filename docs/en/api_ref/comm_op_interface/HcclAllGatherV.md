# HcclAllGatherV

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-04T09:50:04.750Z pushedAt=2026-08-05T02:40:26.129Z -->

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

The operation API of the collective communication operator AllGatherV, which reorders the inputs of all nodes in the communicator by rank ID, concatenates them, and then sends the result to the outputs of all nodes.

Unlike the AllGather operator, the AllGatherV operator supports configuring different data sizes for the inputs of different nodes in the communicator.

![allgatherv](figures/allgatherv.png)

> [!NOTE] Note
> For AllGatherV operations, each node receives the dataset reordered by rank ID, meaning that the AllGatherV output is the same for every node.

## Function Prototype

```c
HcclResult HcclAllGatherV(void *sendBuf, uint64_t sendCount, void *recvBuf, const void *recvCounts, const void *recvDispls, HcclDataType dataType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Name | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Source data buffer address. |
| sendCount | Input | Data size of sendBuf participating in the AllGatherV operation. |
| recvBuf | Output | Destination data buffer address, where the collective communication result is output.<br>The addresses configured for recvBuf and sendBuf cannot be the same. |
| recvCounts | Input | Data size of each rank in recvBuf participating in the AllGatherV operation, as a uint64 array.<br>The i-th element of this array indicates the amount of data to be received from rank i, and this amount must be the same as the sendCount value of rank i. |
| recvDispls | Input | Offset (in dataType units) of each rank's data in recvBuf participating in the AllGatherV operation, as a uint64 array.<br>The i-th element of this array indicates the starting offset in recvBuf where the data received from rank i should be placed. |
| dataType | Input | Data type of the AllGatherV operation, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types). |
| comm | Input | Communicator where the collective communication operation resides. |
| stream | Input | Stream used by this rank. |

### Supported Data Types

- For Ascend 950PR/Ascend 950DT, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.

- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas 300I Duo, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The recvCounts, recvDispls, and dataType must be identical across all ranks.

- For Atlas A3 training products and Atlas A3 inference products, only single-server use cases are supported.

- For Atlas A2 training products and Atlas A2 inference products, only multi-server symmetric deployments are supported. Asymmetric deployments (i.e., asymmetric device counts) are not supported.

<!-- npu="310p" id10 -->

- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of 2 Atlas 300I Duo inference cards (i.e., 4 NPUs) per server.

<!-- end id10 -->