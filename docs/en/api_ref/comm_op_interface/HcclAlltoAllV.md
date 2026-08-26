# HcclAlltoAllV

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-04T09:50:05.510Z pushedAt=2026-08-05T02:40:26.132Z -->

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

The operation API for the collective communication operator AlltoAllV, which sends data (with customizable data volume) to all ranks in the communicator and receives data from all ranks.

![alltoallv](figures/alltoallv.png)

## Function Prototype

```c
HcclResult HcclAlltoAllV(const void *sendBuf, const void *sendCounts, const void *sdispls, HcclDataType sendType, const void *recvBuf, const void *recvCounts, const void *rdispls, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Name | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| sendCounts | Input | A uint64 array that indicates the send data volume. "sendCounts\[i] = n" indicates that this rank sends *n* data units to rank i.<br>For example, if "sendType" is float32, "sendCounts\[i] = n" indicates that this rank sends *n* float32 data units to rank i. |
| sdispls | Input | A uint64 array that indicates the send offset. "sdispls\[i] = n" indicates the offset of the data sent by this rank to rank i in sendBuf relative to the start of sendBuf, in units of sendType. |
| sendType | Input | Data type of the send data, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by model. For details, see [Supported Data Types](#supported-data-types). |
| recvBuf | Output | Address of the destination data buffer, where the collective communication result is output.<br>recvBuf and sendBuf cannot be configured with the same address. |
| recvCounts | Input | A uint64 array that indicates the receive data volume. "recvCounts\[i] = n" indicates that this rank receives *n* data units from rank i.<br>For example, if "recvType" is float32, "recvCounts\[i] = n" indicates that this rank receives *n* float32 data units from rank i. |
| rdispls | Input | A uint64 array that indicates the receive offset. "rdispls\[i] = n" indicates the offset of the data received by this rank from rank i in recvBuf relative to the start of recvBuf, in units of recvType. |
| recvType | Input | Data type of the receive data, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Supported data types vary by model. For details, see [Supported Data Types](#supported-data-types). |
| comm | Input | Communicator where the collective communication operation resides. |
| stream | Input | Stream used by this rank. |

### Supported Data Types

- For Ascend 950PR/Ascend 950DT, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.

- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas training products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.

- For Atlas 300I Duo, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- The performance of the AlltoAllV operation depends on the buffer size for data sharing between NPUs. When the communication data volume exceeds the buffer size, performance degrades significantly. If the AlltoAllV communication data volume is large in your service, configure the environment variable [HCCL_BUFFSIZE](../../user_guide/hccl_env/HCCL_BUFFSIZE.md) to increase the buffer size appropriately to improve communication performance.

<!-- npu="910" id13 -->

- For Atlas training products, the communicator of AlltoAllV must meet the following constraints:

  In cluster networking, single-server 1p and 2p communicators must be within the same cluster (devices 0-3 and devices 4-7 within a server each form a cluster). For single-server 4p, single-server 8p, and multi-server communicators, ranks must be organized by cluster as the basic unit, and the cluster selection must be consistent across servers.

- For Atlas training products, in single-server use cases, the NIC status must be "up"; otherwise, this API will fail to execute.

<!-- end id13 -->
<!-- npu="310p" id14 -->

- For Atlas 300I Duo, only single-server use cases are supported, with a maximum of 2 Atlas 300I Duo inference cards (i.e., 4 NPUs) per server.

<!-- end id14 -->

## Example

```c
// Allocate device memory for the collective communication operation.
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

// Set the send and receive data volumes. The send and receive data volumes are the same.
std::vector<uint64_t> sendCounts(rankSize, 1);
std::vector<uint64_t> recvCounts(rankSize, 1);
std::vector<uint64_t> sdispls(rankSize);
std::vector<uint64_t> rdispls(rankSize);
for (size_t i = 0; i < rankSize; ++i) {
    sdispls[i] = i;
    rdispls[i] = i;
}
// Execute AlltoAllV to send data of the same volume to all ranks in the communicator and receive data of the same volume from all ranks. The data volume is customizable.
HcclAlltoAllV(sendBuf, sendCounts.data(), sdispls.data(), HCCL_DATA_TYPE_FP32,
              recvBuf, recvCounts.data(), rdispls.data(), HCCL_DATA_TYPE_FP32, hcclComm, stream);
// Block and wait for the collective communication task in the task stream to complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Release the memory on the device.
aclrtFree(recvBuf);          // Release the memory on the device.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
