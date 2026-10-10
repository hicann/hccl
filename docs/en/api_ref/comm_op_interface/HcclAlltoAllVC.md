# HcclAlltoAllVC

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T07:57:26.605Z pushedAt=2026-09-18T08:03:58.119Z -->

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
- Atlas training products: Not supported
<!-- end id5 -->

## Description

The operation API of the collective communication operator AlltoAllVC, which sends data (with customizable data volume) to all ranks in the communicator and receives data from all ranks. Compared with AlltoAllV, AlltoAllVC passes the send/receive parameters of all ranks through the input parameter `sendCountMatrix`.

![alltoallvc](figures/alltoallvc.png)

## Prototype

```c
HcclResult HcclAlltoAllVC(const void *sendBuf, const void *sendCountMatrix, HcclDataType sendType, const void *recvBuf, HcclDataType recvType, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| sendCountMatrix | Input | A two-dimensional uint64 array representing the sent data volume. The array shape is `[rankSize][rankSize]`, and `sendCountMatrix[i][j] = n` indicates that rank i sends n data units to rank j.<br>For example, if `sendType` is float32, `sendCountMatrix[i][j] = n` indicates that rank i sends n float32 data units to rank j. |
| sendType | Input | Data type of the send data, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types).|
| recvBuf | Output | Address of the destination buffer. Collective communication results are output to this buffer.<br>The addresses configured for recvBuf and sendBuf cannot be the same, and their memory scope cannot overlap. |
| recvType | Input | Data type of the received data, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types). |
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

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | The passed pointer parameter is null, for example, comm, sendBuf, recvBuf, stream, or sendCountMatrix is nullptr. |
| HCCL_E_PARA | The passed parameter is invalid, for example, the addresses of sendBuf and recvBuf are the same. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, for example, dataType is invalid or the current model does not support it. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

The performance of the AlltoAllVC operation depends on the buffer size for sharing data between NPUs. When the communication data volume exceeds the buffer size, performance will degrade significantly. If the AlltoAllVC communication data volume is large in your service, you are advised to configure the environment variable [HCCL_BUFFSIZE](../../user_guide/hccl_env/HCCL_BUFFSIZE.md) to appropriately increase the buffer size to improve communication performance.

- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.

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

// Set the data amount to send and receive. The send and receive amounts are the same.
std::vector<uint64_t> sendCountMatrix(rankSize * rankSize);
for (uint32_t i = 0; i < rankSize; ++i) {
    for (uint32_t j = 0; j < rankSize; ++j) {
        sendCountMatrix[i * rankSize + j] = count / rankSize;
    }
}

// Execute AlltoAllVC to send data of the same amount to all ranks in the communicator and receive data of the same amount from all ranks. The data amount can be customized.
HcclAlltoAllVC(sendBuf, sendCountMatrix.data(), HCCL_DATA_TYPE_FP32, recvBuf, HCCL_DATA_TYPE_FP32, hcclComm, stream);

// Block until the collective communication tasks in the task stream complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free memory on the device.
aclrtFree(recvBuf);          // Free memory on the device.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
