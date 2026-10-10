# HcclSend

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T08:28:27.023Z pushedAt=2026-09-18T08:03:58.141Z -->

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

The operation API of the point-to-point communication operator Send, which sends data from a specified location on the current node to the destination node.

## Prototype

```c
HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| count | Input | Number of data units to send. |
| dataType | Input | Data type of the sent data, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types).|
| destRank | Input | Rank ID of the destination that receives the data in the communicator. |
| comm | Input | Communicator where the collective communication operation is performed. |
| stream | Input | Stream used by this rank. |

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
<!-- npu="910" id13 -->
- For Atlas training products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.
<!-- end id13 -->

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| Return Value | Description |
| --- | --- |
| HCCL_SUCCESS | API call success. |
| HCCL_E_PTR | The passed pointer parameter is null, e.g., comm or sendBuf is nullptr. |
| HCCL_E_PARA | The passed parameter is invalid, e.g., count exceeds the upper limit or destRank is out of range. |
| HCCL_E_NOT_SUPPORT | The operation is not supported, e.g., the dataType is invalid or not supported by the current model, or self-send and self-receive are not supported when destRank equals the current rank. |
| HCCL_E_INTERNAL | Internal error. |

## Constraints

The HcclSend and HcclRecv APIs use a synchronous call approach. They work in pairs. That is, after a process calls the HcclSend API, it must wait for the paired HcclRecv API to receive the data before making the next API call, as shown in the following figure.

![send_recv](figures/send_recv.png)

- Must serialize the calls to all communication operators in multiple communicators on each device. Out-of-order/multi-threaded concurrent calls and thread reentrancy are not allowed.
- On the same device, the threads that dispatch all collective communication operators in the same communicator must use the same context.

## Example

```c
void *sendBuf = nullptr;
void *recvBuf = nullptr;
uint64_t count = 8;
size_t mallocSize = count * sizeof(float);

// Initialize the communicator.
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// Create a task stream.
aclrtStream stream;
aclrtCreateStream(&stream);

// Perform Send/Recv operations. Devices 0/2/4/6 send data, and devices 1/3/5/7 receive data.
// Call HcclSend and HcclRecv synchronously and use them in pairs.
if (deviceId % 2 == 0) {
    // Allocate device memory to store the input data.
    aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    // Initialize the input data.
    aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE);
    // Perform the Send operation.
    HcclSend(sendBuf, count, HCCL_DATA_TYPE_FP32, deviceId + 1, hcclComm, stream);
} else {
    // Allocate device memory to receive data.
    aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    // Perform the Recv operation.
    HcclRecv(recvBuf, count, HCCL_DATA_TYPE_FP32, deviceId - 1, hcclComm, stream);
}

// Block until the collective communication tasks in the stream complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Free device-side memory.
aclrtFree(recvBuf);          // Free device-side memory.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
