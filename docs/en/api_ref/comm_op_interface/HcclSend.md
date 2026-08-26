# HcclSend

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-04T09:51:13.262Z pushedAt=2026-08-05T02:40:26.155Z -->

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

The operation API of the point-to-point communication operator Send, which sends data from a specified location on the current node to the destination node.

## Function Prototype

```c
HcclResult HcclSend(void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendBuf | Input | Address of the source data buffer. |
| count | Input | Number of data units to send. |
| dataType | Input | Data type of the data to send, of [HcclDataType](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md) type.<br>Different models support different data types. For details, see [Supported Data Types](#supported-data-types). |
| destRank | Input | Rank ID of the data receiver in the communicator. |
| comm | Input | Communicator where the collective communication operation resides. |
| stream | Input | Stream used by the current rank. |

### Supported Data Types

- For Ascend 950PR/Ascend 950DT, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.

- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas training products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

HcclSend and HcclRecv APIs use synchronous calls and work in pairs. That is, after a process calls the HcclSend API, it must wait until the paired HcclRecv API receives the data before making the next API call, as shown in the following figure.

![send_recv](figures/send_recv.png)

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
// HcclSend and HcclRecv APIs use synchronous calls and work in pairs.
if (deviceId % 2 == 0) {
    // Apply for device memory to store input data.
    aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    // Initialize the input data.
    aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE);
    // Perform the Send operation.
    HcclSend(sendBuf, count, HCCL_DATA_TYPE_FP32, deviceId + 1, hcclComm, stream);
} else {
    // Allocate device memory for receiving data.
    aclrtMalloc(&recvBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);
    // Perform the Recv operation.
    HcclRecv(recvBuf, count, HCCL_DATA_TYPE_FP32, deviceId - 1, hcclComm, stream);
}

// Block and wait until the collective communication tasks in the stream are complete.
aclrtSynchronizeStream(stream);

// Release resources.
aclrtFree(sendBuf);          // Release memory on the device.
aclrtFree(recvBuf);          // Release memory on the device.
aclrtDestroyStream(stream);  // Destroy the task stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
