# HcclBatchSendRecv

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-04T09:50:43.336Z pushedAt=2026-08-11T03:40:04.892Z -->

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

Asynchronous batch point-to-point communication API. A single API call can complete multiple send and receive tasks on the current rank. The send and receive operations on the current rank are asynchronous, meaning send and receive tasks do not block each other.

## Function Prototype

```c
HcclResult HcclBatchSendRecv(HcclSendRecvItem* sendRecvInfo, uint32_t itemNum, HcclComm comm, aclrtStream stream)
```

## Parameters

| Parameter | Input/Output | Description |
| --- | --- | --- |
| sendRecvInfo | Input | Pointer to the list of send/receive tasks to be issued on the current rank.<br>The data type is HcclSendRecvItem. For details, see [HcclSendRecvItem](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclSendRecvItem.md). |
| itemNum | Input | Number of send and receive tasks on the current rank. |
| comm | Input | Communicator for the collective communication operation. |
| stream | Input | Stream used by the current rank. |

### Supported Data Types

- For Ascend 950PR/Ascend 950DT, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float8-e5m2, float8-e4m3, float8-e8m0, hifloat8, float16, float32, float64, bfp16.

- For Atlas A3 training products/Atlas A3 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas A2 training products/Atlas A2 inference products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64, bfp16.

- For Atlas training products, supported data types: int8, uint8, int16, uint16, int32, uint32, int64, uint64, float16, float32, float64.

## Return Value

[HcclResult](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/data_type_definition/HcclResult.md): The API returns HCCL_SUCCESS on success, and other values on failure.

## Constraints

- "Asynchronous" means that the receive and send tasks on the same device are asynchronous and do not block each other. However, between devices, send and receive tasks are still synchronous. Therefore, inter-device send and receive tasks must be in one-to-one correspondence, just like HcclSend and HcclRecv.

- For Atlas A2 training products/Atlas A2 inference products, when using this API in large-scale clusters (ranksize > 500), the number of concurrent executions cannot exceed 3.

- For [Atlas 200T A2 Box16](https://support.huawei.com/enterprise/en/doc/EDOC1100318274/287e0458), if a link establishment failure occurs between devices within a server (error code: EI0010), set the environment variable `HCCL_INTRA_ROCE_ENABLE` to `1` and `HCCL_INTRA_PCIE_ENABLE` to `0`, so that the RoCE loop is used for multi-device communication within the server (ensure that a RoCE NIC exists on the server and that the RDMA links between devices with send/recv relationships are reachable). Example environment variable configuration:

    ```bash
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_INTRA_PCIE_ENABLE=0
    ```

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

// Execute Send/Recv to send data to the next node and receive data from the previous node.
// HcclBatchSendRecv can simultaneously issue multiple send/receive tasks on this rank.
uint32_t next = (deviceId + 1) % count;
uint32_t prev = (deviceId - 1 + count) % count;
HcclSendRecvItem sendRecvInfo[2];
sendRecvInfo[0] = HcclSendRecvItem{HCCL_SEND, sendBuf, count, HCCL_DATA_TYPE_FP32, next};
sendRecvInfo[1] = HcclSendRecvItem{HCCL_RECV, recvBuf, count, HCCL_DATA_TYPE_FP32, prev};
HcclBatchSendRecv(sendRecvInfo, 2, hcclComm, stream);

// Block and wait for the collective communication task in the stream to complete.
ACLCHECK(aclrtSynchronizeStream(stream));

// Release resources.
aclrtFree(sendBuf);          // Release memory on the device.
aclrtFree(recvBuf);          // Release memory on the device.
aclrtDestroyStream(stream);  // Destroy the stream.
HcclCommDestroy(hcclComm);   // Destroy the communicator.
```
