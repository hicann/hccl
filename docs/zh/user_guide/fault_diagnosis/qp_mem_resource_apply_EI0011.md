# QP内存资源申请相关（EI0011）

在参数面建链阶段HCCL会创建QP，如果device侧内存不足会上报OOM错误。请通过调整业务配置、减少ROCE链路的使用数量，或释放部分内存解决问题。

## 问题现象

在打屏日志中存在关键字"EI0011"或"Resource_Error_Insufficient_Device_Memory"，如下所示：

```text
[PID: 2103452] 2025-11-03-20:18:46.447.213 Resource_Error_Insufficient_Device_Memory(EI0011): Failed to allocate [size: [0.25MB, 3MB], Affected by QP depth configuration.] bytes of NPU memory.
        Possible Cause: Allocation failure due to insufficient NPU memory.
        Solution: Stop unnecessary processes and ensure the required memory is available.
```

## 解决方法

调整业务配置（如batchSize）、减少ROCE链路的使用数量，或释放部分内存解决问题。

**注意：**

HCCL的其他内存申请如cclBuffer内存申请若出现OOM错误，会由drv组件上报错误码并打印错误信息，可根据报错信息或CANN日志中的堆栈判断是否为HCCL内存申请失败，若为HCCL内存申请失败，可通过配置HCCL_BUFFSIZE环境变量调整申请的内存大小。
