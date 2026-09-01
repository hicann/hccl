# SDMA ERROR（EI0012）

## 问题现象

在打屏日志中会有EI0012的错误码打印，关键字为"Execution_Error_SDMA"，如下所示：

```text
[PID: 3480365] 2025-12-24-14:10:31.094.189 Execution_Error_SDMA(EI0012): SDMA memory copy task exception occurred. Remote rank: [4800]. Base information: [streamID:[351], taskID[5], taskType[Memcpy], tag[], AlgType(level 0-1-2):[null-null-null].]. Task information: [src:[0x12c180000000], dst:[0x12c041800000], size:[0x80], notify id:[0xffffffffffffffff], link type:[HCCS], remote rank:[0]]. Communicator information: [group:[], user define information[], rankSize[0], rankId[0]].
```

且在CANN日志中存在关键字"**fftsplus sdma error**"，如下所示：

```text
[ERROR] RUNTIME(57096,python3.10):2025-05-12-20:55:44.705.025 [task_info.cc:1170]57288 PrintSdmaErrorInfoForFftsPlusTask:fftsplus task execute failed, dev_id=0, stream_id=50, task_id=21, context_id=18, thread_id=0, err_type=4[fftsplus sdma error]
[ERROR] RUNTIME(57096,python3.10):2025-05-12-20:55:44.705.031 [task_info.cc:1270]57288 TaskFailCallBackForFftsPlusTask:fftsplus streamId=50, taskId=21, context_id=18, expandtype=1, rtCode=0x715006c,[fftsplus task exception], psStart=0x0, kernel_name=not found kernel name, binHandle=(nil), binSize=0.
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.132 [task_exception_handler.cc:947] [57288][TaskExecStage][Timeout][Host]Task run failed, base information is streamID:[32], taskID[21], tag[AllGather_group_name_0], AlgType(level 0-1-2):[fullmesh-ring-H-D].
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.140 [task_exception_handler.cc:810] [57288][TaskExecStage][Timeout][Host]Task run failed, groupRank information is group:[group_name_0], user define information[Unspecified], rankSize[8], rankId[0].
[ERROR] HCCL(57096,python3.10):2025-05-12-20:55:44.706.163 [task_exception_handler.cc:737] [57288][TaskExecStage][Timeout][Host]Task run failed, opData information is timeStamp:[2025-05-12-20:54:51.268.778], deviceId[0], index[4], count[3397632], src[0x12c25487ac00], dst[0x12c255000000], dataType[uint8].
```

## 可能原因

在执行SDMA内存拷贝任务时发生了页表转换失败，也就是内存拷贝的输入或者输出地址未分配内存、分配的内存小于内存拷贝大小或者分配的内存已被释放。

常见的问题根因有以下场景：

- 下发通信算子后，未执行流同步确认通信算子执行完成就析构通信域，由于通信域析构时会释放集合通信的HCCL Buffer地址，因此会导致SDMA内存拷贝时页表转换失败。

    可以在CANN日志的run目录下检索通信域销毁的时间：

    ```bash
    grep -r "Entry-HcclCommDestroy" log/run/plog
    ```

  <!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品下，网络链路故障也会导致SDMA ERROR，此时需要检查两端之间的链路状态。
  <!-- end id2 -->
- 调用HCCL通信算子时，传入的输入或输出地址实际分配的内存大小小于传入的数据量Count。
