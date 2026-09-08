# HcclBroadcast

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- Atlas 推理系列产品：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- Atlas 训练系列产品：支持
<!-- end id5 -->

## 功能说明

集合通信算子Broadcast的操作接口，将通信域内root节点的数据广播到其他rank。

![broadcast](figures/broadcast.png)

## 函数原型

```c
HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| buf | 输入/输出 | 数据buffer地址，对于root节点，是源数据buffer地址；对于非root节点，是数据接收buffer地址。 |
| count | 输入 | 参与broadcast操作的数据个数，比如只有一个int32数据参与，则count=1。 |
| dataType | 输入 | Broadcast操作的数据类型，[HcclDataType](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclDataType.md)类型。<br>不同的型号支持的数据类型不同，详细请参见[数据类型说明](#数据类型说明)。|
| root | 输入 | 作为broadcast root的rank id。 |
| comm | 输入 | 集合通信操作所在的通信域。 |
| stream | 输入 | 本rank所使用的stream。 |

### 数据类型说明

<!-- npu="950" id10 -->
- 针对Ascend 950PR/Ascend 950DT，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float8-e5m2、float8-e4m3、float8-e8m0、hifloat8、float16、float32、float64、bfp16。
<!-- end id10 -->
<!-- npu="A3" id11 -->
- 针对Atlas A3 训练系列产品/Atlas A3 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。
<!-- end id11 -->
<!-- npu="910b" id12 -->
- 针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64、bfp16。
<!-- end id12 -->
<!-- npu="910" id13 -->
- 针对Atlas 训练系列产品，支持数据类型：int8、uint8、int16、uint16、int32、uint32、int64、uint64、float16、float32、float64。
<!-- end id13 -->

## 返回值

[HcclResult](https://gitcode.com/cann/hcomm/blob/master/docs/zh/api_ref/comm_mgr_c/data_type_definition/HcclResult.md)

| 返回值 | 说明 |
| --- | --- |
| HCCL_SUCCESS | 接口调用成功。 |
| HCCL_E_PTR | 传入的指针参数为空，如comm、buf、stream等为nullptr。 |
| HCCL_E_PARA | 传入的参数无效，如count超过上限、root越界等。 |
| HCCL_E_NOT_SUPPORT | 操作不被支持，如dataType非法或当前型号不支持。 |
| HCCL_E_INTERNAL | 内部错误。 |

## 约束说明

- 所有rank的count、dataType、root均应相同。
- 全局只能有1个root节点。
- 多个通信域下的所有通信算子在每个Device上需要保证串行下发，不允许乱序、多线程并发下发，也不支持线程重入。
- 在同一Device上，同一通信域内的所有通信算子的下发线程需要使用相同的Context。

## 调用示例

```c
// 申请集合通信操作的Device 内存
void *buf = nullptr;    // 对于root节点，是数据源；对于非root节点，是数据接收buffer
uint64_t count = 8;     // 参与broadcast操作的数据个数
size_t mallocSize = count * sizeof(float);
aclrtMalloc(&buf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY);

// 在root 节点构造输入数据
if (deviceId == rootRank) {    
    aclrtMemcpy(buf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE);
}

// 初始化通信域
uint32_t rankSize = 8;
HcclComm hcclComm;
HcclCommInitRootInfo(rankSize, &rootInfo, deviceId, &hcclComm);

// 创建任务流
aclrtStream stream;
aclrtCreateStream(&stream);

// 执行广播操作，将通信域内root 节点的数据广播至其他rank
HcclBroadcast(buf, count, HCCL_DATA_TYPE_FP32, rootRank, hcclComm, stream);
// 阻塞等待任务流中的集合通信任务执行完成
aclrtSynchronizeStream(stream);

// 释放资源
aclrtFree(buf);              // 释放Device 侧内存
aclrtDestroyStream(stream);  // 销毁任务流
HcclCommDestroy(hcclComm);   // 销毁通信域
```
