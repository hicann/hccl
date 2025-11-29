# HcclDataType<a name="ZH-CN_TOPIC_0000002486992310"></a>

## 功能说明<a name="zh-cn_topic_0000001802547105_section91594268342"></a>

定义集合通信相关操作的数据类型。

## 定义原型<a name="zh-cn_topic_0000001802547105_section141901296351"></a>

```c
typedef enum {
    HCCL_DATA_TYPE_INT8 = 0,     /* int8 */
    HCCL_DATA_TYPE_INT16 = 1,    /* int16 */
    HCCL_DATA_TYPE_INT32 = 2,    /* int32 */
    HCCL_DATA_TYPE_FP16 = 3,     /* float16 */
    HCCL_DATA_TYPE_FP32 = 4,     /* float32 */
    HCCL_DATA_TYPE_INT64 = 5,    /* int64 */
    HCCL_DATA_TYPE_UINT64 = 6,   /* uint64 */
    HCCL_DATA_TYPE_UINT8 = 7,    /* uint8 */
    HCCL_DATA_TYPE_UINT16 = 8,   /* uint16 */
    HCCL_DATA_TYPE_UINT32 = 9,   /* uint32 */
    HCCL_DATA_TYPE_FP64 = 10,    /* fp64 */
    HCCL_DATA_TYPE_BFP16 = 11,   /* bfp16 */
    HCCL_DATA_TYPE_INT128 = 12,  /* int128 */
    HCCL_DATA_TYPE_RESERVED      /* reserved */
} HcclDataType;
```

