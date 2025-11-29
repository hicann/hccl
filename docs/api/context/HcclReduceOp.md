# HcclReduceOp<a name="ZH-CN_TOPIC_0000002486832346"></a>

## 功能说明<a name="zh-cn_topic_0000001755666480_section91594268342"></a>

定义集合通信reduce操作的类型。

## 定义原型<a name="zh-cn_topic_0000001755666480_section141901296351"></a>

```c
typedef enum {
    HCCL_REDUCE_SUM = 0,    /* sum */
    HCCL_REDUCE_PROD = 1,   /* prod */
    HCCL_REDUCE_MAX = 2,    /* max */
    HCCL_REDUCE_MIN = 3,    /* min */
    HCCL_REDUCE_RESERVED    /* reserved */
} HcclReduceOp;
```

