# HcclSendRecvType<a name="ZH-CN_TOPIC_0000002518992207"></a>

## 功能说明<a name="zh-cn_topic_0000001767474342_section162709502369"></a>

用于批量点对点通信操作，用来标识当前的任务类型是发送还是接收。

## 定义原型<a name="zh-cn_topic_0000001767474342_section742411329366"></a>

```c
typedef enum {
    HCCL_SEND = 0,    /* 当前任务是发送任务 */
    HCCL_RECV = 1,    /* 当前任务是接收任务 */
    HCCL_SEND_RECV_RESERVED     /* 保留字段 */
} HcclSendRecvType;
```

