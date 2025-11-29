# HcclResult<a name="ZH-CN_TOPIC_0000002519072193"></a>

## 功能说明<a name="zh-cn_topic_0000001802537733_section91594268342"></a>

定义集合通信相关操作的返回值。

## 定义原型<a name="zh-cn_topic_0000001802537733_section19551536113419"></a>

```c
typedef enum {
    HCCL_SUCCESS = 0,               /* success */
    HCCL_E_PARA = 1,                /* parameter error */
    HCCL_E_PTR = 2,                 /* empty pointer */
    HCCL_E_MEMORY = 3,              /* memory error */
    HCCL_E_INTERNAL = 4,            /* internal error */
    HCCL_E_NOT_SUPPORT = 5,         /* not support feature */
    HCCL_E_NOT_FOUND = 6,           /* not found specific resource */
    HCCL_E_UNAVAIL = 7,             /* resource unavailable */
    HCCL_E_SYSCALL = 8,             /* call system interface error */
    HCCL_E_TIMEOUT = 9,             /* timeout */
    HCCL_E_OPEN_FILE_FAILURE = 10,  /* open file fail */
    HCCL_E_TCP_CONNECT = 11,        /* tcp connect fail */
    HCCL_E_ROCE_CONNECT = 12,       /* roce connect fail */
    HCCL_E_TCP_TRANSFER = 13,       /* tcp transfer fail */
    HCCL_E_ROCE_TRANSFER = 14,      /* roce transfer fail */
    HCCL_E_RUNTIME = 15,            /* call runtime api fail */
    HCCL_E_DRV = 16,                /* call driver api fail */
    HCCL_E_PROFILING = 17,          /* call profiling api fail */
    HCCL_E_CCE = 18,                /* call cce api fail */
    HCCL_E_NETWORK = 19,            /* call network api fail */
    HCCL_E_AGAIN = 20,              /* try again */
    HCCL_E_REMOTE = 21,             /* error cqe */
    HCCL_E_SUSPENDING = 22,         /* error communicator suspending */
    HCCL_E_RESERVED                 /* reserved */
} HcclResult;
```

