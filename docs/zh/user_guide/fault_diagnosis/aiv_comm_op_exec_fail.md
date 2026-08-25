# AIV通信算子执行失败

## 问题现象

针对Atlas A2 训练系列产品/Atlas A2 推理系列产品，在通过export HCCL_OP_EXPANSION_MODE="AIV"启用AIV模式之后，部分场景下HCCL会以kernel的执行方式实现HCCL通信算子的编排及执行，此时若通信算子执行异常，在日志中会有一行如下关键日志打印"fault kernel_name=aiv_all_reduce_***"表明当前为HCCL的aiv算子执行失败：

```text
[ERROR] RUNTIME(699131,python):2025-04-24-21:54:17.707.236 [davinci_kernel_task.cc:1268]699131 PrintErrorInfoForDavinciTask:[INIT][DEFAULT]Aicore kernel execute failed, device_id=0, stream_id=2, report_stream_id=2, task_id=55873, flip_num=2073, fault kernel_name=aiv_all_reduce_***, fault kernel info ext=aiv_all_reduce_910b_bfloat16_t, program id=42, hash=9645272693770703471.
```

此外也会有上述同样的task exception信息打印，仍可以通过[notify wait超时（EI0002）](./notify_wait_timeout_EI0002.md#notify-wait超时ei0002)排查思路分析任务失败的根因，如是否全量超时、是否集群中存在某个先发生异常的节点等。
