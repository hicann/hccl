# AllReduce AIV写语义分核测试

本目录通过CPU桩运行生产TwoShot写kernel，记录搬运、归约和flag依赖。区间内存模型检查输出完整覆盖、普通写入不重叠、读取前数据就绪，以及每个元素的原树形归约表达式。依赖调度器检查所有等待可满足，且每个flag只有一个发布者。

测试覆盖producer发送字节门槛、不同元素宽度、空片和余数片、每原分片唯一归约核、整片完成通知直接发布、单/双producer切换、非2次幂rank、少核回退及额外空闲核。每个rank由`groupSize`个核并行归约，每核负责一个完整原分片；其他consumer参与AG回拷。测试断言没有中间完成flag，每片只由指定归约核发布一次localReduce通知，非空片只等待各源rank的数据就绪。producer仍沿用95KiB发送量与4KiB归约预算门槛，以便独立对比归约策略的性能。

配置仓库UT环境后执行：

```bash
bash build.sh --ut
```

对应CMake目标为`allreduce_aiv_core_test`。该测试验证编排和元素归约顺序；实际浮点运算、NPU流水同步及性能由上板ST验证。
