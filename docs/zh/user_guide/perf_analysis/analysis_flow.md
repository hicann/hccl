# 分析流程

集群性能会受到AI处理器类型、网络、通信算法、通信配置等多方面因素影响， 对于性能问题，会通过Profiling工具进行性能分析，主要分析流程如下：

1. 采集全量的Profiling数据，详见[性能数据采集](perf_data_collect.md)。
2. 判断整网性能的瓶颈点，根据通信算子下发、执行的不同阶段做进一步分析和优化，详见[性能数据分析](perf_data_analysis.md)。

本节内容主要关注HCCL相关的Profiling信息识别及常见案例的分析思路，更多的性能调优案例请参考《[性能问题通用定位指南](https://hiascend.com/document/redirect/mindstudioGeneralPerformanceIssue)》中的“TopN性能问题的解决方案 > 通信问题优化方案”章节，采集到全量的Profiling数据后，参考《[MindStudio Insight工具用户指南](https://hiascend.com/document/redirect/MindStudioInsight)》对Profiling数据进行分析。
