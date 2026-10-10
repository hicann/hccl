# Analysis Process

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T10:59:58.741Z pushedAt=2026-09-18T08:03:58.280Z -->

Cluster performance is steered by multiple factors such as AI processor type, network, communication algorithm, and communication configuration. For performance issues, use profiling for analysis as follows:

1. Collect full profile data. For details, see [Profile Data Collection](perf_data_collect.md).
2. Identify the bottleneck of the overall cluster performance, and perform further analysis and optimization based on the different stages of communication operator dispatch and execution. For details, see [Profile Data Analysis](perf_data_analysis.md).

This section focuses on HCCL-related profile data identification and analysis approaches for common cases. For more performance tuning cases, see "Solutions for TopN Performance Issues > Communication Tuning Solution" in *[General Performance Issue Troubleshooting Guide](https://www.hiascend.com/document/detail/en/mindstudio/latest/practicalcases/GeneralPerformanceIssue/MindStudio/26.1.0/zh/cases/general_performance_issue_troubleshooting_guide/guide.md)*. After collecting full profile data, refer to *[MindStudio Insight Tool User Guide](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md)* to analyze the profile data.
