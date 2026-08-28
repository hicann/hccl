# Analysis Process

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-10T09:33:37.131Z pushedAt=2026-08-10T09:47:11.892Z -->

Cluster performance is steered by multiple factors such as AI processor type, network, communication algorithm, and communication configuration. For performance issues, use profiling for analysis as follows:

1. Collect full profile data. For details, see [Profile Data Collection](perf_data_collect.md).

2. Identify the bottleneck of the overall cluster performance, and perform further analysis and optimization based on the different stages of communication operator dispatch and execution. For details, see [Profile Data Analysis](perf_data_analysis.md).

This section focuses on HCCL-related profile data identification and analysis approaches for common cases. For more performance tuning cases, see "Solutions for TopN Performance Issues > Communication Tuning Solution" in *[General Performance Issue Troubleshooting Guide](https://hiascend.com/en/document/redirect/mindstudioGeneralPerformanceIssue)*. After collecting full profile data, refer to *[MindStudio Insight Tool User Guide](https://hiascend.com/en/document/redirect/MindStudioInsight)* to analyze the profile data.
