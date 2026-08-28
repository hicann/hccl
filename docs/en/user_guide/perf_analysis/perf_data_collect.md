# Profile Data Collection

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-08-10T09:33:33.971Z pushedAt=2026-08-10T09:55:14.916Z -->

Collective communication is a global collaborative behavior within a communicator. It is often difficult to analyze the performance issues of collective communication using the profile data of only one rank. Therefore, you need to collect the profile data of all ranks to accurately identify the performance bottleneck of collective communication. Currently, profile data can be collected in the following two ways:

- Method 1: Refer to *[Performance Tuning Tool User Guide](https://hiascend.com/en/document/redirect/CannCommunityToolProfiling)* to collect profile data.

- Method 2: Refer to *[HCCL Performance Test Tool User Guide](https://hiascend.com/en/document/redirect/CannCommunityToolHcclTest)* and use HCCL Test to collect profile data and perform performance tests.

  Follow the steps below to run HCCL Test for collecting profile data:

    ```bash
    # "1" enables profiling and "0" disables profiling. The default value is 0. When enabled, profile data is collected during HCCL Test execution.
    export HCCL_TEST_PROFILING=1
    # Specify the profile data storage path. The default path is /var/log/npu/profiling.
    export HCCL_TEST_PROFILING_PATH=/home/profiling
    ```

    If HCCL_TEST_PROFILING is enabled, profile data is generated in the directory specified by `HCCL_TEST_PROFILING_PATH` after the HCCL Test tool completes execution. For profile data parsing, see the "Using the msprof Command to Parse, Query, and Export the Profile Data" section in *[Performance Tuning Tool User Guide](https://hiascend.com/en/document/redirect/CannCommunityToolProfiling)*.
    