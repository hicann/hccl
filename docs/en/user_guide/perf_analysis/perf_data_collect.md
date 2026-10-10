# Profile Data Collection

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-12T11:02:51.853Z pushedAt=2026-09-18T08:03:58.286Z -->

Collective communication is a global collaborative behavior within a communicator. It is often difficult to analyze the performance issues of collective communication using the profile data of only one rank. Therefore, you need to collect the profile data of all ranks to accurately identify the performance bottleneck of collective communication. Currently, profile data can be collected in the following two ways:

- Method 1: Refer to *[Performance Tuning Tool User Guide](https://hiascend.com/en/document/redirect/CannCommunityToolProfiling)* to collect profile data.
- Method 2: Refer to the *[HCCL Performance Tester User Guide](https://gitcode.com/cann/oam-tools/blob/master/docs/zh/hccl_test/README.md)* and use HCCL Test to collect profile data and test performance.

  Follow the steps below to run HCCL Test for collecting profile data:

    ```bash
    # "1" indicates that profiling is enabled, "0" indicates that profiling is disabled, and the default value is "0". If profiling is enabled, profile data is collected when HCCL Test is executed.
    export HCCL_TEST_PROFILING=1
    # Specify the path for storing profile data. Default: /var/log/npu/profiling
    export HCCL_TEST_PROFILING_PATH=/home/profiling
    ```

    If HCCL_TEST_PROFILING is enabled, profile data is generated in the directory specified by `HCCL_TEST_PROFILING_PATH` after the HCCL Test tool completes execution. For profile data parsing, see the "Using the msprof Command to Parse, Query, and Export the Profile Data" section in *[Performance Tuning Tool User Guide](https://hiascend.com/en/document/redirect/CannCommunityToolProfiling)*.
