# 性能数据采集

集合通信是一个通信域内的全局协同行为，只通过一个rank的Profiling数据往往难以分析集合通信的性能问题，因此需要采集到全量rank的Profiling数据才能精确地分析到集合通信的性能瓶颈所在。当前支持通过两种方式采集性能数据：

- 方式一：参考《[性能调优工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)》采集Profiling性能数据。
- 方式二：参考《[HCCL性能测试工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolHcclTest)》，使用HCCL Test进行Profiling数据的采集和性能测试。

  参考以下步骤，执行HCCL Test采集性能数据：

    ```bash
    # “1”代表开启Profiling，“0”代表关闭Profiling，默认值为“0”，开启时，执行HCCL Test时采集性能数据
    export HCCL_TEST_PROFILING=1
    # 指定Profiling数据存放路径，默认为/var/log/npu/profiling
    export HCCL_TEST_PROFILING_PATH=/home/profiling
    ```

    若开启HCCL\_TEST\_PROFILING，HCCL Test工具执行完成后会在HCCL\_TEST\_PROFILING\_PATH指定目录下生成Profiling数据，性能数据的解析请参考《[性能调优工具用户指南](https://hiascend.com/document/redirect/CannCommunityToolProfiling)》的“使用msprof命令解析、查询与导出性能数据”章节。
