# 简介

开发者可以通过rank table文件配置参与集合通信的NPU资源信息，通过rank table文件进行集群信息配置有以下使用场景：

- 通过C接口[HcclCommInitClusterInfo](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCommInitClusterInfo.md)或者[HcclCommInitClusterInfoConfig](https://gitcode.com/cann/hcomm/blob/master/docs/api_ref/comm_mgr_c/HcclCommInitClusterInfoConfig.md)初始化通信域时。
- TensorFlow分布式网络通信域初始化时。

针对TensorFlow框架网络，还可以通过环境变量的方式配置资源信息，但开发者仅可以选择rank table或环境变量方式中的一种，不支持两种方式混用，环境变量配置资源信息的方式仅支持如下产品：
<cann-filter npu-type="910b">
- Atlas A2 训练系列产品/Atlas A2 推理系列产品
</cann-filter>
<cann-filter npu-type="910">
- Atlas 训练系列产品
</cann-filter>
