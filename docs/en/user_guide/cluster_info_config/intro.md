# Introduction

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-07-30T03:50:14.074Z pushedAt=2026-07-31T07:30:53.208Z -->

Developers can configure NPU resource information for collective communication through a rank table file. The following scenarios involve cluster information configuration via the rank table file:

- When initializing a communication domain through the C API [HcclCommInitClusterInfo](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCommInitClusterInfo.md) or [HcclCommInitClusterInfoConfig](https://gitcode.com/cann/hcomm/blob/9.1.0/docs/en/api_ref/comm_mgr_c/HcclCommInitClusterInfoConfig.md).

- When initializing the TensorFlow distributed network communication domain.

For TensorFlow framework networks, resource information can also be configured through environment variables. However, developers must choose either the rank table method or the environment variable method; mixing the two is not supported. The environment variable-based resource configuration method is supported only on the following products:

<!-- npu="910b" id1 -->

- Atlas A2 training products/Atlas A2 inference products

<!-- end id1 -->
<!-- npu="910" id2 -->

- Atlas training products

<!-- end id2 -->
