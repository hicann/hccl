# HCCL_OP_RETRY_PARAMS

## 功能描述

当开发者通过环境变量[HCCL_OP_RETRY_ENABLE](HCCL_OP_RETRY_ENABLE.md)开启了HCCL的算子重执行特性时，可通过本环境变量配置第一次重执行的等待时间、最大重执行的次数以及两次重执行的间隔时间。

配置方法如下：

**export HCCL_OP_RETRY_PARAMS="MaxCnt:3,HoldTime:5000,IntervalTime:1000"**

- MaxCnt：最大重传次数，uint32类型，取值范围为\[1,10\]，默认值为1，单位次。
- HoldTime：从检测到通信算子执行失败到开始第一次重新执行的等待时间，uint32类型，取值范围\[0,60000\]，默认值为5000，单位ms。
- IntervalTime：同一个通信算子两次重执行的间隔时间，uint32类型，取值范围\[0,60000\]，默认值为1000，单位ms。

## 配置示例

```bash
export HCCL_OP_RETRY_PARAMS="MaxCnt:5,HoldTime:5000,IntervalTime:5000"
```

## 使用约束

- 仅通过环境变量[HCCL_OP_RETRY_ENABLE](HCCL_OP_RETRY_ENABLE.md)开启了HCCL的重执行特性时（开启任一层级的重执行特性即可），此环境变量才生效。
- 若您调用HCCL C接口初始化具有特定配置的通信域时，通过“HcclCommConfig”的“hcclRetryParams”参数配置了第一次重执行的等待时间，则以通信域粒度的配置优先。

## 支持的型号

Atlas A3 训练系列产品/Atlas A3 推理系列产品
