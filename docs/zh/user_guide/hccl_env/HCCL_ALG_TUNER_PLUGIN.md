# HCCL_ALG_TUNER_PLUGIN

## 功能描述

此环境变量用于指定HCCL Tuner调优插件的动态库。插件可根据通信域、算子类型、数据类型、归约类型和数据量调整候选算法的代价值，供算法选择器选择算法。

- 取值为插件动态库的路径，建议使用绝对路径。
- 未设置、值为空或设置为`none`时，不加载插件，使用原有算法选择逻辑。
- 动态库需导出入口符号`HcclTunerPluginOpsGet()`，返回的`HcclTunerPluginOps`函数表须通过`magicWord`/`version`/`size`校验，且`Create`与`UpdateAlgoCostTable`函数指针不能为空。
- HCCL在进程内首次尝试加载插件时读取此变量。请在启动业务进程前配置；首次加载失败后，该进程不再重试加载。
- 动态库无法加载或必要符号校验失败时，HCCL打印警告并使用原有算法选择逻辑。

插件开发和使用方法（含插件配置文件的读取与JSON配置格式要求）请参见《[Tuner插件示例](../../../../examples/07_tuner_plugin/README.md)》。

> [!NOTE]说明
> 孵化中（Incubating）特性：实验性能力，默认未开放；转稳定前不保证接口兼容。

## 配置示例

```bash
export HCCL_ALG_TUNER_PLUGIN=/path/to/hccl_tuner_example.so
```

禁用插件时，在启动业务进程前执行：

```bash
export HCCL_ALG_TUNER_PLUGIN=none
```

## 使用约束

插件在业务进程内运行，建议仅加载来源可信且与当前HCCL插件接口兼容的动态库。

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id5 -->
- Atlas训练系列产品：不支持
<!-- end id5 -->
<!-- npu="310p" id4 -->
- Atlas推理系列产品：不支持
<!-- end id4 -->
