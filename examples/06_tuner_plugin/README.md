# HCCL Tuner Plugin 示例

## 概述

本目录提供 HCCL Tuner Plugin 的参考实现，演示如何通过外部 `.so` 插件读取 JSON 配置并修改cost table，影响 Selector 的算法选择。

## 编译

```bash
source /usr/local/Ascend/cann/set_env.sh
make
```

产物：`hccl_tuner_example.so`

首次构建会自动下载 nlohmann/json（header-only）到 `third_party/`。离线场景可手动预放 `third_party/nlohmann/json.hpp`，make 检测到即跳过下载。

## 使用

```bash
export HCCL_TUNER_PLUGIN=/path/to/hccl_tuner_example.so
export HCCL_TUNER_CONFIG_FILE=/path/to/hccl_tuner_config.json
```

插件按以下顺序查找配置文件，找到第一个可读的即加载：

1. `$HCCL_TUNER_CONFIG_FILE`（未设置时跳过）
2. `./hccl_tuner_config.json`
3. `/etc/hccl/hccl_tuner_config.json`

## JSON 配置格式

```json
{
  "version": 1,
  "op_types": {
    "allreduce": {
      "rules": [
        {
          "match": {
            "min_ranks": 8, "max_ranks": 8,
            "min_bytes": 0, "max_bytes": 65536,
            "data_type": "fp16"
          },
          "engine": "aicpu",
          "executor": "sole",
          "template": "nhr",
          "cost": 0.0
        }
      ]
    }
  }
}
```

### 顶层字段

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `version` | int | 是 | 配置格式版本，目前必须为 `1` |
| `op_types` | object | 是 | 按算子类型组织的规则集 |

### match 条件（全部 AND，first-match-wins）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `min_ranks` / `max_ranks` | uint32 | 是 | 通信域 rank 数范围 |
| `min_bytes` / `max_bytes` | size_t | 是 | 数据量范围（字节） |
| `data_type` | string | 否 | 数据类型（`int8`/`int16`/`int32`/`int64`/`uint8`/`uint16`/`uint32`/`uint64`/`fp16`/`float16`/`fp32`/`float32`/`fp64`/`float64`/`bfp16`/`bfloat16`） |
| `comm_name` | string | 否 | 通信域名（子串匹配） |
| `min_npus_per_server` / `max_npus_per_server` | uint32 | 否 | 每服务器 NPU 数范围 |
| `min_servers` / `max_servers` | uint32 | 否 | 服务器数范围 |
| `min_pods` / `max_pods` | uint32 | 否 | Pod 数范围 |
| `min_super_pods` / `max_super_pods` | uint32 | 否 | 超节点数范围 |
| `buffer_size` | uint64 | 否 | 通信域 buffer 大小（精确匹配） |

### 命中行为

- `engine` / `executor` / `template`（必填）指定 cost table 中的目标位置（可选值见下方）
- `cost`（必填）设置该位置的 cost 值，`0.0` 表示最优
- Selector 从 cost table 中选最小值确定算法

### engine 可选值

`aicpu` / `ccums` / `ccusched` / `aiv` / `dpu`

### executor 可选值

`sole` / `sequence` / `parallel` / `pipeline` / `concur`

### template 可选值

`mesh` / `mesh2die` / `meshoneshot` / `meshtwoshot` / `meshconcur` / `meshmultilink` / `meshchunk` / `meshchunktwoshot` / `nhr` / `nhrmultilink`

### 支持的 op_type

`allreduce` / `allgather` / `broadcast` / `reduce` / `reduce_scatter` / `scatter` / `alltoall` / `alltoallv`

> 以上 engine / executor / template / op_type 的可选值与 HCCL 算法选择器的维度定义一致。

## 测试

```bash
cd test
make
./test_plugin
```

## 插件接口

插件需导出符号 `hcclTunerPlugin_v1`（类型 `hcclTunerFuncs_v1_t`），包含 `init` 和 `getCollInfo` 两个函数指针。HCCL 核心通过 `dlsym` 获取该符号加载插件。

详见 `include/hccl_tuner_plugin.h`。
