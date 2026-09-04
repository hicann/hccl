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

## 数据采集与配置生成

手工编写 JSON 配置难以猜中各拓扑下的最优算法。本目录提供两个脚本，把这件事变成三步：

```mermaid
flowchart LR
    A["prof_test.py<br/>逐算法拉起 mpirun + hccl_test"] -->|"性能 CSV"| B["optimize_config.py<br/>逐 size 挑带宽最优算法"]
    B -->|"tuner JSON"| C["插件加载<br/>改写 cost table 生效"]
```

### 第一步：采集性能数据（prof_test.py）

prof_test.py 在环境里逐个算法拉起 `mpirun + hccl_test`，解析输出表格，把各算法在每个数据量下的带宽和时延写入 CSV。

运行前请确认：

- Python >= 3.8
- `mpirun` 在 PATH 中（MPICH 或 Open MPI 均可，脚本自动适配两种命令格式）
- 已 source CANN 环境变量（`$ASCEND_HOME_PATH` 已设置），hccl_test 可执行文件位于
  `$ASCEND_HOME_PATH/tools/hccl_test/bin/`（也可通过 `--bin-dir` 指定其他位置）
- 脚本要在 mpirun 会话外运行（登录节点普通 shell）。如果检测到自己在 MPI 会话里，
  脚本会直接报错退出，避免嵌套拉起导致采集进程数被放大

典型命令：

```bash
python3 prof_test.py --op-types allreduce,allgather --engines aicpu --np-total 16 --npus 8 \
    --hostfile hostfile --minbytes 1K --maxbytes 2G --stepfactor 2 \
    --data-types fp32 --output hccl_prof.csv
```

#### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--op-types` | （必填） | 采集的算子，逗号分隔，支持 `all`（8 种白名单） |
| `--np-total` | （必填） | 总 rank 数，即 `mpirun -n` 的值 |
| `--npus` | 未指定 | 每台服务器的 NPU 数，透传给 hccl_test `-p` |
| `--hostfile` | 未指定 | 多机 hostfile 文件，单机不用传。每行写一条 `IP:NPU数`（如 `10.10.130.22:8`），`#` 开头的注释行和空行忽略；服务器数按有效行数统计 |
| `--minbytes` / `--maxbytes` | 未指定 | 数据量区间，对应 `-b` / `-e`；两者必须同时给。单位：纯数字按字节，`K`/`M`/`G` 后缀按 1024 进制（1K=1024B，1M=1024K），支持小数（如 `1.5K`） |
| `--stepbytes` | 未指定 | 线性步进（字节），正整数，对应 `-i`，与 `--stepfactor` 互斥；两者都不传时由 hccl_test 按默认序列扫点 |
| `--stepfactor` | 未指定 | 等比步进因子，对应 `-f`，浮点（典型值 2，即 1K→2K→4K…）；与 `--stepbytes` 互斥 |
| `--data-types` | `fp32` | 数据类型，逗号分隔（16 种），对应 `-d` |
| `--reduce-op` | `sum` | reduce 操作类型，仅对 reduce 类算子下发 `-o` |
| `--engines` | `all` | 采集引擎，支持 `aicpu` / `aiv` / `dpu`，`all` 展开全部（见下方说明） |
| `--algos` | 全量 | 显式指定算法串（`HCCL_ALGO` 语义，如 `sole{mesh};parallel{mesh,nhr}`） |
| `--executors` / `--templates` | 全量 | 按执行器 × 模板展开组合（`--algos` 存在时忽略） |
| `--warmup-iters` | 10 | 预热次数，对应 `-w` |
| `--iters` | 20 | 迭代次数，对应 `-n` |
| `--run-timeout` | 600 | 单轮超时秒数，超时杀掉整轮进程，按失败继续 |
| `--pods` | 0 | Pod 数；0 表示未指定，生成的配置不约束该维度 |
| `--super-pods` | 0 | 超节点数；同上 |
| `--output` | `hccl_prof.csv` | CSV 输出路径 |
| `--bin-dir` | `tools/hccl_test/bin` | hccl_test 目录，相对 `$ASCEND_HOME_PATH` 或绝对路径 |

#### 采集行为说明

**引擎范围**：当前支持 `aicpu` / `aiv` / `dpu`（`all` 展开全部三种）。`dpu` 不下发
`-a` 参数，由拓扑绑定决定。暂不支持 `ccums` / `ccusched`，传入会直接报错：当前
hccl 不支持 ccu_sched_only / ccu_ms_only 模式，配了 ccu 算法也可能回退到 aicpu
算法执行，测出的耗时不准，会污染选优结果。后续放开 ccums / ccusched。

**算法黑名单**：执行器带 concur 的算法、模板带 multilink / meshconcurrent 的算法
不采集（meshconcurrent 为 meshclos 方阵组网专属，executor 仍是 sole，故单独拉
黑）。这类算法在老选择逻辑下的选中条件很苛刻（特定机型、特定拓扑、特定数据量），
普通环境下就算配了，实际可能会选中其他算法，导致测出来的耗时不准，会污染选优
结果。

**拓扑层数剪枝**：脚本按 `--hostfile` 行数、`--pods`、`--super-pods` 估算拓扑
层数（单机=1 层，跨机/跨 Pod=2 层，跨超节点=3 层），只采集层数匹配的算法，
比如单机环境不会采 2 层专属算法——采了运行时也选不上。注意这只是估算下限：
跨超节点环境务必显式传 `--super-pods`，多 Pod 环境要传 `--pods`，否则对应层
的专属算法会被误剪掉。指定 `--algos` 时，层数不匹配的算法会被跳过并在结果里
标注。

**超时容错**：每轮默认 600 秒超时，到点杀掉整个进程树，该轮记为失败并继续后续
采集，单轮挂死不会卡住全程。失败或没解析出数据的轮次会在 CSV 里落一行
`check_result=failed` / `noresult` 的审计记录，原始输出追加到 `<output>.raw`
文件里供排查（缺 so、dlopen 失败等原因都能在里面找到）。

**增量采集**：CSV 是追加写的。文件已存在且表头一致就接着写，表头不一致会拒绝
追加，防止不同格式的数据混进一个文件。因此可以分多次跑，往同一个文件里补数据；
生成配置时按时间戳去重，同一个点只取最新一轮的结果。

### 第二步：生成最优配置（optimize_config.py）

```bash
python3 optimize_config.py --input hccl_prof.csv --output hccl_tuner_config.json
```

脚本读取 CSV，按算子、数据类型、reduce 类型、拓扑维度分组，在每个数据量点上
挑带宽最高的算法（带宽相同比时延），相邻点同算法的合并成区间，区间之间的缝隙
划给右边的区间，最后输出插件格式的 JSON。

两处口径换算需要注意：

- **allgather 系按每 rank 口径换算**：hccl_test 采集的 size 是每个 rank 收到的
  总量，而插件运行时匹配的是每个 rank 发出的量。生成规则前会自动除以 rank 数，
  CSV 里的原始数值不受影响
- **scatter 按总发出量换算**：hccl_test 采集的 size 是每个 rank 收到的量，而插件
  运行时匹配的是 root 总发出量。生成规则前会自动乘以 rank 数，CSV 里的原始数值
  不受影响
- **dpu 规则强制 `min_servers>=2`**：原因见下文「engine 可选值」一节

校验不通过的行（比如插件不支持的 executor）直接丢弃并在日志里给出计数和原因，
不会让一条坏数据导致整份配置失效。

### 第三步：加载生效

```bash
export HCCL_TUNER_PLUGIN=/path/to/hccl_tuner_example.so
export HCCL_TUNER_CONFIG_FILE=/path/to/hccl_tuner_config.json
```

之后正常运行业务即可，插件会按配置改写 cost table。配置格式见下一节。

### 目录下其他文件

`tuner_common.py` 是两个脚本共用的常量表（CSV 字段、算法注册表、黑名单等），
被两边直接引用，不需要单独运行。有两点维护上的事需要知道：

- **算法注册表是人工维护的**：合法算法组合表、拓扑层数约束表从 HCCL 源码的
  算法注册宏归纳而来，不会随 HCCL 版本自动更新。升级 HCCL 后如果新增了算法，
  需要手动同步这两张表，否则新算法采不了（按未注册组合被跳过）
- **黑名单可以放开**：排除逻辑集中在 `tuner_common.py` 的 `is_blacklisted`
  （concur 拦截在函数里写死，multilink 走关键词表），想采集这些算法时改这里

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

- `engine` / `executor` / `template`（必填）指定 cost table 中的目标位置
- `template` 对单级算法是单 template 名（如 `mesh`），对多级算法是 executor 后剩余串小写化的拼接（如 `meshconcurnhrnhr`）
- `cost`（必填）设置该位置的 cost 值，`0.0` 表示最优
- Selector 从 cost table 中选最小值确定算法

### engine 可选值

`aicpu` / `ccums` / `ccusched` / `aiv` / `dpu`

注意：`dpu` 与前 4 个 device 引擎**拓扑互斥、非同台竞争**。dpu 算法注册
`isHostDpuOnly=true`，仅 hostdpu 拓扑（多 server 且末层 CLOS 全连通，见
`CheckHostDPUOnly`）可选；普通环境 dpu 算法全部被 topo 过滤，hostdpu 环境
device 算法全部被过滤。因此 optimize_config 为 dpu rule 强制 `min_servers>=2`：
单机采集的 dpu 数据测得的是静默回退的其他算法耗时，属无效数据。

### executor 可选值

`sole` / `sequence` / `parallel` / `pipeline` / `concur` / `strictordered`

### template 可选值

template 不做枚举校验：单级是单 template 名，多级是 executor 后剩余串小写化的拼接串；rule 是否生效由 cost 表 templateName 匹配决定（Enrich 按 algName 派生），拼写错误匹配不到条目、靠运行时 warning 兜底。

单级：

`mesh` / `mesh2die` / `meshoneshot` / `meshtwoshot` / `meshconcur` / `meshmultilink` / `meshchunk` / `meshchunktwoshot` / `nhr` / `nhrmultilink` / `nhraicpureduce` / `nhrsinglechannel` / `meshconcurrent`

多级（拼接串）：

取 algName 中 executor 后的剩余串小写化，如 `AicpuAllReduceSequenceMeshConcurNHRNHR` → `meshconcurnhrnhr`。

```json
{ "engine": "aicpu", "executor": "sequence", "template": "meshconcurnhrnhr", "cost": 0.0 }
```

### 支持的 op_type

`allreduce` / `allgather` / `broadcast` / `reduce` / `reduce_scatter` / `scatter` / `alltoall` / `alltoallv`

> 以上 engine / executor / template / op_type 的可选值与 HCCL 算法选择器的维度定义一致。

### CSV 列格式

`hccl_prof.csv` 每行一个采集点，列固定为（供自行后处理参考）：

| 列 | 含义 |
|----|------|
| `op_type` / `data_type` / `reduce_type` | 采集维度（reduce 类之外的算子 `reduce_type` 为 `NA`） |
| `size_bytes` | 数据量（字节） |
| `engine` | 采集引擎（当前固定 `aicpu`） |
| `algorithm.executor_type` / `algorithm.template_type` | 算法（多级模板为逗号分隔串，如 `meshconcur,nhr,nhr`） |
| `HCCL_BUFFSIZE` | 采集时的 `HCCL_BUFFSIZE` 环境变量值 |
| `ranks.ranks` / `ranks.npus_per_server` / `ranks.servers` / `ranks.pods` / `ranks.super_pods` | 拓扑 5 字段（未指定的 pods/super_pods 记 0） |
| `check_result` | 结果校验状态：`success` 表示算法结果比对通过；`failed` / `noresult` 表示该轮执行失败或未产出数据，仅供参考，不参与选优 |
| `alg_bandwidth(GB/s)` / `alg_latency(us)` | 性能数据（optimize_config 以带宽选优） |
| `timestamp(ms)` | 毫秒时间戳（增量采集去重依据） |

## 测试

Python 单测（覆盖两个脚本的全部纯函数）：

```bash
python3 test_l0.py
```

C++ 插件测试：

```bash
cd test
make
./test_plugin
```

## 插件接口

插件需导出符号 `hcclTunerPlugin_v1`（类型 `hcclTunerFuncs_v1_t`），包含 `init` 和 `getCollInfo` 两个函数指针。HCCL 核心通过 `dlsym` 获取该符号加载插件。

详见 `include/hccl_tuner_plugin.h`。

## 运行示例

以下为 `./test_plugin` 的典型输出，展示插件的各种行为。

### 命中规则并覆盖 cost

AllReduce 8 ranks、4096B、fp32 匹配规则，将目标算法 cost 从 100.0 改为 0.0（最优偏好），Selector 会选这个算法：

```
[TunerDFX] rule hit: opType=2 nBytes=4096 dataType=3 ruleIdx=0/2
  engine=aicpu executor=sole template=meshoneshot cost=0.000000
[TunerDFX] modify: algName=AicpuAllReduceSoleMeshOneShot
  cost 100.000000 -> 0.000000
```

多级算法同样支持，`template` 为 executor 后剩余串小写化拼接：

```
[TunerDFX] rule hit: opType=2 nBytes=4096 dataType=3 ruleIdx=0/1
  engine=aicpu executor=sequence template=meshconcurnhrnhr cost=0.000000
[TunerDFX] modify: algName=AicpuAllReduceSequenceMeshConcurNHRNHR
  cost 5.000000 -> 0.000000
```

### 不匹配时不干预

4 ranks 不在规则要求的 8~16 范围内，无规则命中，cost table 保持 CostModel 原值：

```
[TunerDFX] no rule matched: opType=2 nBytes=4096 dataType=3
```

### 目标算法已禁用（cost<0）时跳过

目标条目 cost=-1（已被禁用），插件不改它，`SelectMinCost` 中 cost<0 视为 filtered 不参与选择：

```
[TunerDFX] rule hit: opType=2 nBytes=4096 dataType=3 ruleIdx=0/2
  engine=aicpu executor=sole template=meshoneshot cost=0.000000
[TunerDFX] skip disabled: algName=AicpuAllReduceSoleMeshOneShot cost=-1.000000
[TunerDFX] rule matched but no entry modified
```

### Schema 校验失败时整体不干预

配置有拼写错误、缺失必填字段、枚举非法等任意 schema error 时，插件完全不干预（安全降级）：

```
Schema: unknown field 'mtach' in rule, skipping
Schema: rule missing required field 'match'
Schema: invalid engine 'invalid_engine'
Schema: min_ranks(16) > max_ranks(8)
tuner config loaded, schemaErrors=4
Schema validation failed (4 errors), plugin will not intervene
```
