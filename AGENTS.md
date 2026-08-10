# HCCL Agent 规则

本文件是 HCCL 仓的 AI Agent 治理主入口，面向各类支持 AGENTS.md 标准的 AI 编程工具。子目录如存在自己的 `AGENTS.md`，以更近的文件补充本文件；冲突时先遵守用户当前明确要求，再遵守更近的子目录规则。

> 本文件只给**必须知道的硬约束与入口**，详细内容通过链接渐进式披露。改动代码前，请先读第 3 节架构约束。

## 1. 仓库定位

HCCL（Huawei Collective Communication Library）是 CANN 的核心集合通信库，提供 AllReduce、Broadcast、AllGather、ReduceScatter、AlltoAll 等集合通信原语与 Send/Recv 等点对点通信，支持单算子与图模式。对上支持 AI 框架，对下通过 HCOMM 通信基础库（独立仓 `cann/hcomm`）使能昇腾 NPU。

**软件组成**：HCCL = HCCL 集合通信算子库（本仓 `cann/hccl`）+ HCOMM 通信基础库（`cann/hcomm`）。两仓通过 `dlsym` 动态加载解耦，可独立编译、独立版本演进。详见 [`README.md`](./README.md) 与 [架构简介](./docs/zh/architecture/architecture-brief.md)。

## 2. 目录结构

```text
src/
├── ops/                  # 集合通信算子实现（all_reduce/all_gather/broadcast/reduce_scatter/send/recv/...）
│   └── op_common/        # 四大通用组件：executor/selector/template/topo
└── common/               # 通用逻辑：adapter_acl/alg_env_config/log/param_check/sal/hcomm_dlsym/op_graph/utils/hccl_mc2
experimental/             # 社区贡献的试验性代码（结构与 src 一致，含 ops/；不保证兼容性，不编入商用版本）
include/                  # 对外头文件：hccl.h（算子 API）、hccl_mc2.h（MC2 自定义算子框架）
test/                     # ut / st
docs/                     # 资料文档
build.sh                  # 一键编译脚本
```

目标目录结构与各层职责见 [架构简介 3.2 节](./docs/zh/architecture/architecture-brief.md)。

## 3. 软件架构与架构约束（核心）

> **架构权威来源**：[`docs/zh/architecture/architecture-brief.md`](./docs/zh/architecture/architecture-brief.md)。改动 `src/`、`include/` 前必须先读其「3 软件分层逻辑」与末尾「软件架构约束说明」。

### 分层

| 软件层次 | 仓位置 |
|----|--------|
| HCCL 集合通信算子（coll_comm_ops，L1） | 本仓 `cann/hccl` |
| HCOMM 集合通信域管理（HCCM，L2） | `cann/hcomm` |
| HCOMM 基础通信（L3） | `cann/hcomm` |

依赖方向自上而下：`coll_comm_ops`（HCCL）→ `coll_communicator_mgr` → `base_comm`（后两者在 HCOMM 仓）。

### 架构约束（硬性，不可违反）⭐

| 约束 | AI Agent 行为要求 |
|------|------------------|
| **分层依赖方向**：上层依赖下层，下层不能反向依赖上层（HCOMM 的 `base_comm` ↛ `coll_communicator_mgr` ↛ `coll_comm_ops`） | HCCL 不得被 HCOMM 反向依赖；HCCL 算子通过 dlsym 调 HCOMM，不得要求 HCOMM 反向 include HCCL 头 |
| **控制面/数据面分离**：资源管理、拓扑查询（控制面）与数据搬运/同步（数据面）接口独立演进 | HCCL 算子属数据面消费方；不得在算子层引入对 HCOMM 控制面内部实现的耦合 |
| **HCCL 与 HCOMM 解耦**：HCCL 算子通过 `dlsym` 动态加载 HCOMM 接口，两仓独立编译、独立版本演进 | HCCL 不得 `#include` HCOMM 私有头；不得引入对 `cann/hcomm` 的编译期硬依赖；跨仓调用走 `src/common/hcomm_dlsym/` 的符号表 + `dlsym` |
| **新算子落标准结构**：官方新算子落 `src/ops/<op>/`；社区贡献的试验性新算子落 `experimental/ops/<op>/`（结构与 `src` 一致，不保证兼容性、不编入商用版本）。均按 `executor/selector/template` 组织 | 新算子须提供 selector（算法选择）与 template（引擎模板：aicpu/aiv/ccu）；官方算子落 `src/ops/`，社区试验算子落 `experimental/ops/`，禁止散落其他目录 |

### 对外 API

| 层次 | 头文件 | 面向 |
|------|------|------|
| L1 算子 | `include/hccl.h` | AI 框架适配层（AllReduce/Broadcast/AllGather/ReduceScatter/AlltoAll/Send/Recv 等） |
| MC2 自定义算子 | `include/hccl_mc2.h` | 自定义通信算子开发者（KfcOpArgs/OpResCtx 等） |

`include/` 变更需向后兼容。完整 API 分层关系见 [架构简介 3.3 节](./docs/zh/architecture/architecture-brief.md)。

## 4. 构建与测试

```bash
bash build.sh --pkg                      # 编译 host 包（默认）
bash build.sh -u                         # 编译并运行 UT
bash build.sh -s                         # 编译并运行 ST
bash build.sh --static                   # 静态库构建
bash build.sh --asan                     # 启用 AddressSanitizer
bash build.sh --custom_ops_path=<PATH>   # 自定义算子工程
bash build.sh -j64                       # 并行编译
```

完整选项、环境准备与上板测试见 [`docs/zh/build/build.md`](./docs/zh/build/build.md)。推送前优先本地验证 `--pkg` + `--ut` + `--st`。

## 5. 编码规范

- 命名：类/函数 PascalCase；成员变量 `camelCase_`（小驼峰+后缀下划线）；常量与宏 `UPPER_SNAKE_CASE`。
- 风格：遵循根目录 `.clang-format`（120 列、4 空格、指针右对齐、K&R 大括号）；C++14。
- 静态告警：代码须通过 CANN 静态检查要求（CI codecheck 阶段校验），编译无告警。
- pre-commit：clang-format v18.1.8 + OAT 合规检查；新增源文件须带 CANN-2.0 许可头。

参考：[CANN 编码规范](https://gitcode.com/cann/community/tree/master/contributor/coding-standards)、[CANN CI 指南](https://gitcode.com/cann/community/blob/master/contributor/repository/ci-guide.md)、[pre-commit 指导](./docs/zh/build/pre-commit-guide.md)、`.clang-format`、`OAT.xml`。

## 6. 文档编写规范

- 目录组织：`docs/zh/`（中文）与 `docs/en/`（英文）分开存放；API 文档使用 PascalCase（如 `HcclAllReduce.md`）；环境变量文档使用 UPPER_SNAKE_CASE（如 `HCCL_ALGO.md`）。
- **产品名称特殊规则**：单位与数字、中文与英文之间不加空格（如 `50m`、`昇腾AI处理器`），但**产品名称内部允许保留空格**（如 `Ascend 950PR/Ascend 950DT`、`Atlas A3 训练系列产品`），以保持官方产品标识的完整性和可读性。

参考：[CANN 文档编写规范](https://gitcode.com/cann/community/blob/master/contributor/docs/document_writing_specs.md)。

## 7. 贡献流程

- 简单问题：Issue → 认领 → PR → Committer 检视 → `/lgtm` + `/approve` 合入。
- 新功能：Requirement Issue → SIG 决策 → `docs/zh/rfcs/` RFC 评审 → 实现（含 UT+ST）→ 检视合入。
- 所有 PR 必须关联 Issue，描述按 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` 填写。

详见 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

## 8. Agent 工作原则

- 优先小而可审查的变更；除非用户明确要求，避免大范围重构。
- 编辑前先定位文件，用 3-6 条说明计划。
- 不确定 API、配置、路径或事实时，先搜索仓库或查证，不要臆造。
- 改动 `src/` 前先对照第 3 节架构约束：是否违反分层依赖？是否引入对 `cann/hcomm` 的编译期硬依赖（应走 dlsym）？新算子是否落 `src/ops/` 标准结构？
- 严禁把密钥、token、密码、私钥、`.env` 值或凭据写入代码、日志或回复。
- 除非用户要求，不新增遥测、分析上报或额外网络调用。
- 行为变更应在项目已有测试体系下补充或更新测试；优先跑最快相关验证。
- 涉及 `src/` 目录重命名/移动时，同步检查 `CMakeLists.txt`、测试 include 路径、`#include` 相对路径，并清理 build 目录后重新验证。
- 破坏性命令、`git commit`、`git push` 必须得到用户明确许可。
- 默认用中文解释；输出保持简洁、具体、可复制。

---

*架构约束以 [`docs/zh/architecture/architecture-brief.md`](./docs/zh/architecture/architecture-brief.md) 为权威来源；贡献流程以 [`CONTRIBUTING.md`](./CONTRIBUTING.md) 为准。*
