# 检视规范：架构合规

检视 `src/` 变更时逐条对照。权威来源：根 `AGENTS.md` 第 3 节、`docs/zh/architecture/architecture-brief.md`（「3 软件分层逻辑」+ 末尾「软件架构约束说明」）。改 `src/`、`include/`、`pkg_inc/` 前必须先读 architecture-brief。

## 分层依赖方向（硬性）

依赖方向自上而下：`coll_comm_ops`（本仓）→ `coll_communicator_mgr`（hcomm）→ `base_comm`（hcomm）。

- HCCL 不得被 HCOMM 反向依赖；不得要求 HCOMM 反向 include HCCL 头
- HCCL 算子通过 dlsym 调 HCOMM，跨仓调用走 `src/common/hcomm_dlsym/` 的符号表 + `dlsym`
- 新增类/函数先定层级，只调用同层或更下层

## 控制面/数据面分离

- 资源管理、拓扑查询（控制面）与数据搬运、同步（数据面）接口独立演进、互不耦合
- 不得在数据面原语中引入控制面强耦合；控制面不得依赖具体数据面算子实现

## 新算子落标准结构

- 官方新算子落 `src/ops/<op>/`；社区试验性新算子落 `experimental/ops/<op>/`（结构与 src 一致，不保证兼容性、不编入商用版本）
- 均按 `executor/selector/template` 组织；新算子须提供 selector（算法选择）与 template（引擎模板：aicpu/aiv/ccu）
- 禁止散落其他目录

## 目录与结构对齐

- 目录结构对齐 architecture-brief 3.2（`src/ops/` 按算子组织、`src/common/` 通用逻辑、`src/op_common/` 四大通用组件）
- `src/` 重命名/移动时同步检查：`CMakeLists.txt`、测试 include 路径、`#include` 相对路径、`cmake/`、`build.sh`
- 重命名 PR 检查是否混入 brief 未明确要求的无关修改

## 高风险区（变更时提高检视强度）

| 区域 | 风险 |
|------|------|
| 算子主流程（`src/ops/<op>/`） | 算法正确性、selector 覆盖、template 引擎匹配 |
| 通用组件（`src/ops/op_common/` 的 executor/selector/template/topo） | 算法选择逻辑、执行器引擎分发 |
| hcomm_dlsym（`src/common/hcomm_dlsym/`） | 符号表与 dlsym 加载、版本兼容 |
| 环境配置（`src/common/alg_env_config/`） | 环境变量解析与默认值 |
| 图模式与 MC2（`src/common/op_graph/`、`src/common/hccl_mc2/`） | 图构建正确性、自定义算子框架兼容 |
