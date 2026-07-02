# HCCL Test

本目录包含 HCCL 的测试代码，分为系统测试（ST）和单元测试（UT）两大类。

## 目录结构

```
test/
├── st/                         # 系统测试（System Test）
│   ├── algorithm/              # 算法分析器测试
│   │   ├── testcase/           # 算法测试用例
│   │   ├── utils/              # 测试工具代码
│   │   │   └── src/
│   │   │       ├── aicpu/          # AICPU 相关打桩
│   │   │       ├── common/         # 公共工具
│   │   │       │   ├── exception/  # 异常处理
│   │   │       │   └── utils/      # 工具函数
│   │   │       ├── hccl_depends_stub/  # HCCL 依赖接口打桩
│   │   │       ├── hccl_proxy/     # 模拟通信器实现
│   │   │       │   ├── communicator/
│   │   │       │   └── topo_model/
│   │   │       ├── hccl_verifier/  # 验证器
│   │   │       │   ├── mem_conflict_check/   # 内存冲突校验
│   │   │       │   ├── semantics_check/     # 语义校验
│   │   │       │   ├── singletask_check/   # 单任务检查
│   │   │       │   └── task_graph_generator/# Task图生成
│   │   │       ├── sim_world/     # 仿真世界实现
│   │   │       └── ut/            # 算法分析器 UT 测试
│   │   ├── figures/            # 测试说明图片
│   │   ├── CMakeLists.txt
│   │   ├── README.md           # 算法分析器详细使用指南
│   │   └── build.sh            # 编译执行脚本
└── ut/                         # 单元测试（Unit Test）
    └── common/
        └── prepare_ut_env/     # UT 环境准备代码
```

## 测试类型

### 系统测试（ST）

系统测试主要验证 HCCL 集合通信算法逻辑的正确性，包括内存操作校验和语义校验。

#### 算法分析器

算法分析器通过模拟 HCCL 单算子运行流程，验证算法逻辑及内存操作功能。

**原理介绍：**

1. 算法分析器通过对运行HCCL单算子流程的依赖（hcomm和runtime接口）进行打桩，在算法执行过程中获取所有rank的Task序列。
2. 将所有rank的Task信息组成一张**有向无环图**。
3. 基于**图算法**进行校验，包括内存读写冲突校验和语义校验。

**核心功能：**

- **内存冲突校验**：基于Task图中的同步情况分析是否存在可能的读写冲突。
- **语义校验**：通过模拟执行Task图，记录数据搬运信息，验证输出内存中的数据搬运是否符合算子要求。

详情见 [算法分析器使用指导](./st/algorithm/README.md)。

### 单元测试（UT）

在仓库根目录下执行如下命令：

```bash
# 编译并运行所有单元测试用例
bash build.sh --ut

# 编译并运行所有系统测试用例
bash build.sh --st
```
