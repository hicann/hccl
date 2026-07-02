# experimental/ —— 开发者实验与贡献目录

## 1. 目的

`experimental/` 是为 HCCL 社区提供的**实验空间**，与主干 `src/` 的区别：

| | src/ | experimental/ |
|---|---|---|
| 目标 | 生产级代码 | 快速原型验证 |
| 审核 | RFC + SIG 评审 | RFC + SIG 评审 |
| 质量 | 生产级 | 原型级 |
| 稳定性 | 承诺 API 稳定 | 不保证 |

---

## 2. 目录结构

```
experimental/
│
├── ops/                             # 扩展 HCCL 通信算子
└── eco_system/                       # 围绕 HCCL 的生态工具与周边创新
```

> 扩展规则：新类别一律在对应大类下增加子目录。如需新增一级目录，需经 SIG 讨论。

---

## 3. 贡献流程

### 3.1 快速贡献（experimental）

```
Step 1: 确定归属
    ├── 要改 HCCL 库内部？  → ops/
    └── 围绕 HCCL 做工具？  → eco_system/

Step 2: 创建子目录
    ops/<category>/<project_name>/
    或
    eco_system/<category>/<project_name>/

Step 3: 编写 README.md（必需）
    至少包含：动机、设计、用法、现状、限制

Step 4: 提交 PR
    标题：[experimental] ops|eco_system/<category>/<project>: <简述>
    目标分支：master（直接合入 experimental/）
    审核标准：
      ✅ 目录位置正确
      ✅ README.md 完整
      ✅ 代码不修改 experimental/ 之外的任何文件
```

---

## 4. 子目录模板

### 最小模板

```
experimental/<ops|eco_system>/<category>/<project_name>/
├── README.md          # 必需
└── ...                # 其他文件自由组织
```

### 推荐模板（C++ 项目）

```
experimental/<ops|eco_system>/<category>/<project_name>/
├── README.md          # 必需：动机、设计、用法、现状、限制
├── CMakeLists.txt     # 推荐：独立编译
├── src/               # 源代码
├── include/           # 头文件（如有）
├── test/              # 测试（推荐）
└── example/           # 使用示例（推荐）
```

---

## 5. 运行期开关

为防止实验性贡献意外开启影响主干，贡献特性必须通过运行期开关控制是否生效：

**开关命名**：环境变量 `HCCL_EXPERIMENTAL_<NAME>=true`，其中 `<NAME>` 使用大写下划线分隔的英文名（与特性名一致）。

**启用模式**：在项目入口提供 `IsXxxEnabled()` 函数（函数名不强制），内部按以下逻辑判断：

```cpp
bool IsXxxEnabled() {
    constexpr bool xxxEnabled = false;  // 默认 false
    if (!xxxEnabled) return false;
    const char* env = getenv("HCCL_EXPERIMENTAL_<NAME>");
    return env && std::string(env) == "true";
}
```

- 常量值默认 `false`；常量优先级高：常量为 `false` 时直接返回 `false`。
- 常量为 `true` 时再判断环境变量是否为 `"true"`，是则启用。
- 特性入口必须 `if (IsXxxEnabled()) { ... }` 守卫。


## 6. 维护策略

| 策略 | 说明 |
|------|------|
| **稳定性** | experimental 代码不保证 API 稳定性，可随时变更 |
| **季度审查** | 维护者每季度扫描，标记 6 个月无活动项目为 `stale` |
| **代码规范** | 建议遵循 CANN 社区规范，但非强制 |
| **Issue 跟踪** | 实验项目的问题在项目 README 中自行声明，不占用主 Issue |

### Stale 判定标准

- 最近 6 个月内无 commit
- 最近 6 个月内无 Issue/PR 活动
- README 中「现状」标记为已完成或无后续计划

---
