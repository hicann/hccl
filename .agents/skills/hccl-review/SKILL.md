---
name: hccl-review
description: 检视本仓（hccl）在 GitCode 上的 PR。触发词：检视、检视PR、检视本仓PR、代码检视、代码审查、review PR、code review、检视PR123、review hccl PR。从 GitCode 拉取指定 PR 的代码到隔离 worktree，多维度检视（代码质量/风格/逻辑/注释/架构/AGENTS.md 合规/描述与实现吻合度/功能正确性），把检视意见按行提交到 PR（自动行号验证、与已有评论去重），提交检视汇总报告，最后清理 worktree 与临时文件。支持 Windows 与 Linux。
---

# hccl PR 代码检视

从 GitCode 拉取指定 PR 的代码到隔离 worktree，多维度检视，按行提交检视意见，提交汇总报告，清理现场。任意给一个 PR 即可检视。全程全自动执行，不要中途询问用户。检视意见统一用中文（代码标识符、severity 级别名保留英文）。

## 首次使用配置

```bash
# 1) GitCode token（二选一）:
#    a) 环境变量: export GITCODE_TOKEN=<token>  (Windows: $env:GITCODE_TOKEN 或 set)
#    b) git credential 自动读取（已 git clone 凭据时无需配置）
# 2) 本仓本地 clone 路径: 默认当前目录，或用 --repo-root 指定

# 配置校验（通过后无需再配）
python3 .agents/skills/hccl-review/scripts/pr_review.py --check-env --repo-root <本仓路径>
```

依赖：Python 3.7+（仅标准库）、git、能访问 gitcode.com。Windows 与 Linux 均可运行。

## 检视流程（五步）

用户说"检视 PR123"（或给 PR 链接）时，解析 PR 编号后按 Step 1→5 顺序执行。

### Step 1 准备基线与 worktree

- 取 PR 元数据（state、目标分支、base/head sha、变更文件数）：

```bash
python3 .agents/skills/hccl-review/scripts/pr_review.py --pr <N> --meta --repo-root <本仓路径>
```

- **确认目标分支**：元数据的 `target_branch` 是 `master` 还是其他分支。worktree 基于该分支的基线创建：

```bash
cd <本仓路径>
git fetch <remote> <target_branch>                    # remote 用 origin 或 upstream（指向 cann/hccl 的那个）
git fetch <remote> refs/merge-requests/<N>/head:refs/remotes/<remote>/mr/<N>-head
git worktree add --detach .wt/review-<N> refs/remotes/<remote>/mr/<N>-head
```

- **基线核验**（防基线漂移把意见发到非本 PR 的代码上）：本地 `git diff <base_sha>..<head_sha>` 的文件数应与元数据 `changed_files` 一致；不一致时以 `--pr <N> --files` 的 API 权威列表为准，禁止据本地 diff 检视。

- `state == "merged"` 时跳过 Step 4，把检视意见整理成 Issue 跟踪（合入后无法按行提交）。

### Step 2 读代码与行号实证

- 一律 `git show <head_sha>:<path>` 读 head 版本代码，**禁止读本地工作区文件**（可能与 head 不一致）。
- 记录每条检视意见前，必须 `git show <head_sha>:<path> | grep -n "<代码片段>"` 取真实行号，**禁止估算行号**。
- findings.json 每条记录 `code_snippet` 字段（该行内容片段），供脚本自动验证行号。

### Step 3 多维检视

**先读检视规范与仓内权威资料，再检视**（检视规范按五类分档，见 [`references/README.md`](./references/README.md) 索引，按 PR 触碰范围加载）：

| 资料 | 用途 |
|------|------|
| [`references/`](./references/README.md) 检视规范 | 编码安全红线 / 对外 API / 架构合规 / PR 完备性 |
| 根目录 `AGENTS.md` | 架构约束（分层依赖/HCOMM 解耦/新算子目录结构）、编码规范、目录职责 |
| `docs/zh/architecture/architecture-brief.md` | 架构权威来源（改 src/include 前必读） |
| `.clang-format` / `CONTRIBUTING.md` / [CANN 编码规范](https://gitcode.com/cann/community/tree/master/contributor/coding-standards) | 代码风格与贡献规范 |
| PR 关联的 Issue 与 PR 描述 | 功能意图（对照实现是否吻合、是否完整） |

**通用检视维度**（不限于）：正确性（逻辑/边界/未初始化）、资源生命周期（泄漏/Init-Cleanup 对称）、错误处理、并发安全、API 兼容性（include/ 对外接口向后兼容）、安全（越界/注入）、架构合规（对照 AGENTS.md 约束）、命名与风格（对照仓内规范）、注释完整性、性能、测试覆盖（生产代码变更是否补 UT/ST）、PR/Issue 描述与实现的吻合度。

**并行检视**（变更 ≥10 个源文件时）：按模块把文件分组，并行派发多个子 agent，每组返回结构化 JSON findings（schema 见下）。文件少于 10 个时单线程检视。

```json
[{"severity": "CRITICAL|HIGH|MEDIUM|LOW", "file": "src/path.cc", "line": 42,
  "dimension": "Resource Lifecycle", "title": "简短描述",
  "body": "问题描述 + 建议修复（必须给具体改法，不只点问题）",
  "code_snippet": "该行内容片段"}]
```

### Step 4 汇总与提交

```bash
# 4a 行号验证（不通过会拒绝提交，修正 findings 后重跑）
python3 .agents/skills/hccl-review/scripts/pr_review.py --pr <N> --verify-only \
  --findings findings.json --repo-root <本仓路径>

# 4b 预演（看每条意见的 position 与回退计划）
python3 .agents/skills/hccl-review/scripts/pr_review.py --pr <N> --dry-run \
  --findings findings.json --repo-root <本仓路径>

# 4c 提交（自动与已有评论去重；行不在 diff 时自动回退为带 "> file:line" 前缀的 PR 评论）
python3 .agents/skills/hccl-review/scripts/pr_review.py --pr <N> \
  --findings findings.json --repo-root <本仓路径>

# 4d 检视汇总报告（一条 PR 评论）
python3 .agents/skills/hccl-review/scripts/pr_review.py --pr <N> --report \
  --findings findings.json --repo-root <本仓路径>
```

提交前对每条意见自问："这是真实问题吗？有无反例？该行实际内容匹配吗？是不是已有代码而非本次变更？"——存疑则不发。

### Step 4e 检视意见处置（修复检视意见后）

修复检视意见（自己提的或他人的）后，**回复必须发到原检视意见的线程内，不要发独立的顶层评论**；自己提的意见修复后要同时 resolve 关闭。一律用脚本完成，不要手工调 API：

```bash
# 列出本账号在该 PR 的检视意见与状态（含 discussion_id）
python3 .agents/skills/hccl-review/scripts/pr_review.py --pr <N> --list-findings --repo-root <本仓路径>

# 线程回复 + resolve 关闭（自提意见修复后的标准闭环）
python3 .agents/skills/hccl-review/scripts/pr_review.py --pr <N> --dispose --line <行号>   --body "已修复（commit <sha>）：<修复说明>" --repo-root <本仓路径>

# 只回复不关闭（他人意见、或暂不关闭时）
python3 .agents/skills/hccl-review/scripts/pr_review.py --pr <N> --dispose --line <行号>   --body "<回复内容>" --no-resolve --repo-root <本仓路径>
```

### Step 5 清理

```bash
python3 .agents/skills/hccl-review/scripts/pr_review.py --pr <N> --cleanup \
  --worktree <本仓路径>/.wt/review-<N> --repo-root <本仓路径>
```

移除 worktree、prune、删除临时 findings 文件。

## 执行纪律

- 全自动执行：不中途询问"是否继续""是否提交"。
- 检视意见**按行单条提交**，每条必须含可操作的修复建议，统一用中文。
- **严禁测试评论**：不以调试/验证为由发任何测试性评论；验证一律用 `--verify-only` / `--dry-run`。
- 提交前脚本自动去重：新意见相互去重，且不与 PR 上已有评论重复。
- **处置检视意见不发独立评论**：回复走原意见线程（`--dispose`），自提意见修复后同时 resolve 关闭。
- **代码变更后同步描述**：PR 内容演进（新增文件/功能/测试）后，同步更新 PR 描述与关联 Issue，保持描述与实现一致（对照 `references/pr-completeness.md` 的吻合度要求）。

## 更多细节

检视规范索引见 [`references/README.md`](./references/README.md)；GitCode API 端点、认证方式、position 语义与已知坑见 [`references/gitcode-api.md`](./references/gitcode-api.md)。
