---
name: hccl-contribute
description: 本仓（hccl）从代码获取到 PR 合入的端到端贡献工作流。触发词：开发、贡献、上库、发PR、提交PR、提PR、过CI、触发CI、修CI、CI修复、处理检视意见、contribute、submit PR、clone hccl、下载代码、跑UT、跑测试。覆盖：代码下载/更新/worktree 隔离、依赖环境确认、本地构建与测试（调仓内命令）、Issue 查重与创建、fork 推送与 PR 创建（自动触发 CI）、CI 轮询与失败修复、检视意见处置、临时文件清理。支持 Windows 与 Linux。任意子流程可独立运行。
---

# hccl 贡献工作流（从零到 PR 合入）

从代码获取到 PR 合入的完整贡献链路。每个子流程可独立运行（本地已有仓/全新环境均可起步）。全程全自动执行，不要中途询问用户；输出统一用中文（代码标识符保留英文）。

## 按需配置（先看你要做什么）

**只 clone + 本地编译跑测试**（不提交 PR）：无需任何配置，直接走 Step 1→3。

**要提交 Issue/PR、查 CI、处理检视意见**：需 GitCode token（二选一）：

```bash
# a) 环境变量: export GITCODE_TOKEN=<token>  (Windows: $env:GITCODE_TOKEN 或 set)
# b) git credential 自动读取（已 git clone 凭据时无需配置）

# 配置校验（token 未配置时记 WARN 不阻断，纯 clone 场景可直接跳过）
python3 .agents/skills/hccl-contribute/scripts/contribute.py --check-env --repo-root <本仓路径>
```

代码目录：默认当前目录，或用 `--repo-root` 指定；首次 clone 用 `--parent-dir` 指定父目录。

验证脚本功能完好（无 GitCode API 依赖）：`python3 -m unittest discover -s .agents/skills/hccl-contribute/scripts -p test_contribute.py`，末行 OK 即正常。

依赖：Python 3.7+（仅标准库）、git、能访问 gitcode.com。Windows 与 Linux 均可运行 API/git 操作；本地构建须在 Linux 环境（见 Step 3）。commit 的 `git user.email` 必须与 CLA 签署邮箱一致，否则 PR 会被打 `cann-cla/no`。

## 工作流总览

```text
Step 1 代码获取与更新  →  Step 2 依赖环境确认  →  Step 3 本地构建与测试
                                                        ↓
Step 8 检视意见处置  ←  Step 7 CI 失败修复  ←  Step 6 CI 监控  ←  Step 5 PR 创建与提交
                                                        ↑
                                        Step 4 Issue 查重与创建
```

每步可独立运行：本地已有最新代码可从 Step 3/4 起步；只修 CI 从 Step 7 起步；只处理检视意见从 Step 8 起步。

## Step 1 代码获取与更新（可独立运行，无需 token）

```bash
# 已有仓：fetch 最新 master；干净则在 master 上快进，脏工作区自动建隔离 worktree（不动现有改动）
python3 .agents/skills/hccl-contribute/scripts/contribute.py --sync-repo --repo-root <本仓路径>

# 全新环境首次 clone（clone cann/hccl 到 <父目录>/hccl，upstream remote 已配好）
python3 .agents/skills/hccl-contribute/scripts/contribute.py --sync-repo --parent-dir <父目录>
```

输出 JSON 的 `action`：`cloned`（新 clone）/ `fetched_rebase`（master 快进）/ `fetched_worktree`（脏区隔离，用 `repo_root` 指向的新 worktree 继续工作）/ `up_to_date`（已最新）/ `noop`（在 feature 分支，只 fetch 不切）。

开发请在新分支上进行：`git checkout -b feature/issue-<N>-<简述> upstream/<base_branch>`（base_branch 用 `--sync-repo` 输出的值；分支命名前缀：`feature|fix|refactor|perf|docs|test/`）。

## Step 2 依赖环境确认（可独立运行，无需 token）

按仓内 [`docs/zh/build/build.md`](../../../docs/zh/build/build.md) 的「环境准备」节操作（前置依赖版本、CANN 安装、环境变量），本 skill 不复制其内容。最小校验：

```bash
# CANN 是否就绪（build.md 环境验证节）
source <CANN安装路径>/cann/set_env.sh && echo $ASCEND_HOME_PATH
```

环境只须准备一次；已有环境直接进 Step 3。环境类已知坑（详见 [`references/ci-triage.md`](./references/ci-triage.md)「环境坑」节）：

- **master 演进会引入新版 CANN 才有的符号**：编译报 `xxx was not declared in this scope` 且符号在 `acl*` 命名空间，先查本机 CANN 头文件是否含该符号（`grep <符号> $ASCEND_HOME_PATH/include/acl/acl_rt.h`），无则按 build.md 镜像站更新 CANN，不要改代码。
- **WSL 里 `bash -c "source .../set_env.sh"` 会静默失败**（脚本内 `read -r` 需要 stdin）：用 heredoc 方式跑，且跑完校验 `$ASCEND_HOME_PATH` 非空。

## Step 3 本地构建与测试（可独立运行，Linux 环境，无需 token）

按仓内 [`AGENTS.md`](../../../AGENTS.md) 第 4 节与 [`docs/zh/build/build.md`](../../../docs/zh/build/build.md) 的构建命令执行，推送前优先本地验证 `--pkg` + UT + ST。（构建命令以仓内文档为权威来源，本 skill 不复制。）

跑 UT/ST 前的两个前置（缺了会出现成片测试失败，详见 [`references/ci-triage.md`](./references/ci-triage.md)「环境坑」节）：

- **UT 的 aicpu 用例依赖 device kernel 配置**：须先 `build.sh --pkg --full` 生成完整包并安装到 CANN 树（`chmod -R u+w $CANN` 后 `bash build_out/cann-hccl_*.run --full --install-path=$CANN`），否则 CollServiceAiCpu 等套件报 `ccl_kernel.json is not a valid real path`。
- **执行测试时须在已 source CANN 环境变量的同一 shell 里跑** ctest/build.sh，否则测试进程按默认路径找 CANN 报同上错误。

pre-commit（clang-format + OAT 许可检查）本地跑法：`pip3 install pre-commit && pre-commit run --files <改动文件>`；OAT 检查也可 `bash scripts/oat_check.sh <文件>`（exit=0 才通过）。新增源文件须带 CANN-2.0 许可头（对照仓内已有文件逐字节一致）。

## Step 4 Issue 查重与创建（可独立运行，需 token）

所有 PR 必须关联 Issue。先查重避免重复立项：

```bash
# 按标题关键词查重（open+closed）；已有则复用输出编号，无则按仓惯例前缀创建
python3 .agents/skills/hccl-contribute/scripts/contribute.py --issue-ensure \
  --repo hccl --title "<需求标题>" --body-file <issue_body.md>
```

输出 `action: reused`（复用已有 open Issue）/ `reused_closed`（同主题 Issue 已关闭，把 reopen 或新建的决定权交回调用方）/ `created`（新建，标题自动加 `[Requirement|需求建议]: ` 前缀）。

## Step 5 PR 创建与提交（可独立运行，需 token）

```bash
# 前置：已在 feature 分支 commit、工作树干净、--check-env 通过
python3 .agents/skills/hccl-contribute/scripts/contribute.py --submit-pr \
  --repo-root <本仓路径> --repo hccl \
  --title "[feat|fix|docs] <描述>" --body-file <pr_body.md> --issue <N>
```

脚本自动完成：git 身份校验 → push fork（`--force-with-lease`）→ 创建 PR（head 用 `账号:分支` 格式；已存在时自动复用）→ 评论 `/compile` 触发 CI → GET 回查。PR 描述必须按仓内 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md` 模板六章节填写。

## Step 6 CI 监控（可独立运行，需 token）

```bash
# 单次查询
python3 .agents/skills/hccl-contribute/scripts/contribute.py --ci-status --repo hccl --pr <N>

# 轮询至终态（默认 60s 间隔 / 30min 超时）
python3 .agents/skills/hccl-contribute/scripts/contribute.py --ci-status --repo hccl --pr <N> --wait
```

`state` 语义：`running`（流水线运行中）/ `passed`（本轮通过）/ `failed`（本轮失败，进 Step 7）/ `finishing`（running 已消失、终态标签未上，稍等再查）/ `stale`（无 running 历史的残留标签，不可作为本轮结论——此时若刚 push 过应确认 /compile 已触发）/ `not_triggered`（从未触发，需评论 `/compile`）/ `timeout`（轮询超时未达终态，稍后重查）。

**每次 push 后必须重新 `/compile`**（push 会自动失效旧 `ci-pipeline-passed`）；触发后不要重复触发（会打断在跑轮次）。

## Step 7 CI 失败修复（可独立运行，需 token）

```bash
# 收集失败信息：流水线链接 + pre-commit/markdownlint 日志（OBS 直链免登录下载）
python3 .agents/skills/hccl-contribute/scripts/contribute.py --ci-logs --repo hccl --pr <N> --output-dir ./ci_logs
```

诊断与修复按 [`references/ci-triage.md`](./references/ci-triage.md) 的失败模式表执行（OAT 许可头 / codecheck 规则 / markdownlint / 已知非阻塞项）。修复流程：定位失败任务 → 按模式修复 → commit → push（Step 5 的 push 逻辑）→ `/compile` 重触发 → 回 Step 6 轮询。重复直到 `passed`。

## Step 8 检视意见处理（可独立运行，需 token）

```bash
# 列出 PR 未 resolved 的检视意见（含文件/行号/作者/内容）
python3 .agents/skills/hccl-contribute/scripts/contribute.py --list-review-comments --repo hccl --pr <N>
```

对每条意见：先核查事实（`grep` 确认检视者说的是否属实）→ 属实则修改 → commit + push → `/compile` → 回复处置。**回复必须发到原意见线程，不要发独立顶层评论**；自己提的意见修复后同时 resolve。线程回复与 resolve 用 `--list-review-comments` 输出的 `discussion_id`（v5 端点，见 [`references/gitcode-api.md`](./references/gitcode-api.md)「检视意见处置」）。

## 清理

```bash
# 移除 worktree 与临时文件
python3 .agents/skills/hccl-contribute/scripts/contribute.py --cleanup \
  --repo-root <本仓路径> [--worktree <worktree路径>]
```

## 执行纪律

- 全自动执行：不中途询问"是否继续"。
- 破坏性命令、`git commit`、`git push` 必须得到用户明确许可（仓 AGENTS.md 要求）；脚本涉及 push 的操作前确认用户已许可本次提交。
- 严禁向官方仓发测试 PR / 测试评论：验证一律走 `--check-env`、只读查询或自己的 fork 彩排。
- PR 必须关联 Issue；PR 描述与实现保持一致（内容演进后同步更新描述与 Issue）。
- CI 触发后不重复触发 `/compile`（重复触发打断在跑轮次并留下误导性 failed 标签）。
- 构建命令、依赖版本、编码规范以仓内 `AGENTS.md`、`docs/zh/build/build.md` 为权威来源，本 skill 不复制其内容。

## 更多细节

GitCode API 端点、认证方式、CI 标签语义与已验证坑见 [`references/gitcode-api.md`](./references/gitcode-api.md)；CI 失败诊断模式见 [`references/ci-triage.md`](./references/ci-triage.md)。
