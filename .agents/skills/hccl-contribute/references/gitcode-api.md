# GitCode API 参考（贡献流程相关）

贡献流程用到的 GitCode API 端点、认证方式与已验证行为规律。

## 双 API 层

| API | 基址 | 认证 | 用途 |
|-----|------|------|------|
| v5 | `https://gitcode.com/api/v5/` | Bearer 头（脚本内置）或 `access_token` 查询参数 | PR/Issue 创建与查询、评论、标签、检视意见（本 skill 唯一数据源） |
| v4 | `https://api.gitcode.com/api/v4/` | `PRIVATE-TOKEN` 请求头 | 仅供扩展参考，本 skill 不调用（discussions 的 resolved 恒 None、翻页重复且数据不全，实测不可靠） |

**v4 域名必须是 `api.gitcode.com`**（`gitcode.com/api/v4` 返回 HTML 非 JSON）。

## 端点速查

```text
GET  /api/v5/user                                             token 账号校验
POST /api/v5/repos/cann/hccl/issues                          创建 Issue
GET  /api/v5/repos/cann/hccl/issues?state=open               Issue 查重
POST /api/v5/repos/cann/hccl/pulls                           创建 PR
GET  /api/v5/repos/cann/hccl/pulls/{n}                       PR 元数据（labels/state/mergeable）
GET  /api/v5/repos/cann/hccl/pulls/{n}/comments              PR 评论（流水线链接在 cann-robot 评论里）
POST /api/v5/repos/cann/hccl/pulls/{n}/comments              评论（/compile 触发 CI）
GET  /api/v5/repos/cann/hccl/pulls/comments/{id}             单条评论详情（position.new_path 补文件路径）
```

## 关键语义与坑

- **owner 用 `cann`**：PR/Issue 数据挂在官方仓，API 里 `{owner}` 不用 fork owner。
- **PR 创建 `head` 格式**：`{fork用户名}:{分支名}`；fork 改过名时用 `{fork_owner}/{fork_repo}:{branch}` 更稳。
- **PR 已存在**：POST 返回 422 `already exist`，按 `state=open` 列表查回已有 PR 复用，不要重复创建。
- **Issue `labels` 禁数组**：v5 创建 Issue 带 `labels` 数组必 400；标题按仓模板前缀即可，打标由 maintainer 处理（模板预设的 labels 网页创建时自动带上，API 创建不带）。
- **CI 触发**：POST 评论 `{"body": "/compile"}`；**每次 push 后须重新触发**（push 自动移除 `ci-pipeline-passed` 标签，cann-robot 会发 Notification）。
- **CI 标签流转**：`ci-pipeline-failed` →(触发)→ `ci-pipeline-running` →(结束)→ `ci-pipeline-passed` 或 `ci-pipeline-failed`。
- **终态判定须 saw_running**：`ci-pipeline-passed`/`failed` 可能是旧 run 残留；必须先见 `running` 出现且消失，再看终态标签（脚本已内置状态机）。
- **push 分支与 PR head 一致**：PR 追踪 fork 的特定分支，push 目标分支必须与 PR head.ref 相同。
- **commit 邮箱 = CLA 邮箱**：不一致会被打 `cann-cla/no`，可评论 `/cla` 重查。

## CI 日志直链（免登录）

```text
https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/hccl/package/{PR号}/pre-commit.txt
https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/hccl/package/{PR号}/markdownlint.csv
```

流水线详情页 `https://www.openlibing.com/apps/pipelineDetail?pipelineId=...` 的任务日志需浏览器打开（链接在 cann-robot 的触发评论里）。

## 检视意见处置（线程回复 + resolve）

```text
POST /api/v5/repos/cann/hccl/pulls/{n}/discussions/{did}/comments   线程回复（did 为 hex discussion_id）
PUT  /api/v5/repos/cann/hccl/pulls/{n}/comments/{did}               resolve（body {"resolved": true}）
```

两端点均须 Bearer 头认证。did 从 contribute.py `--list-review-comments` 输出的 `discussion_id` 字段获取。他人意见只回复不 resolve（关闭权在提出者）；自提意见修复后 reply+resolve 一站式闭环。

## 错误码与限速

| HTTP | 含义 | 处置 |
|------|------|------|
| 200/201 | 成功 | 写操作仍须 GET 回查 |
| 400 | 参数错误 | 检查 labels 数组 / head 格式 |
| 401 | token 无效 | 检查 GITCODE_TOKEN |
| 422 | 已存在 | 查列表复用已有 Issue/PR |
| 429 | 限速 | 等 60s 重试（脚本 api_request 已内置一次重试） |

- 中文 payload：脚本用 Python urllib UTF-8 编码请求体；手动 curl 时写文件后 `--data-binary @file.json`，禁止 `-d` 内联中文。
- Windows 触发 `/compile` 用 Python/PowerShell，不用 Git Bash（`/compile` 被路径转换毁掉）。
