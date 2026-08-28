# GitCode API 参考（PR 检视相关）

检视 skill 用到的 GitCode API 端点、认证方式与已验证行为规律。

## 双 API 层

| API | 基址 | 认证 | 用途 |
|-----|------|------|------|
| v5 | `https://gitcode.com/api/v5/` | `access_token` 查询参数 | PR 元数据、评论提交、文件列表 |
| v4 | `https://api.gitcode.com/api/v4/` | `PRIVATE-TOKEN` 请求头 | diff_refs 权威基线、discussions 补路径 |

**v4 域名必须是 `api.gitcode.com`**（`gitcode.com/api/v4` 返回 HTML 非 JSON）。

## 端点速查

```text
GET  /api/v5/repos/{owner}/{repo}/pulls/{n}                     PR 元数据（state/target_branch/head.sha）
GET  /api/v5/repos/{owner}/{repo}/pulls/{n}/files               变更文件权威列表（分页 per_page=100）
GET  /api/v5/repos/{owner}/{repo}/pulls/{n}/comments            全部评论（去重用，分页）
POST /api/v5/repos/{owner}/{repo}/pulls/{n}/comments            提交评论（行内 / PR 级）
GET  /api/v4/projects/{owner}%2F{repo}/merge_requests/{n}       diff_refs.{base_sha,head_sha,start_sha}
GET  /api/v4/projects/{owner}%2F{repo}/merge_requests/{n}/discussions  discussions（补评论路径）
```

- `{owner}` 用官方仓 owner（如 `cann`），不用 fork owner——PR 数据挂在官方仓。
- 项目路径 URL 编码：`cann/hccl` → `cann%2Fhccl`。

## 按行提交检视意见（DiffNote）

```json
POST /api/v5/repos/{owner}/{repo}/pulls/{n}/comments
{"body": "评论正文（Markdown）", "commit_id": "<PR head sha>", "path": "src/a.cc", "position": 42}
```

关键语义：

- **`position` 是文件行号**（代码所在行数），不是 diff hunk 内偏移。行必须在 diff 的新增/修改行中，否则返回 400。
- 只有 `commit_id + path + position` 组合产生行内评论（diff_comment）；`path + line + side` 无效（产生 PR 级评论）。
- 行不在 diff 中时回退：只发 `{"body": ...}`，正文首行加 `> file:line` 引用标注目标位置。
- **POST 响应的 `id` 是 discussion_id（hex 字符串）**，不是数字 note_id；核验用 GET comments 列表。
- GET comments 每条含 `comment_type`（`diff_comment`/`pr_comment`）、`diff_position.start_new_line`；**`path` 字段常为 None**，用 v4 discussions 的 `position.new_path` 补齐。

## PR 状态与基线

- **state 检测只信 `state` 字段**：`state == "merged"` 时合入后无法按行提交，改走 Issue 汇总。`merged` 布尔与 `merge_commit_sha` 字段不可靠（实测常为 null）。
- **diff 真值是 v4 `diff_refs`**：`base_sha`/`head_sha` 以 v4 为准；本地 `git merge-base` 可能过期导致 diff 混入 master 演进（基线漂移），文件数与 `changed_files` 不一致时以 v5 `/files` 权威列表为准。
- fork 提交的 PR head：`git fetch <remote> refs/merge-requests/{n}/head:refs/remotes/<remote>/mr/{n}-head --force`（作者 force-push 后须 `--force` 更新本地 ref）。

## 错误码与限速

| HTTP | 含义 | 处置 |
|------|------|------|
| 201 | 创建成功 | GET 核验 comment_type 与行号 |
| 400 | 参数错误 | 检查 position 是否在 diff 新增行、commit_id 是否 head sha |
| 401 | token 无效 | 检查 GITCODE_TOKEN |
| 405 | PR 已合入 | 改走 Issue 汇总 |
| 429 | 限速 | 等 60s 重试 |

- 提交间隔 ≥0.5s（脚本内置，可用 `--delay` 调整）。
- DELETE 评论返回 204 空响应（无 JSON body）。

## 中文内容

脚本用 Python urllib 以 UTF-8 编码请求体并带 `Content-Type: application/json; charset=utf-8`，天然规避 curl 命令行传中文的乱码问题。手动 curl 时必须写文件后 `--data-binary @file.json`，禁止 `-d` 直接内联中文。
