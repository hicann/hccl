# hccl 仓内 AI Skill 简介

hccl 仓内置两个 `.agents/skills/` 下的 skill。opencode、codex、deepseek-harness 等 agent 工具直接把代码仓目录设为项目目录即可使用——对 AI 说一句话自动执行全流程。

## 贡献流程 skill（`.agents/skills/hccl-contribute/`）

对 AI 说「贡献 issue #123 修复」「提交 PR」「CI 修复」「处理检视意见」「本地编译并跑 UT」等指令，自动执行全流程。支持人工与 AI 触发；Issue/PR 自动查重，不会重复立项或提交重复 PR。

主要功能：自动同步代码（脏工作区自动 worktree 隔离），按 AGENTS.md 本地构建与测试，Issue 查重与创建，fork 提交 PR 并自动评论 /compile 触发 CI，轮询 CI 状态（saw_running 防旧标签误判），CI 失败日志获取，未处理检视意见列表，现场清理。

效果：贡献一个 PR 只需一句话，免手工同步、查重、触发 CI、轮询；检视意见修复后配合 review skill 的处置能力回复关闭，形成检视闭环。

## PR 检视 skill（`.agents/skills/hccl-review/`）

对 AI 说「检视 PR123」自动执行全流程。CI 门禁触发和人工触发都支持，不会提交重复或相似的检视意见。

主要功能：自动拉取 PR 到隔离 worktree，按仓内规范多维检视，检视意见按行提交到 GitCode 正确代码行（自动行号验证、与已有评论去重），最后发汇总报告并清理现场。

效果：免手工比对行号与逐条发评论，检视一个 PR 只需一句话；检视意见修复后还能回复关闭提交的检视意见，形成检视闭环。

## 必要 harness（两个 skill 通用结构）

| 路径 | 说明 |
|------|------|
| `SKILL.md` | 通用工作流（contribute 八步 / review 五步），构建命令动态取自仓内 AGENTS.md，无双源漂移 |
| `scripts/` | 机械步骤固化脚本（环境校验/同步/Issue/PR/CI/清理；行号验证/评论提交/去重） |
| `references/` | 按需加载的规范与 API 文档（gitcode-api.md、ci-triage.md；检视规范五类分档见 review skill） |
