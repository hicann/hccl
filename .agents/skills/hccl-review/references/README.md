# 检视规范索引

检视规范按五类组织，检视时按需加载对应文档（非流程；流程见 [`SKILL.md`](../SKILL.md)）。仓内编码/架构规范演进时以根 `AGENTS.md` 与其链接的权威文档为准，本文档体系与之保持同步。

| 文档 | 覆盖 | 何时加载 |
|------|------|---------|
| [`coding-and-security.md`](./coding-and-security.md) | 命名风格、内存/资源/并发/错误处理红线、工具验证纪律 | 所有代码 PR 必读 |
| [`external-api.md`](./external-api.md) | 对外头文件（include/、pkg_inc/）C 接口、ABI、命名一致性、兼容性、模块变更 | PR 触碰 include/、pkg_inc/ 或新增类/文件/目录时 |
| [`architecture.md`](./architecture.md) | 分层依赖、控制面/数据面分离、仓间解耦、legacy 约束、高风险区 | PR 触碰 src/ 时 |
| [`pr-completeness.md`](./pr-completeness.md) | PR/Issue/实现三者吻合度、测试完备性、文档同步 | 所有 PR |
| [`gitcode-api.md`](./gitcode-api.md) | GitCode API 端点、认证、position 语义（检视工具链本身） | 提交检视意见时 |
