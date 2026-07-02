# RFC 文档目录

本目录存放 HCCL 仓的技术方案设计文档（RFC - Request for Comments），用于在编码实现前对齐方案、沉淀设计决策。

## 目录结构

- `0000-template.md` - RFC 编写模板
- `INDEX.md` - RFC 编号登记表，所有已分配的 RFC 编号一览
- `NNNN-xxx-xxx.md` - RFC 文档（4 位编号 + 简短描述）

## 命名规范

RFC 文件命名格式：`{4位编号}-{简短描述}.md`

例如：`0001-add-new-feature.md`

- 编号：4 位零填充（0001–9999）
- 描述：英文小写、连字符分隔、简洁

## 编号机制（核心）

1. **取号 PR**（轻量）：仅修改 [INDEX.md](./INDEX.md)，追加一行登记占位编号
2. **RFC 文档 PR**（重量）：撰写 RFC 文档并提交评审，编号已通过取号 PR 锁定

**取号规则**：

- 顺序分配，**最小未使用编号**优先
- 永不重用，已合入的编号即使 RFC 被后续替代也不回收
- 详见 [INDEX.md](./INDEX.md)

## RFC 生命周期

1. **需求阶段**：在 Issue 中按 Requirement 类型提交需求，待 SIG 组接纳
2. **取号 PR**：在 [INDEX.md](./INDEX.md) 中追加一行登记占位（状态 `reserved`），提交取号 PR
3. **取号 PR 合入**：编号被占用，可开始撰写 RFC
4. **编写阶段**：按 [RFC 模板](./0000-template.md) 撰写系统方案
5. **评审阶段**：通过 RFC 文档 PR 评审，过程中根据意见修改
6. **决策阶段**：
   - **合入**：Maintainer 评审通过，添加 `/lgtm` 与 `/approve` 后合入，INDEX.md 中状态修改为 `accepted`
   - **关闭**：评审未通过，关闭 PR，编号仍保留 `reserved`，编号不回收（作者可重新走评审）
7. **实施阶段**：合入的 RFC 作为实现合约，代码 PR 需遵循 RFC 方案

## 替代关系

当一个 RFC 的实现被后续 RFC 替代时：

- 在被替代的 RFC 文档末尾追加标注：`> Superseded by 00NN`
- 同步将 [INDEX.md](./INDEX.md) 中该行的状态从 `accepted` 改为 `superseded`
- 不修改原编号

## 相关链接

- [贡献指南](../../../CONTRIBUTING.md)
- [RFC 模板](./0000-template.md)
- [RFC 编号登记表](./INDEX.md)
- [SIG 例会](https://etherpad-cann.meeting.osinfra.cn/p/sig-hccl)
