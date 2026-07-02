# RFC 登记索引

本文件登记所有已分配的 RFC 编号。新增 RFC 前，请先在本表领取**最小未使用编号**。

## 取号流程（独立占号，与 RFC 文档创建解耦）

1. 查看「已分配编号」表，找出最小未使用的 N（通常为最后一行编号 + 1）
2. 在本表追加一行登记 N（状态填 `reserved`，标题/作者可填占位）
3. 提交**取号 PR**（仅含本 INDEX.md 的一行更新）
4. 取号 PR 合入后，编号 N 被占用（状态保持 `reserved`），方可开始撰写 RFC 文档
5. RFC 文档 `NNNN-xxx-xxx.md` 由后续独立 PR 提交
6. RFC 文档 PR 合入后，状态变 `accepted`，整个 RFC 正式生效

## 状态说明

| 状态 | 含义 |
|------|------|
| `reserved` | 编号已占位（取号 PR 已合入），RFC 文档待提交/评审中 |
| `accepted` | RFC 文档 PR 已合入，整个 RFC 正式生效 |
| `superseded` | 被后续 RFC 替代，详见原文档末尾 `Superseded by` 标注 |

## 已分配编号

| 编号 | 标题 | 作者 | 状态 | PR |
|------|------|------|------|-----|
| 0001 | BIRS (Batchsize Invariant ReduceScatter) for A3 | Davydov_Danil | accepted | [#1440](https://gitcode.com/cann/hccl/merge_requests/1440) |

## 编号规则

- **格式**：4 位零填充（0001–9999）
- **永不重用**：已合入的编号即使 RFC 被后续替代也不回收
- **顺序分配**：原则上不跳号，下一个编号 = 最大已用编号 + 1
- **替代关系**：在原 RFC 文档末尾标注 `> Superseded by 00NN`，本表状态列同步改为 `superseded`
- **冲突解决**：若两人同时取同一编号，后者需 rebase 后改为新最小编号

## 相关

- [RFC 模板](./0000-template.md)
- [RFC 流程说明](./README.md)
