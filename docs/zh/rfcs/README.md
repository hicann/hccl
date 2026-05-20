# RFC 文档目录

本目录存放HCOMM仓的技术方案设计文档（RFC - Request for Comments）。

## RFC 生命周期

1. **编写**：开发者根据 [RFC模板](./0000-template.md) 编写RFC文档
2. **提交PR**：将RFC作为PR提交到本仓库
3. **评审**：由Maintainer进行评审，可能多轮迭代
4. **决策**：
   - **已接受**：RFC被接纳，可以开始实现
   - **已拒绝**：RFC未被接纳，关闭PR
5. **合入**：RFC评审通过后合入，成为实现合约

## 目录结构

- `0000-template.md` - RFC编写模板
- `0001-xxxx-xxx.md` - RFC文档（编号 + 简短描述）

## 命名规范

RFC文件命名格式：`{编号}-{简短描述}.md`

例如：`0001-add-new-feature.md`

## 相关链接

- [贡献指南](../../../CONTRIBUTING.md)
- [SIG例会](https://etherpad-cann.meeting.osinfra.cn/p/sig-hccl)
