# pre-commit工具使用指导

## 概述

pre-commit是一个Git Hooks框架，用于在 `git commit` 时自动运行代码检查和格式化工具。本项目已配置以下检查：

| Hook             | 功能             | 说明                             |
| ---------------- | ---------------- | -------------------------------- |
| **clang-format** | C/C++ 代码格式化 | 自动格式化代码，保持风格一致     |
| **OAT Check**    | 开源合规检查     | 检测许可证头、禁止二进制文件提交 |

## 环境要求

- **Git**: 2.0+
- **Python**: 3.8+
- **clang-format**: 14.0+ (代码格式化工具)
- **Java**: 17+ (OAT工具依赖，可自动安装)
- **Maven**: 3.6+ (OAT工具依赖，可自动安装)

## 安装步骤

### 1. 安装pre-commit

```bash
# 方式一: 使用pip
pip install pre-commit

# 方式二: 使用系统包管理器 (Ubuntu/Debian)
sudo apt install pre-commit
```

### 2. 安装依赖工具

```bash
# Ubuntu/Debian
sudo apt install clang-format openjdk-17-jre maven

# macOS
brew install clang-format openjdk@17 maven
```

### 3. 项目路径下安装Git Hooks

```bash
# 进入代码仓根目录
cd /path/to/hccl
pre-commit install
```

安装成功后会显示：

```text
pre-commit installed at .git/hooks/pre-commit
```

## 使用方法

### 自动检查（推荐）

每次执行 `git commit` 时，pre-commit会自动运行检查：

```bash
git add .
git commit -m "your commit message"
```

输出示例：

```text
clang-format.............................................................Passed
OAT Compliance Check.....................................................Passed
```

### 手动运行检查

```bash
# 运行所有检查
pre-commit run

# 运行特定类型检查
pre-commit run clang-format
pre-commit run oat-check

# 检查所有文件（不限于暂存区）
pre-commit run --all-files
```

### 跳过检查（紧急情况）

```bash
git commit --no-verify -m "emergency fix"
```

> **注意**: 仅在紧急情况下使用，正常开发流程应保证检查通过。

## 检查项说明

### 1. clang-format

自动格式化C/C++ 代码，遵循项目根目录下 [.clang-format](../../../.clang-format) 配置：

### 2. OAT Compliance Check

OAT (Open Source Audit Tool) 检查开源合规性：

| 检查项         | 说明                           |
| -------------- | ------------------------------ |
| 许可证头检查   | 确保源文件包含CANN License头 |
| 二进制文件检查 | 禁止提交二进制文件             |
| 归档文件检查   | 禁止提交zip/tar等归档文件    |

OAT检查脚本，首次运行时会自动：

1. 检测/安装Java 17
2. 检测/安装Maven
3. 克隆并编译tools_oat工具（约1-2分钟）

## 常见问题

### Q1: 首次提交时OAT检查很慢

**原因**: 首次运行需要克隆并编译OAT工具。

**解决**: 这是正常现象，后续提交会使用缓存的JAR，速度会很快。

## 相关文档

- [pre-commit官方文档](https://pre-commit.com/)
- [clang-format配置](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [OAT工具](https://gitcode.com/openharmony-sig/tools_oat)
- [代码仓集成pre-commit指导](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
