# pre-commit工具使用指导

## 概述

pre-commit是一个Git Hooks框架，用于在`git commit`时自动运行代码检查和格式化工具。本项目已配置以下检查：

| Hook             | 功能             | 说明                       |
| ---------------- | ---------------- | -------------------------- |
| **clang-format** | C/C++代码格式化 | 自动格式化代码，保持风格一致 |
| **OAT Check**    | 开源合规检查     | 检查许可证头与文件类型     |

两项依赖均为自动管理：

- **clang-format** v18.1.8由pre-commit按 [.pre-commit-config.yaml](../../../.pre-commit-config.yaml) 中的`rev`自动下载管理，无需手动安装。
- OAT检查（Python版）同样由pre-commit按该配置中的远程仓自动下载安装，无需手动安装。

## 环境要求

- **Git**：2.0+
- **Python**：3.8+
- 首次运行需网络（拉取clang-format镜像仓与OAT检查仓）

## 安装步骤

### 1. 安装Git与Python

**Linux / macOS**：

```bash
# Ubuntu/Debian
sudo apt install git python3 python3-pip python3-venv

# Fedora/RHEL
sudo dnf install git python3 python3-pip

# openEuler/CentOS
sudo yum install git python3 python3-pip

# macOS (Homebrew)
brew install git python3
```

> **注意**：Fedora/RHEL、openEuler/CentOS的python3已内置venv，但pip（python3-pip）为独立包，上述命令已包含；Homebrew的python3已自带pip与venv，无需额外安装。

**Windows**：

- Git：从 [Git for Windows](https://git-scm.com/download/win) 官网下载安装。
- Python：在PowerShell或CMD终端中执行`winget install Python.Python.3.14`，或通过Microsoft Store安装。

### 2. 安装pre-commit

**Linux / macOS**：

较新系统的Python受PEP 668保护（externally-managed），禁止直接使用pip安装到系统环境，需先创建并激活虚拟环境。虚拟环境建议放在用户主目录下（如`~/.venv`），避免污染代码仓，也可被多个代码仓共用：

```bash
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install pre-commit
```

> **注意**：建议将激活命令写入shell配置文件（如`~/.bashrc`，zsh为`~/.zshrc`），登录后自动激活，无需每次手动执行：

```bash
echo 'source ~/.venv/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

**Windows**：

```bash
py -m pip install pre-commit
```

### 3. 项目路径下安装Git Hooks

**Linux / macOS**：

```bash
# 进入代码仓根目录
cd /path/to/repo
pre-commit install
```

**Windows**：

```bash
# 进入代码仓根目录
cd /path/to/repo
py -m pre_commit install
```

安装成功后会显示：

```bash
pre-commit installed at .git/hooks/pre-commit
```

> **注意**：无需单独安装依赖工具，clang-format与OAT检查环境均在首次运行时自动准备。

## 使用方法

### 自动检查（推荐）

每次执行`git commit`时，pre-commit会自动运行检查：

```bash
git add .
git commit -m "your commit message"
```

输出示例：

```text
clang-format.............................................................Passed
OAT Compliance Check (Python Edition)....................................Passed
```

### 手动运行检查

**Linux / macOS**：

```bash
# 运行所有检查（限于暂存区）
pre-commit run

# 运行特定类型检查（限于暂存区）
pre-commit run clang-format
pre-commit run oat-check

# 运行所有文件的所有检查（不限于暂存区）
pre-commit run --all-files

# 运行所有文件的特定类型检查（不限于暂存区）
pre-commit run clang-format --all-files
pre-commit run oat-check --all-files
```

**Windows**：

```bash
# 运行所有检查（限于暂存区）
py -m pre_commit run

# 运行特定类型检查（限于暂存区）
py -m pre_commit run clang-format
py -m pre_commit run oat-check

# 运行所有文件的所有检查（不限于暂存区）
py -m pre_commit run --all-files

# 运行所有文件的特定类型检查（不限于暂存区）
py -m pre_commit run clang-format --all-files
py -m pre_commit run oat-check --all-files
```

### 跳过检查（紧急情况）

```bash
git commit --no-verify -m "emergency fix"
```

> **注意**：仅在紧急情况下使用，正常开发流程应保证检查通过。

## 检查项说明

### 1. clang-format

自动格式化C/C++代码，遵循项目根目录下 [.clang-format](../../../.clang-format) 配置。

### 2. OAT Compliance Check

OAT（OSS Audit Tool，Python版）检查开源合规性：

| 检查项                 | 说明                                       |
| ---------------------- | ------------------------------------------ |
| Invalid File Type      | 禁止提交二进制、归档文件等非法文件类型     |
| License Header Invalid | 确保源文件包含有效的CANN License头         |

检查环境由pre-commit首次运行时自动创建并缓存，无需其他依赖。

检查扫描pre-commit传入的文件：提交时为暂存文件，手动以`--all-files`运行时为所有文件。CI多次触发hook时，同一HEAD的扫描结果会合并为一份累计报告。

扫描结果摘要写入代码仓根目录`oat_reports/result.txt`（该目录仅本地生成，不纳入git管理）。发现问题时提交会被阻塞。

## License头三种风格

新增文件必须按文件类型选择以下三种风格之一添加标准CANN-2.0 License头，年份取文件入仓年份。

### 风格一：C/C++风格注释

适用于C/C++文件：

```c
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
```

### 风格二：Python风格注释

适用于CMake/Make、Shell、Python、YAML等配置与脚本文件：

```bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
```

### 风格三：XML风格注释

适用于XML文件：

```xml
<!--
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
-->
```

### 特殊规则

- **Shell脚本（`.sh`）**：首行为shebang（如`#!/usr/bin/env bash`），第二行起为License头。
- **Python脚本（`.py`）**：首行为shebang（如`#!/usr/bin/env python3`），第二行为coding声明（`# -*- coding: UTF-8 -*-`），第三行起为License头。例外：`__init__.py`不需要shebang，首行直接为coding声明。
- **XML文件（`.xml`）**：首行为XML声明（`<?xml version="1.0" encoding="UTF-8"?>`），第二行起为License头。
- License头后保留且仅保留一个空行。
- 未列出的文件类型，应根据其注释语法从上述三种风格中选择合适的一种。

## 常见问题

### Q1: 首次提交时检查很慢

**原因**：首次运行需要下载pre-commit管理的clang-format镜像仓与OAT检查仓。

**解决**：这是正常现象，后续运行速度会很快。

### Q2: 提交被OAT检查阻塞

**原因**：发现合规问题（非法文件类型或License头缺失/非法）。

**解决**：查看代码仓根目录`oat_reports/result.txt`中的详情，按上述头风格修复对应文件后重新`git add`并提交。仅紧急情况可使用`git commit --no-verify`跳过。

### Q3: 首次运行OAT检查环境安装失败

**原因**：无法访问网络或代码托管平台，pre-commit未能拉取OAT检查仓。

**解决**：检查网络后重试。环境安装成功后会被pre-commit缓存，后续运行无需网络。

## 相关文档

- [pre-commit官方文档](https://pre-commit.com/)
- [clang-format配置](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [OAT工具](https://gitcode.com/openharmony-sig/tools_oat)
- [代码仓集成pre-commit指导](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
