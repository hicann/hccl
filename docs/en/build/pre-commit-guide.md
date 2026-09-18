# pre-commit Tool Guide

## Overview

pre-commit is a Git Hooks framework that automatically runs code checking and formatting tools during `git commit`. This project is configured with the following checks:

| Hook | Function | Description |
| ---------------- | ---------------- | -------------------------------- |
| **clang-format** | C and C++ code formatting | Automatically formats code to maintain consistent style |
| **OAT Check** | Open source compliance check | Checks license headers and file types |

Both dependencies are managed automatically:

- **clang-format** v18.1.8 is downloaded and managed automatically by pre-commit according to the `rev` field in [.pre-commit-config.yaml](../../../.pre-commit-config.yaml). No manual installation is required.
- The OAT check (Python edition) is also downloaded and installed automatically by pre-commit from the remote repository defined in the same configuration file. No manual installation is required.

## Requirements

- **Git**: 2.0+
- **Python**: 3.8+
- Network access on first run (to fetch the clang-format mirror repo and the OAT check repo)

## Installation Steps

### 1. Install Git and Python

**Linux / macOS**:

```bash
# Ubuntu or Debian
sudo apt install git python3 python3-pip python3-venv

# Fedora or RHEL
sudo dnf install git python3 python3-pip

# openEuler or CentOS
sudo yum install git python3 python3-pip

# macOS (Homebrew)
brew install git python3
```

> **Note**: On Fedora/RHEL and openEuler/CentOS, python3 includes venv but pip (python3-pip) is a separate package, as included in the commands above. On Homebrew, python3 comes with both pip and venv, so no extra installation is needed.

**Windows**:

- Git: download and install from the [Git for Windows](https://git-scm.com/download/win) website.
- Python: run `winget install Python.Python.3.14` in a PowerShell or CMD terminal, or install from the Microsoft Store.

### 2. Install pre-commit

**Linux / macOS**:

On newer systems, the system Python is externally managed (PEP 668), and installing into it with pip is forbidden. Create and activate a virtual environment first. It is recommended to place the virtual environment in the home directory (e.g. `~/.venv`) to avoid polluting the repository, and it can also be shared across repositories:

```bash
python3 -m venv ~/.venv
source ~/.venv/bin/activate
pip install pre-commit
```

> **Note**: It is recommended to add the activation command to the shell configuration file (e.g. `~/.bashrc`, or `~/.zshrc` for zsh) so it is activated automatically on login:

```bash
echo 'source ~/.venv/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

**Windows**:

```bash
py -m pip install pre-commit
```

### 3. Install Git Hooks in the Project Path

**Linux / macOS**:

```bash
# Navigate to the repository root directory
cd /path/to/repo
pre-commit install
```

**Windows**:

```bash
# Navigate to the repository root directory
cd /path/to/repo
py -m pre_commit install
```

A successful installation displays:

```bash
pre-commit installed at .git/hooks/pre-commit
```

> **Note**: There is no separate step for installing dependency tools. The clang-format and OAT check environments are both prepared automatically on the first run.

## Usage

### Automatic Checking (Recommended)

Each time you execute `git commit`, pre-commit automatically runs checks:

```bash
git add .
git commit -m "your commit message"
```

Example output:

```text
clang-format.............................................................Passed
OAT Compliance Check (Python Edition)....................................Passed
```

### Running Checks Manually

**Linux / macOS**:

```bash
# Run all checks (staging area only)
pre-commit run

# Run a specific type of check (staging area only)
pre-commit run clang-format
pre-commit run oat-check

# Run all checks on all files (not limited to the staging area)
pre-commit run --all-files

# Run a specific type of check on all files (not limited to the staging area)
pre-commit run clang-format --all-files
pre-commit run oat-check --all-files
```

**Windows**:

```bash
# Run all checks (staging area only)
py -m pre_commit run

# Run a specific type of check (staging area only)
py -m pre_commit run clang-format
py -m pre_commit run oat-check

# Run all checks on all files (not limited to the staging area)
py -m pre_commit run --all-files

# Run a specific type of check on all files (not limited to the staging area)
py -m pre_commit run clang-format --all-files
py -m pre_commit run oat-check --all-files
```

### Skipping Checks (Emergency)

```bash
git commit --no-verify -m "emergency fix"
```

> **Note**: Use this only in emergencies. During normal development, ensure all checks pass.

## Check Descriptions

### 1. clang-format

Automatically formats C and C++ code according to the [.clang-format](../../../.clang-format) configuration in the project root directory.

### 2. OAT Compliance Check

OAT (OSS Audit Tool, Python edition) checks open source compliance:

| Check Item | Description |
| -------------- | ------------------------------ |
| Invalid File Type | Prevents submission of unauthorized file types such as binaries and archives |
| License Header Invalid | Ensures source files contain a valid CANN License header |

The check environment is created and cached automatically by pre-commit on the first run. No other dependencies are needed.

The check scans the files passed by pre-commit: staged files when committing, or all files when running manually with `--all-files`. When CI triggers the hook multiple times, scan results for the same HEAD are merged into a single cumulative report.

The scan summary is written to `oat_reports/result.txt` in the repository root (this directory is local only and not tracked by git). If issues are found, the commit is blocked.

## License Header Styles

New files must carry the standard CANN-2.0 license header in one of the following three styles, depending on the file type. Use the year the file was added to the repository.

### Style 1: C/C++-style comment

Applies to C/C++ files:

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

### Style 2: Python-style comment

Applies to CMake/Make, Shell, Python, YAML, and similar configuration and script files:

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

### Style 3: XML-style comment

Applies to XML files:

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

### Special Rules

- **Shell scripts (`.sh`)**: Keep the shebang on the first line (e.g., `#!/usr/bin/env bash`); the license header starts from the second line.
- **Python scripts (`.py`)**: The first line is the shebang (e.g., `#!/usr/bin/env python3`), the second line is the coding declaration (`# -*- coding: UTF-8 -*-`), and the license header starts from the third line. Exception: `__init__.py` requires no shebang, so the coding declaration comes first.
- **XML files (`.xml`)**: The first line is the XML declaration (`<?xml version="1.0" encoding="UTF-8"?>`); the license header starts from the second line.
- Keep exactly one blank line after the license header.
- For file types not listed above, choose the appropriate style from the three styles according to the comment syntax of the file.

## Frequently Asked Questions

### Q1: The first commit is slow

**Cause**: The first run needs to download the clang-format mirror repo and the OAT check repo managed by pre-commit.

**Solution**: This is normal. Subsequent runs are much faster.

### Q2: The commit is blocked by the OAT check

**Cause**: Compliance issues were found (invalid file type or missing/invalid license header).

**Solution**: Check the details in `oat_reports/result.txt` in the repository root, fix the reported files (add the correct license header as described above), then re-add and re-commit the files. In emergencies only, skip with `git commit --no-verify`.

### Q3: The OAT check environment fails to install on first run

**Cause**: The network or the code hosting platform is unreachable, so pre-commit cannot fetch the OAT check repo.

**Solution**: Check the network and retry. Once installed, the environment is cached by pre-commit and subsequent runs need no network access.

## Related Documents

- [pre-commit Official Documentation](https://pre-commit.com/)
- [clang-format Configuration](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [OAT Tool](https://gitcode.com/openharmony-sig/tools_oat)
- [pre-commit Integration Guide for Code Repositories (Chinese)](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
