# pre-commit Usage Guide

## Overview

`pre-commit` is a Git hooks framework that automatically runs code check and formatting tools during `git commit`. The following checks have been configured for this project.

 | Hook             | Function            | Description                            |
| ---------------- | ---------------- | -------------------------------- |
| **clang-format** | C/C++ code formatting| Automatically formats code to maintain consistent style.    |
| **OAT Check**    | Open-source compliance check    | Checks license headers and prohibits binary file commits.|

## Environment Requirements

- **Git**: 2.0+
- **Python**: 3.8+
- **clang-format**: 14.0+ (code formatting tool)
- **Java**: 17+ (OAT tool dependency, can be installed automatically)
- **Maven**: 3.6+ (OAT tool dependency, can be installed automatically)

## Installation Procedure

### 1. Install `pre-commit`

```bash
# Method 1: Using pip
pip install pre-commit

# Method 2: Using system package manager (Ubuntu or Debian)
sudo apt install pre-commit
```

### 2. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt install clang-format openjdk-17-jre maven

# macOS
brew install clang-format openjdk@17 maven
```

### 3. Install Git Hooks in the Project Directory

```bash
# Go to the repository root directory
cd /path/to/hccl
pre-commit install
```

After the installation, the following information is displayed:

```text
pre-commit installed at .git/hooks/pre-commit
```

## Usage

### Automatic Check (Recommended)

Pre-commit automatically runs checks each time you execute `git commit`:

```bash
git add .
git commit -m "your commit message"
```

Output:

```text
clang-format.............................................................Passed
OAT Compliance Check.....................................................Passed
```

### Manual Check

```bash
# Run all checks
pre-commit run

# Run specific type checks
pre-commit run clang-format
pre-commit run oat-check

# Check all files (not limited to the staging area)
pre-commit run --all-files
```

### Skipping Checks (Emergency)

```bash
git commit --no-verify -m "emergency fix"
```

> **Note**: Use this only in emergencies. During normal development, ensure that all checks pass.

## Check Item Description

### 1. clang-format

Automatically formats C and C++ code according to the [.clang-format](../../../.clang-format) configuration file in the project root directory.

### 2. OAT Compliance Check

OAT (Open Source Audit Tool) checks open-source compliance:

| Check Item | Description |
| -------------- | ------------------------------ |
| License header check | Ensures that source files contain a CANN License header |
| Binary file check | Prevents binary file submissions |
| Archive file check | Prevents submission of archive files such as zip and tar |

During the first run, the OAT check script automatically performs the following actions:

1. Detects or installs Java 17
2. Detects or installs Maven
3. Clones and compiles the tools_oat tool (about 1 to 2 minutes)

## Common Issues

### Q1: The OAT check is slow during the first commit

**Cause**: The first run requires cloning and compiling the OAT tool.

**Solution**: This is expected. Subsequent commits use the cached JAR and are much faster.

## Related Documents

- [Pre-commit official documentation](https://pre-commit.com/)
- [Clang-format configuration](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)
- [OAT tool](https://gitcode.com/openharmony-sig/tools_oat)
- [Repository pre-commit integration guide (Chinese)](https://gitcode.com/cann/infrastructure/blob/main/docs/SC/pre-commit/pre-commit%E9%85%8D%E7%BD%AE%E6%8C%87%E5%AF%BC%E4%B9%A6.md)
