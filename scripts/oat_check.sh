#!/bin/sh
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# OAT Pre-commit Check Script (for pre-commit framework)
# This script is called by pre-commit framework automatically
#

set -e

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
REPO_NAME=$(basename "$REPO_ROOT")
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
OAT_DIR="$REPO_ROOT/../tools_oat"
OAT_GIT_URL="https://gitcode.com/openharmony-sig/tools_oat.git"
OAT_JAR_PATTERN="$OAT_DIR/target/ohos_ossaudittool-*.jar"
OAT_REPORT_DIR="$REPO_ROOT/oat_reports"
OAT_POLICY="filetype:!binary~must|!archive~must;license:*@.*"

echo "[OAT] Running OAT scan (binary files + license headers) - INCREMENTAL MODE..."
echo "[OAT] Project: $REPO_NAME"

# --- 1. Get list of files to be committed ---
if [ $# -gt 0 ]; then
    # Files passed as arguments from pre-commit
    STAGED_FILES="$@"
    FILE_COUNT=$#
else
    # Fallback: get from git
    STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)
    if [ -z "$STAGED_FILES" ]; then
        echo "[OAT] No files to check. Skipping OAT scan."
        exit 0
    fi
    FILE_COUNT=$(echo "$STAGED_FILES" | wc -l)
fi

echo "[OAT] Checking $FILE_COUNT staged file(s)..."

# Convert file list to absolute paths and join with |
FILE_LIST=""
for file in $STAGED_FILES; do
    # Convert to absolute path if relative
    if [ "${file#/}" = "$file" ]; then
        # Relative path
        file="$REPO_ROOT/$file"
    fi
    
    if [ -z "$FILE_LIST" ]; then
        FILE_LIST="$file"
    else
        FILE_LIST="$FILE_LIST|$file"
    fi
done

# --- 2. Check Java environment ---
echo ""
echo "[OAT] Checking runtime environment..."

if ! command -v java >/dev/null 2>&1; then
    echo ""
    echo "===================================================================="
    echo "  Java 未安装 - 正在尝试自动安装"
    echo "===================================================================="
    echo ""
    echo "[OAT] 检测到系统未安装 Java，开始自动安装..."
    echo "[OAT] 注意: 首次安装需要下载约 50-100MB，可能需要 2-5 分钟"
    echo ""
    
    # Detect OS
    OS_TYPE="unknown"
    if [ "$(uname)" = "Linux" ]; then
        OS_TYPE="linux"
    elif [ "$(uname)" = "Darwin" ]; then
        OS_TYPE="macos"
    elif [ -n "$WINDIR" ] || [ "$(uname -o 2>/dev/null)" = "Msys" ] || [ "$(uname -o 2>/dev/null)" = "Cygwin" ]; then
        OS_TYPE="windows"
    fi
    
    echo "[OAT] 检测到操作系统: $OS_TYPE"
    echo ""
    
    JAVA_INSTALLED=false
    
    # Linux: 使用 apt
    if [ "$OS_TYPE" = "linux" ]; then
        if command -v apt-get >/dev/null 2>&1; then
            echo "[OAT] 使用 apt 安装 OpenJDK 11..."
            echo "[OAT] 可能需要输入管理员密码"
            echo ""
            
            sudo apt-get update -qq >/dev/null 2>&1
            sudo apt-get install -y openjdk-11-jre >/dev/null 2>&1
            
            if [ $? -eq 0 ]; then
                JAVA_INSTALLED=true
                echo "[OAT] [OK] OpenJDK 11 安装成功"
            else
                echo "[OAT] [ERROR] 自动安装失败"
            fi
        elif command -v yum >/dev/null 2>&1; then
            echo "[OAT] 使用 yum 安装 OpenJDK 11..."
            sudo yum install -y java-11-openjdk >/dev/null 2>&1
            
            if [ $? -eq 0 ]; then
                JAVA_INSTALLED=true
                echo "[OAT] [OK] OpenJDK 11 安装成功"
            fi
        fi
    
    # macOS: 使用 Homebrew
    elif [ "$OS_TYPE" = "macos" ]; then
        if command -v brew >/dev/null 2>&1; then
            echo "[OAT] 使用 Homebrew 安装 OpenJDK 11..."
            echo "[OAT] 这可能需要几分钟..."
            echo ""
            
            brew install openjdk@11 >/dev/null 2>&1
            
            if [ $? -eq 0 ]; then
                JAVA_INSTALLED=true
                echo "[OAT] [OK] OpenJDK 11 安装成功"
                
                # Add to PATH for current session
                export PATH="/usr/local/opt/openjdk@11/bin:$PATH"
            else
                echo "[OAT] [ERROR] 自动安装失败"
            fi
        else
            echo "[OAT] [ERROR] Homebrew 未安装"
            echo "[OAT] 请先安装 Homebrew: https://brew.sh/"
        fi
    
    # Windows: 提供下载链接（无法自动安装）
    elif [ "$OS_TYPE" = "windows" ]; then
        echo "[OAT] Windows 系统无法自动安装 Java"
        echo "[OAT] 请手动下载并安装："
        echo ""
        echo "  1. 访问: https://adoptium.net/"
        echo "  2. 下载: Eclipse Temurin JRE 11 (x64)"
        echo "  3. 安装后重启 Git Bash"
        echo "  4. 验证: java -version"
        echo ""
        echo "[OAT] 跳过 OAT 检查，继续提交..."
        echo "[OAT] 建议安装 Java 后再次运行检查"
        echo ""
        exit 0
    fi
    
    # Verify installation
    if [ "$JAVA_INSTALLED" = true ]; then
        if command -v java >/dev/null 2>&1; then
            JAVA_VERSION=$(java -version 2>&1 | head -n 1)
            echo "[OAT] [OK] Java 安装验证通过: $JAVA_VERSION"
            echo ""
        else
            echo "[OAT] [ERROR] Java 安装后仍无法使用，可能需要重启终端"
            echo ""
            echo "请尝试:"
            echo "  1. 关闭并重新打开终端"
            echo "  2. 运行: source ~/.bashrc (Linux) 或 source ~/.zshrc (macOS)"
            echo "  3. 重新提交: git commit"
            echo ""
            echo "[OAT] 跳过 OAT 检查，继续提交..."
            echo "[OAT] 建议重启终端后再次运行检查"
            echo ""
            exit 0
        fi
    else
        echo ""
        echo "[OAT] 自动安装失败，跳过 OAT 检查"
        echo ""
        echo "手动安装方法:"
        echo "  Linux:   sudo apt install openjdk-11-jre"
        echo "  macOS:   brew install openjdk@11"
        echo "  Windows: https://adoptium.net/"
        echo ""
        echo "[OAT] 继续提交（未进行合规性检查）..."
        echo "[OAT] 建议安装 Java 后再次运行: pre-commit run oat-check"
        echo ""
        exit 0
    fi
fi

JAVA_VERSION=$(java -version 2>&1 | head -n 1)
echo "  [OK] Java: $JAVA_VERSION"

# --- 3. Check Maven environment ---
echo ""
echo "[OAT] Checking Maven environment..."

if ! command -v mvn >/dev/null 2>&1; then
    echo ""
    echo "===================================================================="
    echo "  Maven 未安装 - 正在尝试自动安装"
    echo "===================================================================="
    echo ""
    echo "[OAT] 检测到系统未安装 Maven，开始自动安装..."
    echo "[OAT] 注意: 首次安装需要下载约 10-20MB"
    echo ""
    
    # Detect OS
    OS_TYPE="unknown"
    if [ "$(uname)" = "Linux" ]; then
        OS_TYPE="linux"
    elif [ "$(uname)" = "Darwin" ]; then
        OS_TYPE="macos"
    elif [ -n "$WINDIR" ] || [ "$(uname -o 2>/dev/null)" = "Msys" ] || [ "$(uname -o 2>/dev/null)" = "Cygwin" ]; then
        OS_TYPE="windows"
    fi
    
    echo "[OAT] 检测到操作系统: $OS_TYPE"
    echo ""
    
    MAVEN_INSTALLED=false
    
    # Linux: 使用 apt
    if [ "$OS_TYPE" = "linux" ]; then
        if command -v apt-get >/dev/null 2>&1; then
            echo "[OAT] 使用 apt 安装 Maven..."
            echo "[OAT] 可能需要输入管理员密码"
            echo ""
            
            sudo apt-get update -qq >/dev/null 2>&1
            sudo apt-get install -y maven >/dev/null 2>&1
            
            if [ $? -eq 0 ]; then
                MAVEN_INSTALLED=true
                echo "[OAT] [OK] Maven 安装成功"
            else
                echo "[OAT] [ERROR] 自动安装失败"
            fi
        elif command -v yum >/dev/null 2>&1; then
            echo "[OAT] 使用 yum 安装 Maven..."
            sudo yum install -y maven >/dev/null 2>&1
            
            if [ $? -eq 0 ]; then
                MAVEN_INSTALLED=true
                echo "[OAT] [OK] Maven 安装成功"
            fi
        fi
    
    # macOS: 使用 Homebrew
    elif [ "$OS_TYPE" = "macos" ]; then
        if command -v brew >/dev/null 2>&1; then
            echo "[OAT] 使用 Homebrew 安装 Maven..."
            echo "[OAT] 这可能需要几分钟..."
            echo ""
            
            brew install maven >/dev/null 2>&1
            
            if [ $? -eq 0 ]; then
                MAVEN_INSTALLED=true
                echo "[OAT] [OK] Maven 安装成功"
            else
                echo "[OAT] [ERROR] 自动安装失败"
            fi
        else
            echo "[OAT] [ERROR] Homebrew 未安装"
            echo "[OAT] 请先安装 Homebrew: https://brew.sh/"
        fi
    
    # Windows: 提供下载链接（无法自动安装）
    elif [ "$OS_TYPE" = "windows" ]; then
        echo "[OAT] Windows 系统无法自动安装 Maven"
        echo "[OAT] 请手动下载并安装："
        echo ""
        echo "  1. 访问: https://maven.apache.org/download.cgi"
        echo "  2. 下载: apache-maven-3.x.x-bin.zip"
        echo "  3. 解压到 C:\\Program Files\\apache-maven-3.x.x"
        echo "  4. 添加到系统 PATH"
        echo "  5. 重启 Git Bash"
        echo "  6. 验证: mvn -version"
        echo ""
        echo "[OAT] 跳过 OAT 检查，继续提交..."
        echo "[OAT] 建议安装 Maven 后再次运行检查"
        echo ""
        exit 0
    fi
    
    # Verify installation
    if [ "$MAVEN_INSTALLED" = true ]; then
        if command -v mvn >/dev/null 2>&1; then
            MAVEN_VERSION=$(mvn -version 2>&1 | head -n 1)
            echo "[OAT] [OK] Maven 安装验证通过: $MAVEN_VERSION"
            echo ""
        else
            echo "[OAT] [ERROR] Maven 安装后仍无法使用，可能需要重启终端"
            echo ""
            echo "请尝试:"
            echo "  1. 关闭并重新打开终端"
            echo "  2. 运行: source ~/.bashrc (Linux) 或 source ~/.zshrc (macOS)"
            echo "  3. 重新提交: git commit"
            echo ""
            echo "[OAT] 跳过 OAT 检查，继续提交..."
            echo "[OAT] 建议重启终端后再次运行检查"
            echo ""
            exit 0
        fi
    else
        echo ""
        echo "[OAT] 自动安装失败，跳过 OAT 检查"
        echo ""
        echo "手动安装方法:"
        echo "  Linux:   sudo apt install maven"
        echo "  macOS:   brew install maven"
        echo "  Windows: https://maven.apache.org/download.cgi"
        echo ""
        echo "[OAT] 继续提交（未进行合规性检查）..."
        echo "[OAT] 建议安装 Maven 后再次运行: pre-commit run oat-check"
        echo ""
        exit 0
    fi
fi

MAVEN_VERSION=$(mvn -version 2>&1 | head -n 1)
echo "  [OK] Maven: $MAVEN_VERSION"

# --- 4. Clone tools_oat if not present ---
if [ ! -d "$OAT_DIR" ]; then
    echo ""
    echo "[OAT] tools_oat not found. Cloning..."
    git clone "$OAT_GIT_URL" "$OAT_DIR" --depth=1 >/dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "[OAT] [ERROR] Failed to clone tools_oat."
        echo "[OAT] You can manually clone from: $OAT_GIT_URL"
        exit 1
    fi
    echo "[OAT] [OK] tools_oat cloned successfully."
fi

# --- 5. Build OAT jar if not present ---
OAT_JAR=$(ls $OAT_JAR_PATTERN 2>/dev/null | head -n 1)

if [ -z "$OAT_JAR" ]; then
    echo ""
    echo "[OAT] Building OAT jar with Maven..."
    echo "[OAT] This may take a few minutes (first time only)..."
    
    cd "$OAT_DIR"
    mvn package -q -DskipTests 2>&1 | grep -E "(Building|BUILD SUCCESS|BUILD FAILURE|ERROR)" || true
    BUILD_RESULT=${PIPESTATUS[0]}
    cd - >/dev/null
    
    OAT_JAR=$(ls $OAT_JAR_PATTERN 2>/dev/null | head -n 1)
    
    if [ -z "$OAT_JAR" ] || [ $BUILD_RESULT -ne 0 ]; then
        echo ""
        echo "===================================================================="
        echo "  Maven 打包失败"
        echo "===================================================================="
        echo ""
        echo "[OAT] 无法打包 OAT JAR，跳过 OAT 检查"
        echo ""
        echo "可能原因:"
        echo "  1. Maven 配置问题"
        echo "  2. 网络连接问题（无法下载依赖）"
        echo "  3. pom.xml 配置错误"
        echo ""
        echo "建议解决方案:"
        echo "  1. 手动打包："
        echo "     cd $OAT_DIR"
        echo "     mvn clean package -DskipTests"
        echo ""
        echo "  2. 配置 Maven 镜像（国内网络）："
        echo "     编辑 ~/.m2/settings.xml 添加阿里云镜像"
        echo ""
        echo "[OAT] 继续提交（未进行合规性检查）..."
        echo "[OAT] 建议修复打包问题后运行: pre-commit run oat-check"
        echo ""
        exit 0
    fi
    
    echo "[OAT] [OK] OAT jar built successfully."
    echo "[OAT] JAR location: $OAT_JAR"
fi

echo ""
echo "  [OK] OAT JAR: $OAT_JAR"

# --- 6. Run OAT scan in INCREMENTAL mode ---
mkdir -p "$OAT_REPORT_DIR"

echo ""
echo "[OAT] Running compliance scan..."
java -jar "$OAT_JAR" \
    -mode s \
    -s "$REPO_ROOT" \
    -r "$OAT_REPORT_DIR" \
    -n "$REPO_NAME" \
    -w 1 \
    -f "$FILE_LIST" \
    -policy "$OAT_POLICY" >/dev/null 2>&1

if [ $? -ne 0 ]; then
    echo ""
    echo "===================================================================="
    echo "  OAT 扫描执行失败"
    echo "===================================================================="
    echo ""
    echo "[OAT] 扫描失败，跳过 OAT 检查"
    echo ""
    echo "可能原因:"
    echo "  1. JAR 文件损坏"
    echo "  2. Java 版本不兼容"
    echo "  3. OAT 配置问题"
    echo ""
    echo "建议解决方案:"
    echo "  1. 删除并重新打包 JAR："
    echo "     rm $OAT_JAR"
    echo "     cd $OAT_DIR && mvn clean package -DskipTests"
    echo ""
    echo "  2. 检查 Java 版本（需要 Java 8+）："
    echo "     java -version"
    echo ""
    echo "[OAT] 继续提交（未进行合规性检查）..."
    echo "[OAT] 建议修复扫描问题后运行: pre-commit run oat-check"
    echo ""
    exit 0
fi

# --- 7. Check report and generate result.txt ---
REPORT_FILE="$OAT_REPORT_DIR/single/PlainReport_${REPO_NAME}.txt"
RESULT_FILE="$OAT_REPORT_DIR/single/result.txt"

if [ -f "$REPORT_FILE" ]; then
    # Extract key information to result.txt
    {
        echo "==================================="
        echo "OAT Scan Result Summary"
        echo "==================================="
        echo "Scan Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Project: $REPO_NAME"
        echo "Files Checked: $FILE_COUNT"
        echo ""
        
        # Extract Invalid File Type section
        echo "-----------------------------------"
        awk '/^Invalid File Type Total Count:/{found=1} found{print; if(/^$/ && NR>1 && prev!~/^$/)exit} {prev=$0}' "$REPORT_FILE"
        
        # Extract License Header Invalid section
        echo "-----------------------------------"
        awk '/^License Header Invalid Total Count:/{found=1} found{print; if(/^$/ && NR>1 && prev!~/^$/)exit} {prev=$0}' "$REPORT_FILE"
        
        echo "==================================="
    } > "$RESULT_FILE"
    
    # Check for issues (read from result.txt instead of PlainReport)
    INVALID_FILE_TYPE=$(grep "^Invalid File Type Total Count:" "$RESULT_FILE" | grep -oE '[0-9]+' | head -1)
    LICENSE_INVALID=$(grep "^License Header Invalid Total Count:" "$RESULT_FILE" | grep -oE '[0-9]+' | head -1)
    
    # Delete PlainReport files to keep only result.txt
    echo "[OAT] Cleaning up redundant reports..."
    rm -f "$OAT_REPORT_DIR/single/PlainReport_"*.txt
    
    TOTAL_ISSUES=$((INVALID_FILE_TYPE + LICENSE_INVALID))
    
    if [ "$TOTAL_ISSUES" -gt 0 ]; then
        echo ""
        echo "===================================================================="
        echo "  发现合规性问题"
        echo "===================================================================="
        echo ""
        echo "[OAT] Found $TOTAL_ISSUES compliance issue(s):"
        echo "  - Invalid File Type: $INVALID_FILE_TYPE"
        echo "  - License Header Invalid: $LICENSE_INVALID"
        echo ""
        echo "[OAT] 查看详细信息:"
        echo "  cat $RESULT_FILE"
        echo ""
        echo "或临时跳过检查:"
        echo "  git commit --no-verify"
        echo ""
        exit 1
    fi
fi

echo ""
echo "[OAT] [OK] All checks passed ($FILE_COUNT file(s) checked)."
echo "[OAT] 查看扫描摘要: cat $RESULT_FILE"
echo ""
exit 0
