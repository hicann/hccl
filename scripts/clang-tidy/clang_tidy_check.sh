#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Clang-Tidy incremental check wrapper: block only complexity warnings on newly added lines.
# Skips (exit 0) only when the gate cannot even start: clang-tidy not installed,
# compile_commands.json missing, or base ref unresolvable on a local box (CI
# blocks on an unresolvable base ref). An incremental diff that cannot be
# computed (shallow checkout beyond healing) blocks (exit 1) in CI and warns
# + skips locally; escape hatch: ENABLE_CLANG_TIDY_CHECK=off.
# Usage: bash scripts/clang-tidy/clang_tidy_check.sh [--build-dir build] [--base-ref upstream/master]
python3 "$(dirname "$0")/clang_tidy_check.py" "$@"
