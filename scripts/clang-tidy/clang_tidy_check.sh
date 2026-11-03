#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Clang-Tidy incremental check wrapper. The check always runs and every finding
# is logged; only the build outcome is policy-controlled here. Rollout policy:
# report-only by default - a non-zero verdict from the detector (new-code
# complexity violations, or an interception that cannot complete, e.g. the
# incremental diff cannot be computed) does not fail the build.
# ENABLE_CLANG_TIDY_BLOCK=on restores blocking (the verdict propagates, exit 1);
# ENABLE_CLANG_TIDY_CHECK=off in build.sh skips the check entirely. The
# detector itself (clang_tidy_check.py) is unchanged and stays unit-tested.
# Usage: bash scripts/clang-tidy/clang_tidy_check.sh [--build-dir build] [--base-ref upstream/master]
rc=0
python3 "$(dirname "$0")/clang_tidy_check.py" "$@" || rc=$?
clang_tidy_block="${ENABLE_CLANG_TIDY_BLOCK:-off}"
if [ "$rc" -ne 0 ] && [ "${clang_tidy_block,,}" != "on" ]; then
    echo "[clang-tidy] WARNING: verdict above is report-only and does not fail the build (ENABLE_CLANG_TIDY_BLOCK=${clang_tidy_block}); set ENABLE_CLANG_TIDY_BLOCK=on to block"
    exit 0
fi
exit "$rc"
