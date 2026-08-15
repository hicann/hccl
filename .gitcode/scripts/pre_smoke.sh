#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

echo "start run test case, please wait ..."
cd ${WORKSPACE}

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0
source /usr/local/Ascend/cann/set_env.sh

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."

# ==============================
# 运行测试主循环
# ==============================

source /usr/local/Ascend/cann/set_env.sh
arm_package=$(basename "${arm_run_url}")
wget -nv ${arm_run_url}
# Add execute permission to the downloaded package
echo "Adding execute permission: chmod +x ${arm_package}"
chmod +x "${arm_package}" || echo "Failed to add execute permission to the package"
echo "y" | bash "${arm_package}" --full --install-path=/usr/local/Ascend --quiet
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --cb_test_verify 2>&1 | tee -a ./run_test.log

# ==============================
# 打包log
# ==============================
set +e
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

set -e

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=`date +%Y%m%d`"."`date +%H%M%S`
if grep -E '\b(FAIL|errors|fail|failed|error|ERROR:|Error|error:)\b' "./run_test.log" | grep -v "error)"; then
    echo "$date_time : run test case failed"
    exit 1
else
    echo "$date_time : run test case success"
    exit 0
fi
