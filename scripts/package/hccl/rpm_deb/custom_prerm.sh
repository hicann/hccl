#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

unset PYTHONPATH
export PIP_BREAK_SYSTEM_PACKAGES=1

run_pip()
{
    if python3 -m pip --version >/dev/null 2>&1; then
        python3 -m pip "$@"
    elif command -v pip3 >/dev/null 2>&1; then
        pip3 "$@"
    else
        return 127
    fi
}

if run_pip uninstall -y es_hccl >/dev/null 2>&1; then
    echo "[hccl] es_hccl uninstalled successfully"
else
    echo "[hccl] es_hccl is not installed or uninstall failed, skip"
fi

# 仅删除空目录，存在其他组件文件时不会误删
rmdir "${INSTALL_PATH}/python/site-packages" 2>/dev/null || true
rmdir "${INSTALL_PATH}/python" 2>/dev/null || true

rmdir "${INSTALL_PATH}/opp/built-in/op_impl/aicpu/kernel" 2>/dev/null || true

unset PIP_BREAK_SYSTEM_PACKAGES