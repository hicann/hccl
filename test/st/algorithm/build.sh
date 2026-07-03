# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

#!/bin/bash
set -e
trap 'echo "❌ Error occurred in build.sh at line $LINENO"; exit 1' ERR

# 获取shell脚本目录作为根目录
SHELL_DIR=$(cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

# 获取CPU核数用于并发编译和执行
CPU_NUM=$(cat /proc/cpuinfo | grep "^processor" | wc -l)

function log() {
    local current_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$current_time] "$1
}

# 创建build编译目录
cd $SHELL_DIR
mkdir -p ./build && cd ./build/ && rm -rf ../build/*

# 配置 ST 用例代码
CMAKE_ARGS="-DBUILD_OPEN_PROJECT=ON"
if [ "${ENABLE_GCOV}" == "on" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GCOV=ON"
fi
if [ -n "${ST_TASKS}" ]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DST_TASKS=${ST_TASKS}"
fi

log "Info: Building st with ${CPU_NUM} parallel jobs"
log "Info: CMAKE_ARGS=${CMAKE_ARGS}"
cmake .. ${CMAKE_ARGS}
if [ $? -ne 0 ]; then
    log "Error: cmake config failed"
    exit 1
fi

# 编译 ST 用例代码
cmake --build . -j ${CPU_NUM}
if [ $? -ne 0 ]; then
    log "Error: cmake build failed"
    exit 1
fi

log "Info: ST build completed successfully!"
