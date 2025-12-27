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
 
set -e
 
INPUT_FILE="$1"
OUTPUT_FILE="$2"
OUTPUT_DIR=$(dirname ${OUTPUT_FILE})
 
if [ ! -f "${INPUT_FILE}" ];then
    echo "ERROR: input file '${INPUT_FILE}' not found."
    exit 1
fi
 
if [ -n "${tagInfo}" ];then
    TIME_STAMP_ENV=$(echo "${tagInfo}" | sed -n 's/.*\([0-9]\{8\}_[0-9]\{9\}\).*/\1/p')
fi
 
if [ -n "${TIME_STAMP_ENV}" ];then
    TIME_STAMP=${TIME_STAMP_ENV}
else
    TIME_STAMP=$(date +"%Y%m%d_%H%M%S%3N")
fi
 
if [ ! -d "${OUTPUT_DIR}" ];then
    mkdir -p ${OUTPUT_DIR}
fi
 
cp -f ${INPUT_FILE} ${OUTPUT_FILE}
 
if ! grep -q "^timestamp=" "$OUTPUT_FILE"; then
    if [ -s "$OUTPUT_FILE" ] && [ "$(tail -c 1 "$OUTPUT_FILE")" != "" ]; then
        printf '\n' >> "$OUTPUT_FILE"
    fi
 
    printf 'timestamp=%s\n' "${TIME_STAMP}" >> "$OUTPUT_FILE"
fi
 
exit 0