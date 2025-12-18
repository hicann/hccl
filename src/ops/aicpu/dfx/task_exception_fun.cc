/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dfx/task_exception_fun.h"

#include <string>
#include <sstream>
#include <memory>
#include "log.h"
#include "alg_param.h"

namespace ops_hccl {

void GetScatterOpInfo(const void *opInfo, char *outPut, size_t size)
{
    const OpParam *param = reinterpret_cast<const OpParam *>(opInfo);
    std::stringstream ss;
    ss << "tag:" << param->algTag << ", ";
    ss << "group:" << param->commName << ", ";
    ss << "count:" << param->DataDes.count << ", ";
    ss << "dataType:" << param->DataDes.dataType << ", ";
    ss << "opType:" << param->opType << ", ";
    ss << "rootId:" << param->root << ", ";
    ss << "dstAddr:0x" << std::hex << param->inputPtr << ", ";
    ss << "srcAddr:0x" << std::hex << param->outputPtr << ".";

    std::string strTmp = ss.str();
    s32 sRet = strncpy_s(outPut, size, strTmp.c_str(), std::min(size, strTmp.size()));
    if (strTmp.size() >= size || sRet != EOK) {
        HCCL_ERROR("%s strncpy_s fail, src size[%u], dst size[%u], sRet[%d]", strTmp.size(), size, sRet);
    }
}

}