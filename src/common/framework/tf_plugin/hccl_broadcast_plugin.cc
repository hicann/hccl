/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/register.h"

namespace domi {
static Status AutoMappingFnHcomBroadcast(const google::protobuf::Message* op_src, ge::Operator& op)
{
    if (op_src == nullptr) {
        return FAILED;
    }
    map<string, pair<string, string>> value;
    value["in"] = pair<string, string>("x", "T");
    value["out"] = pair<string, string>("y", "T");
    if (AutoMappingFnDynamic(op_src, op, value) == SUCCESS) {
        return SUCCESS;
    } else {
        return FAILED;
    }
}

// register HcomBroadcast op to GE
REGISTER_CUSTOM_OP("HcomBroadcast")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("HcomBroadcast")
    .ParseParamsFn(AutoMappingFnHcomBroadcast)
    .ImplyType(ImplyType::HCCL);
} // namespace domi
