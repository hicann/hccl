/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_ALG_SELECTOR_EXECUTION
#define HCCLV2_COLL_ALG_SELECTOR_EXECUTION

#include <string>
#include "alg_param.h"
#include "auto_selector_base.h"

namespace ops_hccl {
class ExecuteSelector {
public:
    ExecuteSelector();
    
    HcclResult  Run(OpParam &opParam, TopoInfo* topoInfo, std::string &selectAlgName, OpExecuteConfig &opExecuteConfig);

};
} // namespace Hccl
#endif