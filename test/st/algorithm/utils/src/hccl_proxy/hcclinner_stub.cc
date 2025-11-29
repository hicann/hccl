/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include "hccl/hcom.h"

const bool& GetExternalInputHcclEnableEntryLog()
{
    static const bool value = false;
    return value;
}
 
const bool& GetExternalInputHcclAicpuUnfold()
{
    static const bool value = false;
    return value;
}
 
bool HcclCheckLogLevel(int logType, int moduleId)
{
    return true;
}
 
bool IsErrorToWarn()
{
    return false;
}
 
const u32& GetExternalInputIntraRoceSwitch()
{
    static const u32 value = 0;
    return value;
}
 
HcclResult InitExternalInput()
{
    return HCCL_SUCCESS;
}
 
HcclWorkflowMode GetWorkflowMode()
{
    return HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}
 
const bool& GetExternalInputInterHccsDisable()
{
    static const bool getExternalInputInterHccsDisableValue = false;
    return getExternalInputInterHccsDisableValue;
}

namespace hccl {
u64 GetDebugConfig()
{
    return 0;
}
}  // namespace hccl