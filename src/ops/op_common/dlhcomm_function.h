/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_DLHCOMM_FUNCTION
#define OPS_HCCL_DLHCOMM_FUNCTION

#include <functional>
#include <mutex>
#include <dlfcn.h>
#include <hccl/hccl_types.h>
#include "hccl_res.h"
#include "hccl_host_comm_dl.h"
#include <atomic>

namespace ops_hccl {
class DlHcommFunction {
public:
    ~DlHcommFunction();
    static DlHcommFunction& GetInstance();
    HcclResult DlHcommFunctionInit();
    std::function<HcclResult(HcclComm, ThreadHandle, void*, uint32_t, void**)> dlHcclThreadResGetInfo{};
    std::function<HcclResult(HcclComm, HcclConfigType, uint32_t, void*)> dlHcclConfigGetInfo{};
    std::function<HcclResult(ThreadHandle, void*, uint32_t, void**)> dlHcommThreadResGetInfo{};

private:
    void* handle_{nullptr};
    std::mutex handleMutex_;
    DlHcommFunction(const DlHcommFunction&) = delete;
    DlHcommFunction() = default;
    DlHcommFunction& operator=(const DlHcommFunction&) = delete;
    HcclResult DlHcommFunctionInterInit();
};
} // namespace ops_hccl
#endif
