/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_ORDER_LAUNCH
#define OPS_ORDER_LAUNCH

#include <string>
#include <memory>
#include "hccl.h"

#include "alg_param.h"
#include "executor_v2_base.h"
#include "alg_type.h"
#include "execute_selector.h"
#include "acl/acl_rt.h"

using HcclRtEvent = void*;

namespace ops_hccl {

enum class OrderLaunchMode {
    ORDER_LAUNCH_OPBASE,
    ORDER_LAUNCH_ACLGRAPH,
    ORDER_LAUNCH_GE,
};

class HcclRtEventGuard {
public:
    HcclRtEventGuard() = default;
    ~HcclRtEventGuard()
    {
        if (event_ != nullptr) {
            (void)aclrtDestroyEvent(event_);
        }
    }
    HcclRtEventGuard(const HcclRtEventGuard&) = delete;
    HcclRtEventGuard& operator=(const HcclRtEventGuard&) = delete;
    HcclResult Create()
    {
        aclError ret = aclrtCreateEventExWithFlag(&event_, ACL_EVENT_SYNC);
        CHK_PRT_RET(
            ret != ACL_SUCCESS, HCCL_ERROR("aclrtCreateEventExWithFlag failed, ret[%d] event[%p].", ret, event_),
            HCCL_E_RUNTIME);
        return HCCL_SUCCESS;
    }
    HcclRtEvent Get() const { return event_; }

private:
    HcclRtEvent event_ = nullptr;
};

HcclResult HcclOrderLaunchToOrderStream(
    HcclComm comm, OpParam& param, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout, OrderLaunchMode mode,
    HcclRtEvent event);
HcclResult HcclOrderLaunchToKernelStream(
    HcclComm comm, ThreadHandle unfoldThread, u32 notifyIdx, u32 timeout, OrderLaunchMode mode, HcclRtEvent event);

} // namespace ops_hccl
#endif
