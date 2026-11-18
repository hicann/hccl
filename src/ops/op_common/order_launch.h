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

namespace ops_hccl {

enum class OrderLaunchMode {
    ORDER_LAUNCH_OPBASE,
    ORDER_LAUNCH_ACLGRAPH,
    ORDER_LAUNCH_GE,
};

class OrderLaunch {
public:
    OrderLaunch() = default;
    ~OrderLaunch() = default;
    OrderLaunch(const OrderLaunch&) = delete;
    OrderLaunch& operator=(const OrderLaunch&) = delete;

    HcclResult HcclOrderLaunchPrepare(HcclComm comm, const OpParam& param, ThreadHandle& unfoldThread);
    HcclResult HcclOrderLaunchToOrderStream(HcclComm comm, OpParam& param, ThreadHandle unfoldThread, u32 notifyIdx);
    HcclResult HcclOrderLaunchToKernelStream(HcclComm comm, u32 notifyIdx);

private:
    u32 execTimeout_{CUSTOM_TIMEOUT};
    OrderLaunchMode launchMode_{OrderLaunchMode::ORDER_LAUNCH_OPBASE};
    bool orderLaunchRequired_{true};
    // Phase1获取并缓存的Host侧保序线程句柄，Phase2直接复用，避免二次acquire结果漂移
    ThreadHandle hostOrderThread_{0};
};

} // namespace ops_hccl
#endif
