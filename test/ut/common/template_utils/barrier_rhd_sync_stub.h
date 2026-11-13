/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace barrier_rhd_sync_ut {
inline constexpr uint32_t REQUEST_MSG_ID = 7;
extern std::vector<std::string> calls;
extern int failSyncCall;
extern int failSendCall;
extern int failDpuSendCall;
extern uint32_t responseMsgId;
extern bool failBatchEnd;
extern bool failBatchStart;
extern bool useHostRemoteEndpoint;
void Reset();
} // namespace barrier_rhd_sync_ut
