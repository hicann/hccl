/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_HCOMM_GATE_STUB_H
#define UT_HCOMM_GATE_STUB_H

#include <cstdint>

void GateStubResetNotifyRecordCnt();

uint32_t GateStubGetNotifyRecordCnt();

uint64_t GateStubGetLastNotifySrcThread();

uint64_t GateStubGetLastNotifyDstThread();

uint32_t GateStubGetLastNotifyIdx();

#endif // UT_HCOMM_GATE_STUB_H
