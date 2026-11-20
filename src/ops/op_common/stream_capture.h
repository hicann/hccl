/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_STREAM_CAPTURE_H
#define OPS_HCCL_STREAM_CAPTURE_H

#include "hccl.h"

namespace ops_hccl {
/* 判定stream是否处于图模式捕获（录制）态：nullptr合法（图编译期离线选路无runtime stream）必非捕获，
 * 直接返回false且不进入ACL捕获查询——防空/脏指针传入；
 * 独立轻依赖编译单元（仅ACL头+log），供op_common/协商模块复用，UT可直连真实实现、打桩下沉到ACL层。
 */
bool IsStreamInCaptureMode(aclrtStream stream);
} // namespace ops_hccl

#endif // OPS_HCCL_STREAM_CAPTURE_H
