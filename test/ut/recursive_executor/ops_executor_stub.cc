/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UT 桩：仅提供 OpsExecutor 的构造/析构符号，供 algo_desc.cc 中的
// HcclAlgorithm::GetExecutor（std::make_unique<OpsExecutor>）链接使用。
// 完整 OpsExecutor 实现依赖 src 大量运行时符号，不适合在 UT 中整体编译。

#include "executor/ops_executor.h"

namespace ops_hccl {

OpsExecutor::OpsExecutor(HcclAlgorithm& algo, const OpParam& param) : algo_(algo) {}

OpsExecutor::~OpsExecutor() {}

} // namespace ops_hccl
