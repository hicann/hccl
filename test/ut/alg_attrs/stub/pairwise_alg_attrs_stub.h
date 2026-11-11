/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// pairwise_alg_attrs_stub.cc 可注入返回值的 accessor 声明。
// 测试通过 SetHcclAlgoOverride / SetEnvOverride 注入 HCCL_ALGO / 环境变量值，
// 通过 ResetAlgAttrsStub 恢复默认（空串）。

#ifndef PAIRWISE_ALG_ATTRS_STUB_H
#define PAIRWISE_ALG_ATTRS_STUB_H

#include <string>

namespace ops_hccl {
void SetHcclAlgoOverride(const std::string& algo);
void SetEnvOverride(const std::string& key, const std::string& value);
void ResetAlgAttrsStub();
} // namespace ops_hccl

#endif // PAIRWISE_ALG_ATTRS_STUB_H
