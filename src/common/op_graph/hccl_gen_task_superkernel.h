/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_gen_task_superkernel.h
 * \brief
 */

#ifndef CANN_HCCL_GEN_TASK_SUPERKERNEL_H_
#define CANN_HCCL_GEN_TASK_SUPERKERNEL_H_

#include <string>
#include <vector>
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "graph/arg_desc_info.h"

namespace ops {
// A5 及以后代际判断（纯函数，便于 UT 覆盖）：Ascend950/Ascend960/... 及 A5 同代 Ascend350 系列（含带后缀型号）返回
// true。
bool IsA5OrLaterShortSocVersion(const std::string& shortSocVersion);

// 构造 HCCL SuperKernel 子算子的 args format：布局固定为 hidden input 在前，随后 input、output。
ge::AscendString BuildSuperKernelArgsFormat();

// 版本守卫（GE 弱符号兼容方案，见 cann/ge!5881）：GE 自 9.3.0 起将 TaskDefBuilder 开放为弱符号，
// 并新增弱符号版本探测接口 ge::IsVersionSupported。老 toolkit（<9.3.0）的 GE 库无这些符号，
// 弱引用运行期解析为空指针，故须先判空再调用版本接口；返回 false 表示运行环境不支持
// SuperKernel GenTask，注册方不注册回调、回调方优雅失败，由 toolkit 自带的旧 superkernel
// 实现接管兜底（存量行为不变）。
bool IsGeSuperKernelGenTaskSupported();

ge::Status
HcclSuperKernelGenTaskCallback(const gert::ExeResGenerationContext* context, std::vector<std::vector<uint8_t>>& tasks);

} // namespace ops
#endif // CANN_HCCL_GEN_TASK_SUPERKERNEL_H_
