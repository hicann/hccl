/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UT 专用包装编译单元（pairwise_alg_attrs_test 目标独享）：
// 以直接 #include 生产源文件的方式，使 ins_v2_all_to_all_v_sole_executor.cc 中的
// REGISTER_ALG_ATTRS 注册体在本测试进程中执行，AlgAttrsRegistry 中的
// topoCustomCheck / topoPriorityCheck lambda 即为生产真实实现，UT 断言直接作用于
// 生产注册项（非测试内复刻条件）。
//
// 同时将 REGISTER_EXEC_V2 展开置空：executor 注册会实例化
// InsV2AlltoAllVSoleExecutor 及 AICPU/AIV/CCU 算法模板（vtable 隐式实例化拉入
// 设备侧符号），与"仅验证 attrs 注册"的 UT 目标无关。置空后本 TU 不再实例化任何
// executor/template 代码，仅需少量链接桩（见 stub/pairwise_alg_attrs_stub.cc）。
//
// 本包装仅存在于 test/ 目录，不构成对生产源文件的任何修改。

#include "coll_alg_v2_exec_registry.h"

#undef REGISTER_EXEC_V2
#define REGISTER_EXEC_V2(type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate)

#include "ins_v2_all_to_all_v_sole_executor.cc"
