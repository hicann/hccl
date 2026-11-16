/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSR_HOST_TEST_COLL_ALG_V2_EXEC_REGISTRY_H
#define BSR_HOST_TEST_COLL_ALG_V2_EXEC_REGISTRY_H

// CollAlgExecRegistryV2 的桩：被测 .cc 包含它只是为了文件末尾的 REGISTER_EXEC_V2。
//
// 生产头把注册展开成 CollAlgExecRegistryV2::Instance().Register(...) + AddAlgToAllAlgos(...)，
// 这两者在 UT 里既没有实现、也没有被测价值（工厂注册与被测的编排/切片/tag 逻辑无关），所以这里
// 把注册宏展开为空。CollAlgExecRegistryV2 类本身不提供——若被测代码真的去查注册表，UT 会在编译期
// 报错，而不是静默拿到一个空实现。
//
// 其余注册宏（REGISTER_EXECUTOR_BY_TOPO / _BY_TWO_TEMPS / _BY_FOUR_TEMPS / REGISTER_EXEC_V2_MULTI）
// BSR 的执行器一个都没用，故不提供。

#include "executor_v2_base.h"

#define REGISTER_EXEC_V2(type, name, insCollAlgBase, AlgTopoMatch, InsAlgTemplate)

#endif // BSR_HOST_TEST_COLL_ALG_V2_EXEC_REGISTRY_H
