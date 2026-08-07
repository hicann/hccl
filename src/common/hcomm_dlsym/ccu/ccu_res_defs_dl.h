/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RES_DEFS_DL_H
#define CCU_RES_DEFS_DL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * CCU 资源类型与描述符句柄的本地定义。
 *
 * 背景：hcomm 包从某个 9.1.0 修订版开始才提供 ccu_res_defs.h，但同一个 9.1.0 版本号的
 * 旧 hcomm 包并不包含该文件，编译期 CANN_VERSION_NUM 无法区分这两种 9.1.0 包。
 * 因此 HCCL 不再依赖 hcomm 的 ccu_res_defs.h，统一用本地定义。
 *
 * ABI 安全性：本文件中的 HcommCcuResDescHandle（uint64_t）与 HcommCcuResType（enum）
 * 的二进制布局与 hcomm 包内定义完全一致，运行时通过 dlsym 调用 hcomm 库函数时
 * 参数传递 ABI 兼容。若未来 hcomm 侧调整枚举值或类型大小，需同步更新此处。
 */

/**
 * @brief CCU资源描述符句柄类型
 */
typedef uint64_t HcommCcuResDescHandle;

/**
 * @brief CCU资源类型枚举
 */
typedef enum {
    HCOMM_CCU_RES_TYPE_INVALID = -1,   ///< 无效资源类型
    HCOMM_CCU_RES_TYPE_LOOP = 0,       ///< Loop资源
    HCOMM_CCU_RES_TYPE_CCU_BUF = 1,    ///< CCU Buffer资源
    HCOMM_CCU_RES_TYPE_VARIABLE = 2,   ///< Variable资源
    HCOMM_CCU_RES_TYPE_ADDRESS = 3,    ///< Address资源
    HCOMM_CCU_RES_TYPE_EVENT = 4,      ///< Event资源
    HCOMM_CCU_RES_TYPE_CCU_THREAD = 5, ///< CCU Thread资源
    HCOMM_CCU_RES_TYPE_INSTRUCTION = 6 ///< Instruction资源
} HcommCcuResType;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CCU_RES_DEFS_DL_H
