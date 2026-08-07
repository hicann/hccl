/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef DEV_TYPE_H
#define DEV_TYPE_H

#include "hccl_types.h"
#include <unordered_map>
#include <string>

typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

const u32 HCCL_TAG_MAX_LEN = 191; // 最大的tag 长度

/**
 * @brief stream handle.
 */
typedef void* rtStream_t;

// 2 is sizeof(float16), 8 is sizeof(float64), 2 is sizeof(bfloat16)..
constexpr u32 HCCL_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED]
    = {sizeof(s8),
       sizeof(s16),
       sizeof(s32),
       2,
       sizeof(float),
       sizeof(s64),
       sizeof(u64),
       sizeof(u8),
       sizeof(u16),
       sizeof(u32),
       8,
       2,
       16,
       2,
       1,
       1,
       1,
       1};

// 对内芯片类型
enum class HcclDevType {
    DEV_TYPE_910 = 0,
    DEV_TYPE_310P3 = 1, // PG
    DEV_TYPE_910B = 2,
    DEV_TYPE_310P1 = 3, // AG
    DEV_TYPE_910_93 = 4,
    DEV_TYPE_NOSOC = 5,
    DEV_TYPE_950 = 6,
    DEV_TYPE_MC62 = 7,
    DEV_TYPE_960 = 8,
    DEV_TYPE_COUNT = 9
};

const std::unordered_map<std::string, HcclDevType> HCCL_SOC_VER_CONVERT{
    {"Ascend310P1", HcclDevType::DEV_TYPE_310P3},
    {"Ascend310P3", HcclDevType::DEV_TYPE_310P3},
    {"Ascend310P5", HcclDevType::DEV_TYPE_310P3},
    {"Ascend310P7", HcclDevType::DEV_TYPE_310P3},
    {"Ascend310B1", HcclDevType::DEV_TYPE_310P3}, // 临时映射，临时当前Ascend310B1
                                                  // torch_npu未与hccl的so解耦；计划20250630完成解耦，解耦后删除
    {"Ascend910", HcclDevType::DEV_TYPE_910},
    {"Ascend910A", HcclDevType::DEV_TYPE_910},
    {"Ascend910B", HcclDevType::DEV_TYPE_910},
    {"Ascend910ProA", HcclDevType::DEV_TYPE_910},
    {"Ascend910ProB", HcclDevType::DEV_TYPE_910},
    {"Ascend910PremiumA", HcclDevType::DEV_TYPE_910},
    {"Ascend910B1", HcclDevType::DEV_TYPE_910B},
    {"Ascend910B2", HcclDevType::DEV_TYPE_910B},
    {"Ascend910B2C", HcclDevType::DEV_TYPE_910B},
    {"Ascend910B3", HcclDevType::DEV_TYPE_910B},
    {"Ascend910B4", HcclDevType::DEV_TYPE_910B},
    {"Ascend910B4-1", HcclDevType::DEV_TYPE_910B},
    {"Ascend910_9391", HcclDevType::DEV_TYPE_910_93},
    {"Ascend910_9381", HcclDevType::DEV_TYPE_910_93},
    {"Ascend910_9392",
     HcclDevType::
         DEV_TYPE_910_93}, // Ascend910_9392、Ascend910_9382为预留类型，当前版本暂不支持，待跟随后续版本节奏交付
    {"Ascend910_9382", HcclDevType::DEV_TYPE_910_93},
    {"Ascend910_9372", HcclDevType::DEV_TYPE_910_93},
    {"Ascend910_9362", HcclDevType::DEV_TYPE_910_93},
    {"Ascend950PR_958b", HcclDevType::DEV_TYPE_950},
    {"nosoc", HcclDevType::DEV_TYPE_NOSOC}};

#define hccl_weak_alias(name, aliasname) _hccl_weak_alias(name, aliasname)
#define _hccl_weak_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((weak, alias(#name)))

#ifdef __cplusplus
extern "C" {
#endif
HcclResult HcclGetDeviceType(HcclDevType& devType);
#ifdef __cplusplus
} // extern "C"
#endif

#endif // DEV_TYPE_H
