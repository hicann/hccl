/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_COMMON_H
#define OPS_HCCL_COMMON_H

#include <map>
#include <set>
#include <vector>
#include <queue>
#include <string>
#include <unordered_map>
#include <algorithm>
#include "dtype_common.h"
#include "log.h"
#include "hccl_types.h"
#include "hccl_res.h"

#ifndef T_DESC
#define T_DESC(_msg, _y) ((_y) ? true : false)
#endif

#ifndef HCCL_DEPRECATED
#define HCCL_DEPRECATED(class) [[deprecated("Use "#class" instead")]]
#endif

#if T_DESC("公共常量及宏", true)
/* 未使用的参数声明 */
#define UNUSED_PARAM(x) (void)(x)
constexpr u32 INVALID_VALUE_RANKID = 0xFFFFFFFF; // rank id非法值
constexpr u32 INVALID_VALUE_RANKSIZE = 0xFFFFFFFF; // rank size非法值
constexpr u32 INVALID_UINT = 0xFFFFFFFF;
constexpr s32 INVALID_INT = 0xFFFFFFFF;
// 系统常用参数
constexpr u64 SYS_MAX_COUNT = 0x7FFFFFFFF; // 系统当前支持的最大count数
constexpr s32 HOST_DEVICE_ID = -1;
constexpr u32 GROUP_NAME_MAX_LEN = 127; // 最大的group name 长度

#endif

/* 公共模块函数返回值定义,跟业务层同步  */
const std::map<HcclDataType, std::string> HCOM_DATA_TYPE_STR_MAP{
    {HcclDataType::HCCL_DATA_TYPE_INT8, "int8"},
    {HcclDataType::HCCL_DATA_TYPE_INT16, "int16"},
    {HcclDataType::HCCL_DATA_TYPE_INT32, "int32"},
    {HcclDataType::HCCL_DATA_TYPE_INT64, "int64"},
    {HcclDataType::HCCL_DATA_TYPE_UINT64, "uint64"},
    {HcclDataType::HCCL_DATA_TYPE_FP16, "float16"},
    {HcclDataType::HCCL_DATA_TYPE_FP32, "float32"},
    {HcclDataType::HCCL_DATA_TYPE_UINT8, "uint8"},
    {HcclDataType::HCCL_DATA_TYPE_UINT16, "uint16"},
    {HcclDataType::HCCL_DATA_TYPE_UINT32, "uint32"},
    {HcclDataType::HCCL_DATA_TYPE_FP64, "float64"},
    {HcclDataType::HCCL_DATA_TYPE_BFP16, "bfloat16"},
    {HcclDataType::HCCL_DATA_TYPE_INT128, "int128"},
    {HcclDataType::HCCL_DATA_TYPE_RESERVED, "reserved"}
};

inline std::string GetDataTypeEnumStr(HcclDataType dataType)
{
    auto iter = HCOM_DATA_TYPE_STR_MAP.find(dataType);
    if (iter == HCOM_DATA_TYPE_STR_MAP.end()) {
        return "HcclDataType(" + std::to_string(dataType) + ")";
    } else {
        return iter->second;
    }
}

const std::map<HcclReduceOp, std::string> HCOM_REDUCE_OP_STR_MAP{
    {HcclReduceOp::HCCL_REDUCE_SUM, "sum"},
    {HcclReduceOp::HCCL_REDUCE_PROD, "prod"},
    {HcclReduceOp::HCCL_REDUCE_MAX, "max"},
    {HcclReduceOp::HCCL_REDUCE_MIN, "min"},
    {HcclReduceOp::HCCL_REDUCE_RESERVED, "reserved"}
};

inline std::string GetReduceOpEnumStr(HcclReduceOp reduceOp)
{
    auto iter = HCOM_REDUCE_OP_STR_MAP.find(reduceOp);
    if (iter == HCOM_REDUCE_OP_STR_MAP.end()) {
        return "HcclReduceOp(" + std::to_string(reduceOp) + ")";
    } else {
        return iter->second;
    }
}

constexpr u32 HCCL_ALGO_LEVEL_0 = 0;        // HCCL 算法层级0
constexpr u32 HCCL_ALGO_LEVEL_1 = 1;        // HCCL 算法层级1
constexpr u32 HCCL_ALGO_LEVEL_2 = 2;        // HCCL 算法层级2
constexpr u32 HCCL_ALGO_LEVEL_3 = 3;        // HCCL 算法层级3
constexpr u32 HCCL_ALGO_LEVEL_NUM = 4;      // HCCL 算法层级最多4级
constexpr u32 HCCL_INVALID_PORT = 65536;  // HCCL 默认无效端口号

inline std::string GetDataTypeEnumStr(u32 dataType)
{
    auto hcclDataType = static_cast<HcclDataType>(dataType);
    return GetDataTypeEnumStr(hcclDataType);
}

// server内link类型
enum class LinkTypeInServer {
    HCCS_TYPE = 0,
    PXI_TYPE = 1,
    SIO_TYPE = 2,
    HCCS_SW_TYPE = 3,
    RESERVED_LINK_TYPE
};

enum class HcclRtDeviceModuleType {
    HCCL_RT_MODULE_TYPE_SYSTEM = 0,  /**< system info*/
    HCCL_RT_MODULE_TYPE_AICORE,      /**< AI CORE info*/
    HCCL_RT_MODULE_TYPE_VECTOR_CORE, /**< VECTOR CORE info*/
    HCCL_RT_DEVICE_MOUDLE_RESERVED,
};

enum class HcclRtDeviceInfoType {
    HCCL_INFO_TYPE_CORE_NUM,
    HCCL_INFO_TYPE_PHY_CHIP_ID,
    HCCL_INFO_TYPE_SDID,
    HCCL_INFO_TYPE_SERVER_ID,
    HCCL_INFO_TYPE_SUPER_POD_ID,
    HCCL_INFO_TYPE_CUST_OP_ENHANCE,
    HCCL_RT_DEVICE_INFO_RESERVED,
};

/**
 * @enum HcclMemType
 * @brief 内存类型枚举定义
 */
typedef enum {
    HCCL_MEM_TYPE_DEVICE, ///< 设备侧内存（如NPU等）
    HCCL_MEM_TYPE_HOST,   ///< 主机侧内存
    HCCL_MEM_TYPE_NUM     ///< 内存类型数量
} HcclMemType;

typedef struct {
    HcclMemType type;
    void *addr;
    uint64_t size;
} HcclMem;
#endif // HCCL_COMMON_H
