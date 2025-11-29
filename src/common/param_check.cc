/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "param_check.h"
#include <unordered_set>
#include "hccl_common.h"
#include "adapter_error_manager_pub.h"

namespace ops_hccl {

HcclResult HcomCheckGroupName(const char *group)
{
    if (group != nullptr) {
        u32 groupLen = strnlen(group, GROUP_NAME_MAX_LEN + 1);
        if (groupLen == (GROUP_NAME_MAX_LEN + 1) || groupLen == 0) {
            HCCL_ERROR("[Check][GroupName]errNo[0x%016llx] group name[%s] length[%lu] is invalid",
                HCOM_ERROR_CODE(HCCL_E_PARA), group, groupLen);
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckOpParam(const char *tag, const u64 count, const HcclDataType dataType, const char *group,
    const void *stream)
{
    HcclResult ret = HcomCheckGroupName(group);
    RPT_INPUT_ERR(ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({tag, "group", group, "please check group"}));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] group name is invalid",
        HCOM_ERROR_CODE(ret)), ret);

    CHK_RET(HcomCheckOpParam(tag, count, dataType, stream));

    return HCCL_SUCCESS;
}

HcclResult HcomCheckOpParam(const char *tag, const u64 count, const HcclDataType dataType, const void *stream)
{
    CHK_RET(HcomCheckOpParam(tag, count, dataType));

    RPT_INPUT_ERR(stream == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({tag, "stream", "nullptr", "please check stream"}));
    CHK_PTR_NULL(stream);

    return HCCL_SUCCESS;
}

struct EnumHash {
    template <typename T>
    std::size_t operator()(T t) const {
        return static_cast<std::size_t>(t);
    }
};

const std::unordered_set<HcclDataType, EnumHash> HCCL_SUPPORT_DATA_TYPE = {
    HCCL_DATA_TYPE_INT8,
    HCCL_DATA_TYPE_INT16,
    HCCL_DATA_TYPE_INT32,
    HCCL_DATA_TYPE_FP16,
    HCCL_DATA_TYPE_FP32,
    HCCL_DATA_TYPE_INT64,
    HCCL_DATA_TYPE_UINT64,
    HCCL_DATA_TYPE_UINT8,
    HCCL_DATA_TYPE_UINT16,
    HCCL_DATA_TYPE_UINT32,
    HCCL_DATA_TYPE_FP64,
    HCCL_DATA_TYPE_BFP16,
    HCCL_DATA_TYPE_INT128
};

HcclResult HcomCheckTag(const char *tag)
{
    CHK_PTR_NULL(tag);

    u32 tagLen = strnlen(tag, TAG_MAX_LEN + 1);
    if (tagLen == (TAG_MAX_LEN + 1) || tagLen == 0) {
        HCCL_ERROR("[Check][Tag]errNo[0x%016llx] tag is too long", HCOM_ERROR_CODE(HCCL_E_PARA));
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckCount(const u64 count)
{
    if (count > SYS_MAX_COUNT) {
        HCCL_ERROR("[Check][Count]errNo[0x%016llx] count[%llu] is invalid(bigger than MAX count[%llu])",
            HCOM_ERROR_CODE(HCCL_E_PARA), count, SYS_MAX_COUNT);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckDataType(const HcclDataType dataType)
{
    if (HCCL_SUPPORT_DATA_TYPE.find(dataType) == HCCL_SUPPORT_DATA_TYPE.end()) {
        HCCL_ERROR("[Check][DataType]errNo[0x%016llx] data type[%s] not supported",
            HCOM_ERROR_CODE(HCCL_E_NOT_SUPPORT), GetDataTypeEnumStr(dataType).c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult HcomCheckOpParam(const char *tag, const u64 count, const HcclDataType dataType)
{
    HcclResult ret = HcomCheckTag(tag);
    RPT_INPUT_ERR(ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcomCheckTag", "tag", tag, "please check tag"}));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] tag is invalid",
        HCOM_ERROR_CODE(ret)), ret);

    ret = HcomCheckCount(count);
    RPT_INPUT_ERR(ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({tag, "count", std::to_string(count), "please check count"}));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] count is out of range",
        HCOM_ERROR_CODE(ret)), ret);

    ret = HcomCheckDataType(dataType);
    RPT_INPUT_ERR(ret != HCCL_SUCCESS, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({tag, "dataType", GetDataTypeEnumStr(dataType), "please check dataType"}));
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[Check][OpParam]errNo[0x%016llx] dataType is invalid",
        HCOM_ERROR_CODE(ret)), ret);

    return HCCL_SUCCESS;
}

HcclResult HcomCheckUserRank(const u32 totalRanks, const u32 userRank)
{
    if (userRank >= totalRanks) {
        HCCL_ERROR("[Check][UserRank]errNo[0x%016llx] userRank:[%u] is out of range[0 ~ %u]",
            HCOM_ERROR_CODE(HCCL_E_PARA), userRank, totalRanks - 1);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

}