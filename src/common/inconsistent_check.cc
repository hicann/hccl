/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "inconsistent_check.h"
#include "hcom.h"
#include "alg_env_config.h"
#include "adapter_error_manager_pub.h"
#include "hccl_res_expt_dl.h"

namespace ops_hccl {
thread_local std::set<std::string> g_inconsistentCheckedList;
bool NeedInconsistentCheck(const OpParam& param)
{
    if (HcommIsSupportHcclCommAddExchangeInfo()) {
        // 以下场景不校验参数一致性，其余场景均校验：
        // inconsistentCheckSwitch 为 off
        // inconsistentCheckSwitch 为 first 或空，单算子模式下非首次下发且非增量建链模式
        std::string tagStr = param.algTag;
        bool noCheck = (GetInconsistentCheckSwitch() == 0) && (param.opMode == OpMode::OPBASE) &&
            (g_inconsistentCheckedList.find(tagStr) != g_inconsistentCheckedList.end());
        bool increCreateChannelFlag = (param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) &&
            (param.opMode == OpMode::OPBASE);
        if (GetInconsistentCheckSwitch() == -1 || (noCheck && !increCreateChannelFlag)) {
            return false;
        } else {
            return true;
        }
    }
    return false;
}

HcclResult CompareOpExchangeInfos(HcclComm comm, const OpParam& param, const AlgResourceRequest &resRequest,
    const OpExchangeInfo &exchangeInfo)
{
    if (HcommIsSupportHcclCommGetExchangeInfo()) {
        if (param.engine != COMM_ENGINE_CCU) {
            for (u32 level = 0; level < resRequest.channels.size(); level++) {
                CHK_RET(InconsistentCheckParams(comm, exchangeInfo, resRequest.channels[level]));
            }
        } else {
            for (auto &kernelInfo: resRequest.ccuKernelInfos) {
                CHK_RET(InconsistentCheckParams(comm, exchangeInfo, kernelInfo.channels));
            }
        }
        std::string tagStr = param.algTag;
        g_inconsistentCheckedList.insert(tagStr);
    }
    return HCCL_SUCCESS;
}

HcclResult InconsistentCheckParams(HcclComm comm, const OpExchangeInfo &exchangeInfo,
    const std::vector<HcclChannelDesc> &channels)
{
    CHK_PTR_NULL(comm);
    if (channels.empty()) {
        HCCL_INFO("[InconsistentCheckParams] channels is empty.");
        return HCCL_SUCCESS;
    }
    for (auto &channel : channels) {
        OpExchangeInfo rmtExchangeInfo{};
        uint32_t rmtDataLen = 0;
        CHK_RET(HcclCommGetExchangeInfo(comm, channel.remoteRank, sizeof(OpExchangeInfo),
            reinterpret_cast<void*>(&rmtExchangeInfo), &rmtDataLen));
        if (rmtDataLen == 0) {
            HCCL_INFO("[InconsistentCheckParams] rmtDataLen is 0. Skip. remoteRank[%u]", channel.remoteRank);
            continue;
        } else if (rmtDataLen != sizeof(OpExchangeInfo)) {
            HCCL_ERROR("[InconsistentCheckParams] locDataLen is not equal to rmtDataLen. remoteRank[%u]", 
                channel.remoteRank);
            return HCCL_E_PARA;
        }
        HCCL_INFO("[InconsistentCheckParams] check OpExchangeInfo from remoteRank[%u]", channel.remoteRank);
        if (exchangeInfo.cclBufferSize != rmtExchangeInfo.cclBufferSize) {
            CHK_RET(ReportOpExchangeInfoCheckFailed(exchangeInfo, "HcclBufferSize",
                std::to_string(exchangeInfo.cclBufferSize), std::to_string(rmtExchangeInfo.cclBufferSize)));
        }
        if (exchangeInfo.root != rmtExchangeInfo.root) {
            CHK_RET(ReportOpExchangeInfoCheckFailed(exchangeInfo, "RootRankId", exchangeInfo.root,
                rmtExchangeInfo.root));
        }
        CHK_RET(InconsistentCheckOpType(exchangeInfo, rmtExchangeInfo.opType));
        if (exchangeInfo.opExecuteConfig != rmtExchangeInfo.opExecuteConfig) {
            CHK_RET(ReportOpExchangeInfoCheckFailed(exchangeInfo, "OpExecuteConfig",
                static_cast<uint32_t>(exchangeInfo.opExecuteConfig),
                static_cast<uint32_t>(rmtExchangeInfo.opExecuteConfig)));
        }
        if (exchangeInfo.reduceType != rmtExchangeInfo.reduceType) {
            CHK_RET(ReportOpExchangeInfoCheckFailed(exchangeInfo, "HcclReduceOp",
                static_cast<uint32_t>(exchangeInfo.reduceType), static_cast<uint32_t>(rmtExchangeInfo.reduceType)));
        }
        if (exchangeInfo.dataType != rmtExchangeInfo.dataType) {
            CHK_RET(ReportOpExchangeInfoCheckFailed(exchangeInfo, "HcclDataType",
                static_cast<uint32_t>(exchangeInfo.dataType), static_cast<uint32_t>(rmtExchangeInfo.dataType)));
        }
        if (exchangeInfo.count != rmtExchangeInfo.count) {
            CHK_RET(ReportOpExchangeInfoCheckFailed(exchangeInfo, "DataCount", std::to_string(exchangeInfo.count),
                std::to_string(rmtExchangeInfo.count)));
        }
        if (exchangeInfo.aivCoreLimit != rmtExchangeInfo.aivCoreLimit) {
            HCCL_RUN_WARNING("[InconsistentCheckParams]op information aivCoreLimit check fail."
                " expectValue[%u] remotePara[%u]", exchangeInfo.aivCoreLimit, rmtExchangeInfo.aivCoreLimit);
        }
        if (strncmp(exchangeInfo.group, rmtExchangeInfo.group, MAX_LENGTH) != 0) {
            CHK_RET(ReportOpExchangeInfoCheckFailed(exchangeInfo, "GroupName", exchangeInfo.group,
                rmtExchangeInfo.group));
        }
        if (strncmp(exchangeInfo.tag, rmtExchangeInfo.tag, TAG_LENGTH) != 0) {
            CHK_RET(ReportOpExchangeInfoCheckFailed(exchangeInfo, "OpTag", exchangeInfo.tag,
                rmtExchangeInfo.tag));
        }
        HCCL_INFO("[InconsistentCheckParams] success. remoteRank[%u]", channel.remoteRank);
    }
    HCCL_INFO("[InconsistentCheckParams] all exchangeInfos checked successfully. tag[%s]", exchangeInfo.tag);
    return HCCL_SUCCESS;
}

HcclResult InconsistentCheckOpType(const OpExchangeInfo &exchangeInfo, const HcclCMDType &rmtOpType)
{
    HcclCMDType locOpType = exchangeInfo.opType;
    if (locOpType == HcclCMDType::HCCL_CMD_SEND || locOpType == HcclCMDType::HCCL_CMD_RECEIVE) {
        // HcclCMDType::HCCL_CMD_SEND和HcclCMDType::HCCL_CMD_RECEIVE的枚举值需确保大于等于0
        uint32_t expectValue = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_SEND) +
            static_cast<uint32_t>(HcclCMDType::HCCL_CMD_RECEIVE) - static_cast<uint32_t>(locOpType);
        if ((rmtOpType != HcclCMDType::HCCL_CMD_SEND && rmtOpType != HcclCMDType::HCCL_CMD_RECEIVE) ||
            locOpType == rmtOpType) {
            CHK_RET(ReportOpExchangeInfoCheckFailed(exchangeInfo, "OpType", expectValue,
                static_cast<uint32_t>(rmtOpType)));
        }
    } else if (locOpType != rmtOpType) {
        CHK_RET(ReportOpExchangeInfoCheckFailed(exchangeInfo, "OpType",
            static_cast<uint32_t>(locOpType), static_cast<uint32_t>(rmtOpType)));
    }
    return HCCL_SUCCESS;
}

HcclResult ReportOpExchangeInfoCheckFailed(const OpExchangeInfo &exchangeInfo, const std::string &paraName,
    uint32_t expectVal, uint32_t remotePara)
{
    std::string opInfo = "Unknown";
    CHK_RET(GetOpTypeName(exchangeInfo, opInfo));

    RPT_INPUT_ERR(true, "EI0005",
        std::vector<std::string>({"ccl_op", "group", "para_name", "local_para", "remote_para"}),
        std::vector<std::string>({opInfo, exchangeInfo.group, paraName, std::to_string(expectVal),
        std::to_string(remotePara)}));
    HCCL_ERROR("[ReportOpExchangeInfoCheckFailed]op information %s check fail. expectValue[%u] remotePara[%u]",
        paraName.c_str(), expectVal, remotePara);
    return HCCL_E_PARA;
}

HcclResult ReportOpExchangeInfoCheckFailed(const OpExchangeInfo &exchangeInfo, const std::string &paraName,
    const std::string &expectVal, const std::string &remotePara)
{
    std::string opInfo = "Unknown";
    CHK_RET(GetOpTypeName(exchangeInfo, opInfo));
    RPT_INPUT_ERR(true, "EI0005",
        std::vector<std::string>({"ccl_op", "group", "para_name", "local_para", "remote_para"}),
        std::vector<std::string>({opInfo, exchangeInfo.group, paraName, expectVal, remotePara}));
    HCCL_ERROR("[ReportOpExchangeInfoCheckFailed]op information %s check fail. expectValue[%s] remotePara[%s]",
        paraName.c_str(), expectVal.c_str(), remotePara.c_str());
    return HCCL_E_PARA;
}

HcclResult GetOpTypeName(const OpExchangeInfo &exchangeInfo, std::string &opInfo)
{
    for (const auto &pair : HCCL_OPTYPE_NAME_MAP) {
        if (pair.second == exchangeInfo.opType) {
            opInfo = std::string(pair.first);
            break;
        }
    }
    return HCCL_SUCCESS;
}

}