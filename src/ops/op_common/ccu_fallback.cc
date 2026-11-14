/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring> // 包含strncmp函数
#include <vector>
#include "ccu_fallback.h"
#include "ccu_fallback_c.h"
#include "comm_negotiation_c.h"
#include "op_common.h"
#include "inconsistent_check.h"
#include "log.h"
#include "dlhcomm_function.h"

// CCU 协商基础设施（协商子通信域/资源生命周期/CommWorker/EngineCtx缓存）已迁移至公共协商模块
// comm_negotiation.{h,cc}：资源协商走 NegotiateCommC(ALL_REDUCE_MIN)，参数检视走 NegotiateCommC(ALL_GATHER)；
// 本文件只保留 CCU 策略层（门控/参数比对/回退决策/结果缓存）。

namespace ops_hccl {
// 协商typeTag：资源协商与参数检视共用同一协商子通信域（复刻迁移前单子通信域行为，
// 公共模块缓冲按需扩容复用）
constexpr char CCU_NEGOTIATION_TAG[] = "ccu";

struct CheckParamInfo {
    uint64_t count;
    uint32_t opExecuteConfig;
    uint32_t dataType;
    uint32_t opType;
    // 载荷有效性标记：各rank填充同值，收包侧校验，防协商原语静默假成功（设备侧任务
    // 失败但host全链路返回成功，recvBuf为清零后的全0垃圾被误当合法协商结果）
    uint32_t magic;
};
static constexpr uint32_t CCU_PARAM_MAGIC = 0x5a5a5a5a;
static constexpr uint32_t EXEC_CONFIG_LEVEL_CCU_MS = 3;
static constexpr uint32_t EXEC_CONFIG_LEVEL_CCU_SCHED = 2;

// 入参必为归一化后的基模式：CCU only 模式在 Selector() 命中 CCU 算法后已归一化为
// CCU_MS/CCU_SCHED（见 op_common.cc），跨 rank 协商交换与比较的配置不会出现 *_ONLY。
static uint32_t GetOpExecuteConfigLevel(OpExecuteConfig config)
{
    switch (config) {
        case OpExecuteConfig::CCU_MS:
            return EXEC_CONFIG_LEVEL_CCU_MS;
        case OpExecuteConfig::CCU_SCHED:
            return EXEC_CONFIG_LEVEL_CCU_SCHED;
        case OpExecuteConfig::AICPU_TS:
            return 1;
        default:
            return 0;
    }
}

HcclResult CheckCcuResNegotiation(HcclComm comm, const OpParam& param, bool localResAvailable)
{
    HCCL_INFO(
        "[%s] start, comm[%p], commName[%s], algName[%s], tag[%s], opType[%u], localResAvailable[%d].", __func__, comm,
        param.commName, param.algName, param.tag, static_cast<u32>(param.opType), static_cast<int>(localResAvailable));

    if (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE
        || param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        HCCL_INFO("[%s] send/recv/batch_send_recv op, skip ccu res negotiation, tag[%s].", __func__, param.tag);
        return HCCL_SUCCESS;
    }

    // 全域 AND 聚合本端资源可用性（公共协商机制，单rank/子通信域防递归由公共模块处理）；
    // 协商基础设施失败（子通信域建链/原语执行/环境异常）按"全域资源不足"同语义处理，返回
    // HCCL_E_UNAVAIL 交由上层释放已申请的CCU通道并回退，禁止上抛（环境抖动上抛会放大为算子执行失败）；
    // 协商结果为"存在资源不足的rank"时同样返回 HCCL_E_UNAVAIL
    bool allRanksCapable = false;
    HcclResult negoRet = NegotiateCommC(
        comm, CCU_NEGOTIATION_TAG, NEGOTIATE_OP_ALL_REDUCE_MIN, &localResAvailable, sizeof(bool), &allRanksCapable,
        sizeof(bool));
    if (negoRet != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[%s] negotiation infrastructure failed, ret[%d], treat as insufficient resources, fallback.", __func__,
            negoRet);
        return HCCL_E_UNAVAIL;
    }
    if (!allRanksCapable) {
        HCCL_WARNING("[%s] negotiation failed, some NPUs have insufficient resources, fallback.", __func__);
        return HCCL_E_UNAVAIL;
    }

    HCCL_INFO("[%s] negotiation success, all NPUs have sufficient resources.", __func__);
    return HCCL_SUCCESS;
}

static HcclResult IsCommExpansionModeCcu(HcclComm comm, bool& isCommCcuMode)
{
    isCommCcuMode = false;
    auto& hcommFunction = ops_hccl::DlHcommFunction::GetInstance();
    if (!hcommFunction.dlHcclConfigGetInfo) {
        HCCL_WARNING("[%s] HcclConfigGetInfo is not supported.", __func__);
        return HCCL_SUCCESS;
    }

    HcclOpExpansionMode mode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    uint32_t infoLen = sizeof(HcclOpExpansionMode);
    CHK_RET(
        hcommFunction.dlHcclConfigGetInfo(comm, HcclConfigType::HCCL_CONFIG_TYPE_OP_EXPANSION_MODE, infoLen, &mode));

    if (mode == static_cast<HcclOpExpansionMode>(HCCL_OP_EXPANSION_EXT_CCU_MS)
        || mode == static_cast<HcclOpExpansionMode>(HCCL_OP_EXPANSION_EXT_CCU_SCHED)
        || mode == static_cast<HcclOpExpansionMode>(HCCL_OP_EXPANSION_EXT_CCU_MS_ONLY)
        || mode == static_cast<HcclOpExpansionMode>(HCCL_OP_EXPANSION_EXT_CCU_SCHED_ONLY)) {
        isCommCcuMode = true;
    }

    HCCL_INFO(
        "[%s] expansionMode[%d], isCommCcuMode[%d].", __func__, static_cast<int>(mode),
        static_cast<int>(isCommCcuMode));
    return HCCL_SUCCESS;
}

// 校验协商收包载荷有效性：任一rank的magic不符视为载荷无效（协商原语静默假成功/
// 数据损坏时recvBuf为协商前清零的全0，或为非法残留值，均无法通过magic校验）
static bool IsValidNegotiationPayload(const CheckParamInfo* recvInfos, u32 rankSize)
{
    for (u32 i = 0; i < rankSize; i++) {
        if (recvInfos[i].magic != CCU_PARAM_MAGIC) {
            HCCL_WARNING(
                "[%s] negotiation payload invalid on rank[%u], magic[0x%x] != expected[0x%x], suspect silent "
                "negotiation failure, fallback to aicpu.",
                __func__, i, recvInfos[i].magic, CCU_PARAM_MAGIC);
            return false;
        }
    }
    return true;
}

static HcclResult
CompareCcuParam(const CheckParamInfo* recvInfos, u32 rankSize, const OpParam& param, OpExecuteConfig& lowestConfig)
{
    bool isVariableCountOp
        = (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC
           || param.opType == HcclCMDType::HCCL_CMD_ALLGATHER_V
           || param.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V
           || param.opType == HcclCMDType::HCCL_CMD_HALF_ALLTOALLV);

    if (!isVariableCountOp) {
        for (u32 i = 0; i < rankSize; i++) {
            if (recvInfos[i].count != param.DataDes.count) {
                OpExchangeInfo exchangeInfo{};
                exchangeInfo.opType = param.opType;
                exchangeInfo.count = param.DataDes.count;
                exchangeInfo.dataType = param.DataDes.dataType;
                errno_t cpyRet = strncpy_s(exchangeInfo.group, MAX_LENGTH, param.commName, COMM_INDENTIFIER_MAX_LENGTH);
                CHK_PRT_RET(
                    cpyRet != EOK, HCCL_ERROR("[%s] strncpy_s for group failed, ret[%d].", __func__, cpyRet),
                    HCCL_E_MEMORY);
                // opType、dataType、datacount均会导致数据量不一致，参照一致性校验进行规范格式报错
                CHK_RET(InconsistentCheckOpType(i, exchangeInfo, static_cast<HcclCMDType>(recvInfos[i].opType)));
                if (recvInfos[i].dataType != static_cast<uint32_t>(param.DataDes.dataType)) {
                    CHK_RET(ReportOpExchangeInfoCheckFailed(
                        i, exchangeInfo, "HcclDataType", static_cast<uint32_t>(param.DataDes.dataType),
                        recvInfos[i].dataType));
                } else {
                    CHK_RET(ReportOpExchangeInfoCheckFailed(
                        i, exchangeInfo, "DataCount", std::to_string(param.DataDes.count),
                        std::to_string(recvInfos[i].count)));
                }
            }
        }
    }

    lowestConfig = static_cast<OpExecuteConfig>(recvInfos[0].opExecuteConfig);
    uint32_t minLevel = GetOpExecuteConfigLevel(lowestConfig);
    for (u32 i = 1; i < rankSize; i++) {
        OpExecuteConfig config = static_cast<OpExecuteConfig>(recvInfos[i].opExecuteConfig);
        uint32_t level = GetOpExecuteConfigLevel(config);
        if (level < minLevel) {
            minLevel = level;
            lowestConfig = config;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CheckCcuParamAndFallback(
    HcclComm comm, OpParam& param, std::unique_ptr<TopoInfoWithNetLayerDetails>& topoInfo, std::string& algName)
{
    HCCL_INFO(
        "[%s] entry, comm[%p], commName[%s], algName[%s], tag[%s], opType[%u], opExecuteConfig[%u], count[%llu].",
        __func__, comm, param.commName, algName.c_str(), param.tag, static_cast<u32>(param.opType),
        static_cast<uint32_t>(param.opExecuteConfig), param.DataDes.count);

    if (param.opType == HcclCMDType::HCCL_CMD_SEND || param.opType == HcclCMDType::HCCL_CMD_RECEIVE
        || param.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
        // 与send/recv同理，batch允许子集参与，故不做跨rank协商；hostDPU场景默认不进协商
        HCCL_INFO(
            "[%s] send/recv/batch_send_recv op, skip ccu param check and fallback, tag[%s].", __func__, param.tag);
        return HCCL_SUCCESS;
    }

    if (strncmp(param.tag, "SelectAlg_", strlen("SelectAlg_")) == 0) {
        HCCL_INFO("[%s] aiv sk selector, skip ccu param check, tag[%s].", __func__, param.tag);
        return HCCL_SUCCESS;
    }

    bool isCcuMode = false;
    CHK_RET(IsCommExpansionModeCcu(comm, isCcuMode));
    if (!isCcuMode) {
        HCCL_INFO("[%s] not ccu mode, skip ccu param check, tag[%s].", __func__, param.tag);
        return HCCL_SUCCESS;
    }

    u32 rankSize = 0;
    CHK_RET(HcclGetRankSize(comm, &rankSize));
    if (rankSize <= 1) {
        HCCL_INFO("[%s] single rank, skip param check.", __func__);
        return HCCL_SUCCESS;
    }

    // 以传入的algName与opExecuteConfig为key查询协商结果缓存，命中则跳过跨rank协商
    std::string cacheTag
        = algName + "_" + std::to_string(static_cast<uint32_t>(param.opExecuteConfig)) + "_ccuParamCache";
    void* cacheCtx = nullptr;
    uint64_t cacheCtxSize = 0;
    HcclResult getRet = HcclEngineCtxGet(comm, cacheTag.c_str(), CommEngine::COMM_ENGINE_CCU, &cacheCtx, &cacheCtxSize);
    if (getRet == HCCL_SUCCESS && cacheCtx != nullptr && cacheCtxSize >= sizeof(FallbackCtxData)) {
        auto* ctxData = static_cast<FallbackCtxData*>(cacheCtx);
        if (ctxData->opExecuteConfig == param.opExecuteConfig) {
            HCCL_INFO(
                "[%s] cache hit, no fallback needed, algName[%s], opExecuteConfig[%u].", __func__, ctxData->algName,
                static_cast<uint32_t>(ctxData->opExecuteConfig));
            return HCCL_SUCCESS;
        }
        // 此前该key发生过回退，直接应用缓存的回退结果
        HCCL_INFO(
            "[%s] cache hit, apply cached fallback, algName[%s], opExecuteConfig[%u].", __func__, ctxData->algName,
            static_cast<uint32_t>(ctxData->opExecuteConfig));
        param.opExecuteConfig = ctxData->opExecuteConfig;
        CHK_RET(SetCommEngine(param));
        algName = ctxData->algName;
        CHK_RET(SetOpParamAlgTag(param, algName));
        return HCCL_SUCCESS;
    }

    // 参数检视协商（公共协商机制 AllGather）：本端参数载荷全域收集后逐rank比对
    CheckParamInfo localInfo{};
    localInfo.count = param.DataDes.count;
    localInfo.opExecuteConfig = static_cast<uint32_t>(param.opExecuteConfig);
    localInfo.dataType = static_cast<uint32_t>(param.DataDes.dataType);
    localInfo.opType = static_cast<uint32_t>(param.opType);
    localInfo.magic = CCU_PARAM_MAGIC;
    HCCL_INFO(
        "[%s] param negotiation, tag[%s], count[%llu], config[%u], dataType[%u], opType[%u].", __func__, param.tag,
        localInfo.count, localInfo.opExecuteConfig, localInfo.dataType, localInfo.opType);
    std::vector<CheckParamInfo> recvInfos(rankSize);
    // 协商基础设施失败（子通信域建链/原语执行/环境异常）按不支持处理：跳过参数比对，统一降级
    // AICPU_TS，复用下方与配置不一致相同的 ONLY 模式拒绝 / ReSelector 回退 / 结果缓存路径；
    // 禁止上抛（环境抖动上抛会放大为整图加载失败），缓存后同key后续算子直接应用降级结果不再重试
    OpExecuteConfig lowestConfig = OpExecuteConfig::DEFAULT;
    HcclResult negoRet = NegotiateCommC(
        comm, CCU_NEGOTIATION_TAG, NEGOTIATE_OP_ALL_GATHER, &localInfo, sizeof(CheckParamInfo), recvInfos.data(),
        recvInfos.size() * sizeof(CheckParamInfo));
    if (negoRet != HCCL_SUCCESS) {
        HCCL_WARNING("[%s] param negotiation infrastructure failed, ret[%d], fallback to aicpu.", __func__, negoRet);
        lowestConfig = OpExecuteConfig::AICPU_TS;
    } else if (!IsValidNegotiationPayload(recvInfos.data(), rankSize)) {
        // 协商原语静默假成功：按协商基础设施失败同语义统一降级AICPU_TS，禁止把垃圾
        // 参数当合法协商结果（变长算子无count比对，垃圾opExecuteConfig会导致误回退/
        // CCU only模式误拒绝）
        lowestConfig = OpExecuteConfig::AICPU_TS;
    } else {
        CHK_RET(CompareCcuParam(recvInfos.data(), rankSize, param, lowestConfig));
    }

    if (lowestConfig != param.opExecuteConfig) {
        if (param.commOpExpansionMode == static_cast<HcclOpExpansionMode>(HCCL_OP_EXPANSION_EXT_CCU_MS_ONLY)
            || param.commOpExpansionMode == static_cast<HcclOpExpansionMode>(HCCL_OP_EXPANSION_EXT_CCU_SCHED_ONLY)) {
            HCCL_ERROR(
                "[%s] config mismatch, local[%u] vs lowest[%u], ccu only mode does not support fallback.", __func__,
                static_cast<uint32_t>(param.opExecuteConfig), static_cast<uint32_t>(lowestConfig));
            return HCCL_E_NOT_SUPPORT;
        }
        HCCL_INFO(
            "[%s] config mismatch, local[%u] vs lowest[%u], fallback.", __func__,
            static_cast<uint32_t>(param.opExecuteConfig), static_cast<uint32_t>(lowestConfig));
        CHK_RET(ReSelector(comm, param, topoInfo, algName, lowestConfig));
    } else {
        HCCL_INFO(
            "[%s] param check success, all ranks consistent, opExecuteConfig[%u].", __func__,
            static_cast<uint32_t>(lowestConfig));
    }

    // 缓存协商结束后的最终algName与opExecuteConfig，后续同key调用跳过协商
    void* newCtx = nullptr;
    CHK_RET(HcclEngineCtxCreate(comm, cacheTag.c_str(), CommEngine::COMM_ENGINE_CCU, sizeof(FallbackCtxData), &newCtx));
    auto* ctxData = static_cast<FallbackCtxData*>(newCtx);
    int copyRet = sprintf_s(ctxData->algName, sizeof(ctxData->algName), "%s", algName.c_str());
    if (copyRet <= 0) {
        HCCL_ERROR("[%s] sprintf_s for algName failed, ret[%d].", __func__, copyRet);
        (void)HcclEngineCtxDestroy(comm, cacheTag.c_str(), CommEngine::COMM_ENGINE_CCU);
        return HCCL_E_INTERNAL;
    }
    ctxData->opExecuteConfig = param.opExecuteConfig;
    return HCCL_SUCCESS;
}
} // namespace ops_hccl

__attribute__((weak)) HcclResult CheckCcuResNegotiationC(HcclComm comm, const void* param, bool localResAvailable)
{
    if (param == nullptr) {
        HCCL_ERROR("[%s] param is null.", __func__);
        return HCCL_E_PARA;
    }
    const auto& opParam = *static_cast<const ops_hccl::OpParam*>(param);
    return ops_hccl::CheckCcuResNegotiation(comm, opParam, localResAvailable);
}

__attribute__((weak)) HcclResult
CheckCcuParamAndFallbackC(HcclComm comm, void* param, void** topoInfo, char* algNameBuf, uint32_t algNameBufLen)
{
    if (param == nullptr || topoInfo == nullptr || algNameBuf == nullptr) {
        HCCL_ERROR("[%s] invalid param.", __func__);
        return HCCL_E_PARA;
    }
    auto& opParam = *static_cast<ops_hccl::OpParam*>(param);
    auto& topoInfoPtr = *static_cast<std::unique_ptr<ops_hccl::TopoInfoWithNetLayerDetails>*>(*topoInfo);
    std::string algName(algNameBuf);
    HcclResult ret = ops_hccl::CheckCcuParamAndFallback(comm, opParam, topoInfoPtr, algName);
    if (ret != HCCL_SUCCESS) {
        return ret;
    }
    errno_t copyRet = strncpy_s(algNameBuf, algNameBufLen, algName.c_str(), algName.size());
    if (copyRet != EOK) {
        HCCL_ERROR("[%s] strncpy_s for algName failed.", __func__);
        return HCCL_E_MEMORY;
    }
    return HCCL_SUCCESS;
}
