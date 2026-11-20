/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_op_utils.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "hccl/hcom.h"
#include "hccl_rt_utils.h"
#include "graph_mode_c_api.h"
#include "log.h"

namespace hccl {
HcclDataType GeDataTypeToHccl(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT:
            return HCCL_DATA_TYPE_FP32;
        case ge::DT_FLOAT16:
            return HCCL_DATA_TYPE_FP16;
        case ge::DT_INT8:
            return HCCL_DATA_TYPE_INT8;
        case ge::DT_INT16:
            return HCCL_DATA_TYPE_INT16;
        case ge::DT_INT32:
            return HCCL_DATA_TYPE_INT32;
        case ge::DT_INT64:
            return HCCL_DATA_TYPE_INT64;
        case ge::DT_UINT8:
            return HCCL_DATA_TYPE_UINT8;
        case ge::DT_UINT16:
            return HCCL_DATA_TYPE_UINT16;
        case ge::DT_UINT32:
            return HCCL_DATA_TYPE_UINT32;
        case ge::DT_UINT64:
            return HCCL_DATA_TYPE_UINT64;
        case ge::DT_DOUBLE:
            return HCCL_DATA_TYPE_FP64;
        case ge::DT_BF16:
            return HCCL_DATA_TYPE_BFP16;
        // bool 按 1 字节搬移类算子语义映射 int8 (对照旧路径 hcom_ome_util kConstOpHcclDataType
        // 的 DT_BOOL→INT8 与其注释"搬移类算子将 int8 当 bool")
        case ge::DT_BOOL:
            return HCCL_DATA_TYPE_INT8;
        // FP8 族 (对照旧路径 kConstOpHcclDataType + 引擎 HCOM_DATA_TYPE_MAP 的并集; E8M0 仅引擎表有)。
        // 版本前提由探测门控保证: probe 通过 ⇒ 自定义算子框架代 toolkit ⇒ hccl_types.h 必含 FP8 枚举
        case ge::DT_HIFLOAT8:
            return HCCL_DATA_TYPE_HIF8;
        case ge::DT_FLOAT8_E5M2:
            return HCCL_DATA_TYPE_FP8E5M2;
        case ge::DT_FLOAT8_E4M3FN:
            return HCCL_DATA_TYPE_FP8E4M3;
        case ge::DT_FLOAT8_E8M0:
            return HCCL_DATA_TYPE_FP8E8M0;
        default:
            return HCCL_DATA_TYPE_RESERVED;
    }
}

HcclReduceOp StringToReduceOp(const char* reduction)
{
    if (reduction == nullptr) {
        return HCCL_REDUCE_RESERVED;
    }
    if (strcmp(reduction, "sum") == 0) {
        return HCCL_REDUCE_SUM;
    }
    if (strcmp(reduction, "prod") == 0) {
        return HCCL_REDUCE_PROD;
    }
    if (strcmp(reduction, "max") == 0) {
        return HCCL_REDUCE_MAX;
    }
    if (strcmp(reduction, "min") == 0) {
        return HCCL_REDUCE_MIN;
    }
    return HCCL_REDUCE_RESERVED;
}

uint64_t GetHcclDataTypeSize(HcclDataType dt)
{
    switch (dt) {
        case HCCL_DATA_TYPE_INT8:
        case HCCL_DATA_TYPE_UINT8:
        // FP8 族与 GeDataTypeToHccl 的映射同步 (HIF8/E4M3/E5M2/E8M0 均为 1 字节)
        case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3:
        case HCCL_DATA_TYPE_FP8E5M2:
        case HCCL_DATA_TYPE_FP8E8M0:
            return 1;
        case HCCL_DATA_TYPE_INT16:
        case HCCL_DATA_TYPE_UINT16:
        case HCCL_DATA_TYPE_FP16:
        case HCCL_DATA_TYPE_BFP16:
            return 2;
        case HCCL_DATA_TYPE_INT32:
        case HCCL_DATA_TYPE_UINT32:
        case HCCL_DATA_TYPE_FP32:
            return 4;
        case HCCL_DATA_TYPE_INT64:
        case HCCL_DATA_TYPE_UINT64:
        case HCCL_DATA_TYPE_FP64:
            return 8;
        case HCCL_DATA_TYPE_INT128:
            return 16;
        default:
            return 0;
    }
}

ge::graphStatus CalcLoopMaxTime(
    const std::string& opType, uint64_t count, uint64_t unitSize, uint64_t cclBuffSize, uint32_t rankSize,
    uint32_t& loopMaxTime)
{
    // 与 LaunchRefresh 循环同源推导: loopMaxTime = ceil(count / maxCountPerLoop), 保证 tags.size() 恒等于
    // 实际迭代数 (公式表详见 hccl_op_utils.h 注释)
    loopMaxTime = 0;
    uint64_t maxCountPerLoop = 0;
    if (unitSize == 0) {
        HCCL_ERROR("[CalcLoopMaxTime] unitSize is 0, opType=%s.", opType.c_str());
        return ge::GRAPH_FAILED;
    }
    if (opType == HCCL_KERNEL_OP_TYPE_ALLREDUCE || opType == HCCL_KERNEL_OP_TYPE_BROADCAST
        || opType == HCCL_KERNEL_OP_TYPE_REDUCE || opType == HCCL_KERNEL_OP_TYPE_SEND
        || opType == HCCL_KERNEL_OP_TYPE_RECEIVE) {
        // 组 1: input buf 单份容量
        maxCountPerLoop = cclBuffSize / unitSize;
    } else if (opType == HCCL_KERNEL_OP_TYPE_REDUCESCATTER || opType == HCCL_KERNEL_OP_TYPE_ALLGATHER) {
        if (rankSize == 0) {
            HCCL_ERROR("[CalcLoopMaxTime] rankSize is 0, opType=%s.", opType.c_str());
            return ge::GRAPH_FAILED;
        }
        // 组 2: buf 需容纳 rankSize 份数据
        maxCountPerLoop = cclBuffSize / (static_cast<uint64_t>(rankSize) * unitSize);
    } else {
        // 不 loop (AllGatherV/ReduceScatterV/AlltoAll 系等): 不走 CCL buf 分迭代
        return ge::GRAPH_SUCCESS;
    }
    if (maxCountPerLoop == 0) {
        // CCL buf 装不下单迭代最小数据
        HCCL_ERROR(
            "[CalcLoopMaxTime] cclBuffSize=%lu cannot hold one iteration, opType=%s.", cclBuffSize, opType.c_str());
        return ge::GRAPH_FAILED;
    }
    // 出参保持 u32 (对照 InfoStore u32& loopMaxTime, BuildTagVec/tags 索引同宽);
    // u64 商到 u32 的窄化显式受检, 超限时报错而非静默截断 — 保住 tags.size() == 实际迭代数的不变式
    const uint64_t loopCount = (count + maxCountPerLoop - 1) / maxCountPerLoop;
    if (loopCount > UINT32_MAX) {
        HCCL_ERROR("[CalcLoopMaxTime] loop count %lu exceeds uint32 range, opType=%s.", loopCount, opType.c_str());
        return ge::GRAPH_FAILED;
    }
    loopMaxTime = static_cast<uint32_t>(loopCount);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
BuildTagVec(const std::string& tagSeed, const char* group, uint32_t loopMaxTime, std::vector<std::string>& tags)
{
    // 对应 InfoStore GenerateOpTagFromTaskInfo: 调 HcomGenerateCclOpTag 生成 base tag
    // tagSeed 语义见头文件: 集合通信类为 opType, P2P (Send/Recv) 为预构 baseTag 串
    char cTag[CCL_OP_TAG_MAX_LEN] = {0};
    HcclResult ret
        = HcomGenerateCclOpTag(tagSeed.c_str(), static_cast<s64>(CommNumHcom::COMM_VALUE_DEFAULT), group, cTag);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[BuildTagVec] HcomGenerateCclOpTag failed for tagSeed '%s', ret=%d.", tagSeed.c_str(),
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    std::string tag = cTag;

    // 对应 InfoStore GetTagVectorInfo: loopMaxTime==0 → {tag}; loopMaxTime>0 → {tag + "_looptime_" + i}
    tags.clear();
    if (loopMaxTime == 0) {
        tags.push_back(tag);
    } else {
        tags.reserve(loopMaxTime);
        for (uint32_t i = 0; i < loopMaxTime; i++) {
            tags.push_back(tag + "_looptime_" + std::to_string(i));
        }
    }

    HCCL_INFO(
        "[BuildTagVec] tagSeed=%s tag=%s loopMaxTime=%u tags.size=%zu.", tagSeed.c_str(), tag.c_str(), loopMaxTime,
        tags.size());
    return ge::GRAPH_SUCCESS;
}

// 环境推导的 CCL buf 可用尺寸 — 对照 InfoStore GetCCLBufferAvailableSize (ge 仓 hcom_base_funs.cc,
// 常量 op_hcom_comm.h): size = HCCL_BUFFSIZE 环境变量(MB, 缺省 200) × 1MB - 1MB 保留值, 与通信域状态无关。
// kernel_builder 与 graph_optimizer 的 opParam 组装 (SetOpParamGraphModeHCCLBufferSize) 均以
// 此值为输入; 与 tag/loop 用的实际 in-CCL buffer 尺寸 (GetHcomInCCLbufferSize / 基类 GetOrCreateInCCLbuf) 是两个来源。
ge::graphStatus GetCCLBufferAvailableSize(uint64_t& size)
{
    constexpr uint64_t DEFAULT_BUFF_SIZE_MB = 200;                           // HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE
    constexpr uint64_t FIXED_CALC_BUFFER_SIZE = 1ULL * 1024 * 1024;          // HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE
    constexpr uint64_t INBUFFER_UNALIGNED_RESERVE_SIZE = 1ULL * 1024 * 1024; // CCL_COMM_INBUFFER_UNALIGNED_RESERVE_SIZE

    // InfoStore 注释: HCCL 校验过该环境变量, 这里直接读取
    const char* envValue = std::getenv("HCCL_BUFFSIZE");
    uint64_t buffSizeMb = DEFAULT_BUFF_SIZE_MB;
    if (envValue != nullptr) {
        char* endPtr = nullptr;
        buffSizeMb = std::strtoull(envValue, &endPtr, 10);
        // 结果范围检查 (对照 InfoStore SalStrToULong 的 INVALID_UINT 拦截):
        // 负数 / 前导空白+负数 / 溢出均回绕成超范围 u64, 一个检查全拦
        if (endPtr == envValue || *endPtr != '\0' || buffSizeMb < 2 || buffSizeMb > UINT32_MAX) {
            HCCL_ERROR("[GetCCLBufferAvailableSize] invalid HCCL_BUFFSIZE '%s'.", envValue);
            return ge::GRAPH_FAILED;
        }
    }
    size = buffSizeMb * FIXED_CALC_BUFFER_SIZE - INBUFFER_UNALIGNED_RESERVE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CalcOnlineResources(
    HcclOpState& st, uint32_t rankSize, uint64_t cclBufferAvailSize, const std::string& opType, HcclCMDType cmdType)
{
    OpParamGraphMode* opParam = nullptr;
    HcclResult ret = HcclCreateOpParamGraphMode(&opParam);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CalcOnlineResources] HcclCreateOpParamGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // setter 逐个校验返回码: 任一失败即 opParam 参数缺失, 继续计算会静默产出错误的
    // scratchMemSize / streamNum / taskNum (坏资源比显式失败更难排查)
    HcclResult setRet = HcclSetAivSelectOpParamGraphMode(opParam, st.aivCoreLimit);
    if (setRet == HCCL_SUCCESS) {
        setRet = HcclSetOpParamGraphModeOpType(opParam, opType.c_str());
    }
    if (setRet == HCCL_SUCCESS) {
        setRet = HcclSetOpParamGraphModeDataType(opParam, st.dataType);
    }
    if (setRet == HCCL_SUCCESS) {
        setRet = HcclSetOpParamGraphModeRankSize(opParam, &rankSize);
    }
    if (setRet == HCCL_SUCCESS) {
        setRet = HcclSetOpParamGraphModeDataCount(opParam, &st.count);
    }
    if (setRet == HCCL_SUCCESS) {
        setRet = HcclSetOpParamGraphModeHCCLBufferSize(opParam, &cclBufferAvailSize);
    }
    if (setRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CalcOnlineResources] HcclSetOpParamGraphMode* failed, ret=%d, opType=%s.", static_cast<int>(setRet),
            opType.c_str());
        HcclDestroyOpParamGraphMode(opParam);
        return ge::GRAPH_FAILED;
    }

    uint64_t opMemSize = 0;
    ret = HcclCalcOpResOnlineGraphMode(opParam, &opMemSize, &st.streamNum, &st.taskNum, &st.aivCoreNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CalcOnlineResources] HcclCalcOpResOnlineGraphMode failed, ret=%d.", static_cast<int>(ret));
        HcclDestroyOpParamGraphMode(opParam);
        return ge::GRAPH_FAILED;
    }

    char algName[ALG_NAME_MAX_LEN] = {0};
    ret = HcclSelectAlgGraphMode(
        st.group, st.count, st.dataType, st.reduceOp, cmdType, st.aivCoreLimit, &st.ifAiv, algName);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CalcOnlineResources] HcclSelectAlgGraphMode failed, ret=%d.", static_cast<int>(ret));
        HcclDestroyOpParamGraphMode(opParam);
        return ge::GRAPH_FAILED;
    }

    if (st.ifAiv) {
        // 取值来源: InfoStore HcomOpsKernelBuilder::HcomCalcOpRunningParam 同款 (kernel_builder.cc:906),
        // 512 = AIV workspace 字节数 (字节), 3 = AIV task 数 (个数) — 与旧路径取值一致
        constexpr uint32_t AIV_WORKSPACE_MEM_SIZE = 512;
        constexpr uint32_t AIV_TASK_NUM = 3;
        st.streamNum = 0;
        opMemSize = AIV_WORKSPACE_MEM_SIZE;
        st.taskNum = AIV_TASK_NUM;
    }

    HcclDestroyOpParamGraphMode(opParam);

    st.scratchMemSize = opMemSize;
    // st.cclBuffSize 由调用方从实际 in-CCL buffer 取得 (GetOrCreateInCCLbuf), 此处不写 —
    // 对照 InfoStore: tag/loop 用实际 buffer 尺寸, 在线计算用环境推导 available size, 两个来源

    HCCL_INFO(
        "[CalcOnlineResources] opType=%s rankSize=%u scratchMemSize=%lu streamNum=%u taskNum=%u "
        "aivCoreNum=%u ifAiv=%d algName=%s cclBufferAvailSize=%lu.",
        opType.c_str(), rankSize, st.scratchMemSize, st.streamNum, st.taskNum, st.aivCoreNum, st.ifAiv, algName,
        cclBufferAvailSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus
CrackCleaner::Clean(HcclOpState& st, const std::vector<void*>& crackAddrs, const std::vector<uint64_t>& crackSizes)
{
    // 地址与长度两两配对: 长度不一致属调用方拼接错误, 在空检查之前拦截
    if (crackAddrs.size() != crackSizes.size()) {
        HCCL_ERROR(
            "[CrackCleaner][Clean] crackAddrs(%zu) / crackSizes(%zu) size mismatch, opType='%s'.", crackAddrs.size(),
            crackSizes.size(), st.opType);
        return ge::GRAPH_FAILED;
    }
    if (crackAddrs.empty()) {
        return ge::GRAPH_SUCCESS;
    }

    // 申请零缓冲（按最大 crack size）: 经 MallocWorkSpace 分配, 生命周期由 GE context 构造方管理
    // (MallocWorkSpace 契约: 调用者不释放); 执行期与重放期均不释放 — 零缓冲作为 D2D 清零拷贝源
    // 须跨重放存活 (语义同 .h 注释与 InfoStore CleanInterMemory 的 crackMem 成员持久化)
    auto maxIt = std::max_element(crackSizes.begin(), crackSizes.end());
    uint64_t maxSize = *maxIt;
    if (maxSize == 0) {
        return ge::GRAPH_SUCCESS;
    }
    HCCL_INFO("[CrackCleaner][Clean] alloc crack zero mem, opType='%s' size=%llu.", st.opType, maxSize);
    void* crackZeroBuf = st.ctx->MallocWorkSpace(maxSize);
    if (crackZeroBuf == nullptr) {
        HCCL_ERROR(
            "[CrackCleaner][Clean] MallocWorkSpace for crack zero buf failed, opType='%s' size=%llu.", st.opType,
            maxSize);
        return ge::GRAPH_FAILED;
    }
    std::vector<char> crackMemTemp(maxSize, 0);
    HcclResult ret = HcclRtMemSyncCopy(
        crackZeroBuf, maxSize, crackMemTemp.data(), maxSize,
        hccl::HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[CrackCleaner][Clean] HcclRtMemSyncCopy H2D failed, opType='%s', ret=%d.", st.opType,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // D2D 拷贝零到每个 crack
    for (size_t i = 0; i < crackAddrs.size(); i++) {
        if (crackSizes[i] == 0) {
            continue;
        }
        HcclResult copyRet = HcclRtMemAsyncCopy(
            crackAddrs[i], crackSizes[i], crackZeroBuf, crackSizes[i],
            hccl::HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE, static_cast<aclrtStream>(st.stream));
        if (copyRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[CrackCleaner][Clean] HcclRtMemAsyncCopy D2D failed for crack[%zu] addr=%p size=%lu, "
                "opType='%s', ret=%d.",
                i, crackAddrs[i], crackSizes[i], st.opType, static_cast<int>(copyRet));
            return ge::GRAPH_FAILED;
        }
    }

    HCCL_DEBUG("[CrackCleaner][Clean] cleared %zu cracks.", crackAddrs.size());
    return ge::GRAPH_SUCCESS;
}
// 按 shape 声明的元素数计算 counts 有效字节数:
} // namespace hccl
