/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 调用的 hcom.h (pkg_inc/hccl/hcom.h) 接口:
//   HcomGetCommHandleByGroup — 按 group 获取通信域 handle (hccl_custom_op.cc)
//   GetOrCreateInCCLbuf (基类) — get-or-create 取 in-CCL buffer 实际尺寸 (CalcResourcesStatic,
//                                对照 InfoStore GetHcomInCCLbufferSize; HcomGetInCCLbuffer/HcomCreateCommCCLbuffer)
//   HcomGetRankId            — 获取本 rank 的 rankId (CalcResourcesStatic, ⑤ root-only 判定)
//   HcomCreateCommCCLbuffer  — 创建通信域 CCL buffer (hccl_custom_op.cc GetOrCreateInCCLbuf/OutCCLbuf)
//   HcomGetInCCLbuffer       — 获取输入 CCL buffer (hccl_custom_op.cc GetOrCreateInCCLbuf)
//   HcomGetOutCCLbuffer      — 获取输出 CCL buffer (hccl_custom_op.cc GetOrCreateOutCCLbuf)
//
// 另调用 hccl_comm.h: HcclGetRankSize (CalcResourcesStatic)
// 另使用 hcom.h 常量: HCCL_KERNEL_OP_TYPE_REDUCE, HCCL_CMD_REDUCE, rtStream_t
// 通信调用: HcclReduceGraphMode — 本仓 src/ops/reduce/reduce.h, hccl.map 导出, extern "C" 前向声明,
//           root 参数位于 op 与 group 之间 (第 6 参)
// 公共工具 (CalcLoopMaxTime/BuildTagVec/CalcOnlineResources/CrackCleaner) 见 hccl_op_utils.h/.cc

#include "hcom_reduce_op.h"

#include <cstddef>

#include "acl/acl_rt.h"
#include "hccl.h" // 本仓源头 include/hccl.h (HcclReduce 单算子接口), 非 CANN 安装产物
#include "hccl/hccl_comm.h"
#include "hccl/hcom.h"
#include "hccl_op_utils.h"
#include "hccl_rt_utils.h"
#include "log.h"

// 前向声明, 避免引入内部头文件依赖链 (签名对照本仓 src/ops/reduce/reduce.h)
extern "C" {
HcclResult HcclSetAivCoreLimitGraphMode(const char* group, uint32_t aivCoreLimit);
HcclResult HcclReduceGraphMode(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root,
    const char* group, aclrtStream stream, const char* tag, void** streams, size_t streamCount, void* scratchMemAddr,
    uint64_t scratchMemSize);
}

// ExtractParams 公共校验: 首 input/attrs 判空、root_rank/reduction/dataType 解析 (静态/动态路径共用)
static ge::graphStatus ExtractCommonParams(
    hccl::HcclOpState& st, const char* fnName, const gert::RuntimeAttrs*& attrs, const char*& reduction,
    const char*& group)
{
    const gert::Tensor* firstInput = st.ctx->GetInputTensor(HcomReduceOp::INPUT_INDEX);
    if (firstInput == nullptr) {
        HCCL_ERROR("[HcomReduceOp][%s] input tensor is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    attrs = st.ctx->GetAttrs();
    if (attrs == nullptr) {
        HCCL_ERROR("[HcomReduceOp][%s] attrs is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    const int64_t* rootRank = attrs->GetInt(HcomReduceOp::ATTR_ROOT_RANK);
    if (rootRank == nullptr) {
        HCCL_ERROR("[HcomReduceOp][%s] root_rank attr is missing.", fnName);
        return ge::GRAPH_FAILED;
    }
    if (*rootRank < 0 || *rootRank > static_cast<int64_t>(UINT32_MAX)) {
        // 上界防 static_cast<uint32_t> 截断: 超界值截断后可能落入 [0, rankSize) 变成"合法"root
        HCCL_ERROR(
            "[HcomReduceOp][%s] invalid root_rank %ld, out of [0, %lu].", fnName, *rootRank,
            static_cast<unsigned long>(UINT32_MAX));
        return ge::GRAPH_FAILED;
    }
    reduction = attrs->GetStr(HcomReduceOp::ATTR_REDUCTION);
    group = attrs->GetStr(HcomReduceOp::ATTR_GROUP);
    if (reduction == nullptr) {
        HCCL_ERROR("[HcomReduceOp][%s] reduction attr is null.", fnName);
        return ge::GRAPH_FAILED;
    }
    if (group == nullptr) {
        HCCL_ERROR("[HcomReduceOp][%s] group attr is null.", fnName);
        return ge::GRAPH_FAILED;
    }

    st.dataType = hccl::GeDataTypeToHccl(firstInput->GetDataType());
    if (st.dataType == HCCL_DATA_TYPE_RESERVED) {
        HCCL_ERROR("[HcomReduceOp][%s] unsupported data type %d.", fnName, static_cast<int>(firstInput->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    st.reduceOp = hccl::StringToReduceOp(reduction);
    if (st.reduceOp == HCCL_REDUCE_RESERVED) {
        HCCL_ERROR("[HcomReduceOp][%s] unsupported reduction '%s'.", fnName, reduction ? reduction : "(null)");
        return ge::GRAPH_FAILED;
    }
    st.root = static_cast<uint32_t>(*rootRank);
    return ge::GRAPH_SUCCESS;
}

// 静态路径 input 遍历: 每 input 分配输出张量, count 走 InfoStore CalcCommonCount 默认分支口径
// (每 input 的 shapeSize*dataTypeSize 向上对齐 512B 累加再除以 dataTypeSize; 与 AllReduce 的
// CalcCountForAlignedOp 口径差异: 基于 GetShape().GetShapeSize() 而非 GetSize()), 并记录融合缝隙 (crack)
static ge::graphStatus AllocOutputsAndCalcCracks(hccl::HcclOpState& st, size_t& inputCount)
{
    constexpr uint64_t ALIGNED_SIZE = 512;
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    uint64_t totalSize = 0;
    inputCount = 0;

    for (size_t i = 0;; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        if (inTensor == nullptr) {
            break;
        }
        inputCount++;

        gert::Tensor* outTensor
            = st.ctx->MallocOutputTensor(i, inTensor->GetShape(), inTensor->GetFormat(), inTensor->GetDataType());
        if (outTensor == nullptr) {
            HCCL_ERROR("[HcomReduceOp][ExtractParamsStatic] failed to allocate output tensor %zu.", i);
            return ge::GRAPH_FAILED;
        }

        // crackOffset = 当前 tensor 在融合 buffer 中的偏移 + 实际数据大小
        // crackSize = 512B 对齐后大小 - 实际数据大小 (padding 缝隙)
        uint64_t shapeSize = static_cast<uint64_t>(inTensor->GetShapeSize());
        uint64_t inputSize = shapeSize * unitSize;
        uint64_t blockSize = (inputSize + ALIGNED_SIZE - 1) / ALIGNED_SIZE * ALIGNED_SIZE;
        if (blockSize > inputSize) {
            st.crackOffsets.push_back(totalSize + inputSize);
            st.crackSizes.push_back(blockSize - inputSize);
        }

        totalSize += blockSize;
        st.inputPtrs.push_back(const_cast<void*>(inTensor->GetAddr()));
        st.outputPtrs.push_back(outTensor->GetAddr());

        HCCL_INFO(
            "[HcomReduceOp][ExtractParamsStatic] input[%zu] addr=%p output addr=%p shapeSize=%lu "
            "inputSize=%lu(B) blockSize=%lu(B).",
            i, inTensor->GetAddr(), outTensor->GetAddr(), shapeSize, inputSize, blockSize);
    }

    if (inputCount == 0) {
        HCCL_ERROR("[HcomReduceOp][ExtractParamsStatic] no input tensors.");
        return ge::GRAPH_FAILED;
    }
    st.count = totalSize / unitSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceOp::ExtractParamsStatic(hccl::HcclOpState& st)
{
    const gert::RuntimeAttrs* attrs = nullptr;
    const char* reduction = nullptr;
    const char* group = nullptr;
    HCCL_GE_CHK_RET(ExtractCommonParams(st, "ExtractParamsStatic", attrs, reduction, group));

    size_t inputCount = 0;
    HCCL_GE_CHK_RET(AllocOutputsAndCalcCracks(st, inputCount));

    st.group = group;
    st.stream = st.ctx->GetStream();

    const int64_t* fusion = attrs->GetInt(ATTR_FUSION);
    const int64_t* fusionId = attrs->GetInt(ATTR_FUSION_ID);

    HCCL_INFO(
        "[HcomReduceOp][ExtractParamsStatic] inputCount=%zu input[0]=%p output[0]=%p count=%lu dataType=%d "
        "reduceOp=%d root=%u group=%s stream=%p fusion=%ld fusionId=%ld.",
        inputCount, st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.reduceOp, st.root,
        st.group ? st.group : "(null)", st.stream, fusion ? *fusion : -1, fusionId ? *fusionId : -1);

    return ge::GRAPH_SUCCESS;
}

// 动态路径 input 遍历: 每 input 分配输出张量并取独立 count (不对齐, lowering GetCountByShape 口径),
// 校验 input/output shape 一致后存入 inputCounts
static ge::graphStatus AllocOutputsAndCalcCounts(hccl::HcclOpState& st, size_t& inputCount)
{
    inputCount = 0;
    for (size_t i = 0;; i++) {
        const gert::Tensor* inTensor = st.ctx->GetInputTensor(i);
        if (inTensor == nullptr) {
            break;
        }
        inputCount++;

        gert::Tensor* outTensor
            = st.ctx->MallocOutputTensor(i, inTensor->GetShape(), inTensor->GetFormat(), inTensor->GetDataType());
        if (outTensor == nullptr) {
            HCCL_ERROR("[HcomReduceOp][ExtractParamsDynamic] failed to allocate output tensor %zu.", i);
            return ge::GRAPH_FAILED;
        }

        uint64_t count = static_cast<uint64_t>(inTensor->GetShapeSize());
        uint64_t outCount = static_cast<uint64_t>(outTensor->GetShapeSize());
        if (count != outCount) {
            HCCL_ERROR(
                "[HcomReduceOp][ExtractParamsDynamic] input[%zu] count=%lu != output count=%lu.", i, count, outCount);
            return ge::GRAPH_FAILED;
        }
        st.inputCounts.push_back(count);
        st.inputPtrs.push_back(const_cast<void*>(inTensor->GetAddr()));
        st.outputPtrs.push_back(outTensor->GetAddr());

        HCCL_INFO(
            "[HcomReduceOp][ExtractParamsDynamic] input[%zu] addr=%p output[%zu] addr=%p count=%lu.", i,
            inTensor->GetAddr(), i, outTensor->GetAddr(), count);
    }

    if (inputCount == 0) {
        HCCL_ERROR("[HcomReduceOp][ExtractParamsDynamic] no input tensors.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceOp::ExtractParamsDynamic(hccl::HcclOpState& st)
{
    // 动态路径 (lowering-style): count 取 input[0] 的 shape (GetCountByShape 口径, 不 512B 对齐), 不计算 crack
    // 对照 lowering HcomReduceKernel: inputNum < 1 报错, 允许多 input, 仅 input[0] 参与通信 (其余与 lowering
    // 一致地忽略, 不做 batch slicing)
    const gert::RuntimeAttrs* attrs = nullptr;
    const char* reduction = nullptr;
    const char* group = nullptr;
    HCCL_GE_CHK_RET(ExtractCommonParams(st, "ExtractParamsDynamic", attrs, reduction, group));

    size_t inputCount = 0;
    HCCL_GE_CHK_RET(AllocOutputsAndCalcCounts(st, inputCount));

    // 仅 input[0] 参与通信 (对照 lowering HcomLaunchReduceKernel: inputAddrs[0]/outputAddrs[0]/inputShapes[0])
    st.count = st.inputCounts[0];
    st.group = group;
    st.stream = st.ctx->GetStream();

    HCCL_INFO(
        "[HcomReduceOp][ExtractParamsDynamic] inputCount=%zu commCount(input[0])=%lu dataType=%d reduceOp=%d "
        "root=%u group=%s stream=%p.",
        inputCount, st.count, st.dataType, st.reduceOp, st.root, st.group ? st.group : "(null)", st.stream);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceOp::CalcResourcesStatic(hccl::HcclOpState& st)
{
    uint32_t rankSize = 0;
    HcclResult ret = HcclGetRankSize(st.comm, &rankSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomReduceOp][CalcResourcesStatic] HcclGetRankSize failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    // root 上界校验: 旧图路径由图编译上游保证 root < rankSize, eager 路径无该环节,
    // 在 rankSize 可得处显式拦截 (root 越界时 ⑤ root-only 回写可能静默不执行)
    if (st.root >= rankSize) {
        HCCL_ERROR(
            "[HcomReduceOp][CalcResourcesStatic] root %u out of range [0, %u), group '%s'.", st.root, rankSize,
            st.group ? st.group : "(null)");
        return ge::GRAPH_FAILED;
    }
    st.rankSize = rankSize;

    // 本 rank 的 rankId 在此阶段取得 (对照 InfoStore HcomReduceLoop 的 HcomGetRankId), 供 ⑤ root-only 判定
    ret = HcomGetRankId(st.group, &st.rankId);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceOp][CalcResourcesStatic] HcomGetRankId failed for group '%s', ret=%d.", st.group,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // st.cclBuffSize: 实际 in-CCL buffer 尺寸 — 仅 Refresh 路径消费 (GenerateTagVec 的 loopMaxTime 与
    // LaunchRefresh 的 maxCountPerLoop, 两处均门控 needRefresh); Direct 直传用户 buffer 不经 CCL 中转,
    // 无需创建 (旧路径无 Direct 分支恒走中转循环, 拆分后收敛到 Refresh)。get-or-create 语义对照
    // InfoStore GetHcomInCCLbufferSize (查→建→重查)
    if (st.needRefresh) {
        void* cclBufPtr = nullptr;
        HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, cclBufPtr, st.cclBuffSize));
    }

    // 在线计算输入: 环境推导 available size — 对照 InfoStore GetCCLBufferAvailableSize
    // (kernel_builder 与 graph_optimizer 的 SetOpParam 组装同源), 与实际 buffer 尺寸是两个来源
    uint64_t cclBufferAvailSize = 0;
    HCCL_GE_CHK_RET(hccl::GetCCLBufferAvailableSize(cclBufferAvailSize));

    return hccl::CalcOnlineResources(st, rankSize, cclBufferAvailSize, HCCL_KERNEL_OP_TYPE_REDUCE, HCCL_CMD_REDUCE);
}

ge::graphStatus HcomReduceOp::GenerateTagVec(hccl::HcclOpState& st)
{
    // 对应 InfoStore GenerateOpTagFromTaskInfo: needRefresh 时算 loopMaxTime
    // loopMaxTime 与 LaunchRefresh 循环同源推导 (公共 hccl::CalcLoopMaxTime): maxCountPerLoop =
    // st.cclBuffSize / unitSize (单一来源), loopMaxTime = ceil(count / maxCountPerLoop),
    // 保证 tags.size() 恒等于实际迭代数。Reduce 属组 1 公式 (等长, 膨胀系数 1), 不消费 rankSize, 惯例传 1
    uint32_t loopMaxTime = 0;
    if (st.needRefresh) {
        uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
        HCCL_GE_CHK_RET(
            hccl::CalcLoopMaxTime(HCCL_KERNEL_OP_TYPE_REDUCE, st.count, unitSize, st.cclBuffSize, 1, loopMaxTime));
    }

    // 对应 InfoStore GetTagVectorInfo: loopMaxTime==0 → {tag}; loopMaxTime>0 → {tag + "_looptime_" + i}
    return hccl::BuildTagVec(HCCL_KERNEL_OP_TYPE_REDUCE, st.group, loopMaxTime, st.tags);
}

ge::graphStatus HcomReduceOp::LaunchHcclOpStatic(hccl::HcclOpState& st)
{
    // 静态路径 (InfoStore-style): needRefresh 判定已收编基类 Execute, 此处只消费 st.needRefresh
    if (st.scratchMemSize > 0) {
        st.scratchMem = st.ctx->MallocWorkSpace(st.scratchMemSize);
        if (st.scratchMem == nullptr) {
            HCCL_ERROR("[HcomReduceOp][LaunchHcclOpStatic] MallocWorkSpace failed, size=%lu.", st.scratchMemSize);
            return ge::GRAPH_FAILED;
        }
    }

    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceOp][LaunchHcclOpStatic] HcclSetAivCoreLimitGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // needRefresh 判定已收编基类 Execute
    HCCL_GE_CHK_RET(GenerateTagVec(st));

    // attach 流申请 + 交付 (同 AllReduce)
    // follow 流申请 (按 st.streamNum, 复刻旧 main_stream_id 池化; AIV/流数 0 自动跳过)
    HCCL_GE_CHK_RET(RequestFollowStreams(st));
    HCCL_GE_CHK_RET(RequestAttachStream(st));
    HCCL_GE_CHK_RET(RegisterUnfoldStreamToModel(st));
    if (!st.needRefresh) {
        return LaunchDirect(st);
    }
    return LaunchRefresh(st);
}

ge::graphStatus HcomReduceOp::LaunchHcclOpDynamic(hccl::HcclOpState& st)
{
    // 动态路径 (lowering-style): 对照 lowering HcomReduceKernel/HcomLaunchReduceKernelV2 —
    // 无 batch slicing (仅 input[0] 参与通信), 直传用户 buffer, 无 offline 资源计算和 scratchMem。
    // 单算子接口 HcclReduce (无 tag/scratchMem), 对照 lowering HcomReduceKernel 的 HcceReduce →
    // dlsym "HcclReduce"; AIV 算力上限用 GraphMode 版 (对照 V2 kernel HcceSetAivCoreLimitGraphMode)
    HcclResult ret = HcclSetAivCoreLimitGraphMode(st.group, st.aivCoreLimit);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceOp][LaunchHcclOpDynamic] HcclSetAivCoreLimitGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    HCCL_INFO(
        "[HcomReduceOp][LaunchHcclOpDynamic] input=%p output=%p count=%lu root=%u (direct, user buffers).",
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.root);

    ret = HcclReduce(
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.reduceOp, st.root, st.comm,
        static_cast<aclrtStream>(st.stream));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomReduceOp][LaunchHcclOpDynamic] HcclReduce failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceOp::LaunchDirect(hccl::HcclOpState& st)
{
    HCCL_INFO(
        "[HcomReduceOp][LaunchDirect] input=%p output=%p count=%lu root=%u crackNum=%zu (non-refresh, user "
        "buffers directly).",
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.root, st.crackOffsets.size());

    // 直连无 CCL buf 中转, 不做超限拦截 — 对照 InfoStore direct 分支 (INFO.cc) 同样无
    // CheckHcomOpMemSize (该检查仅挂 Loop 路径入口), 任意 size 由 HcclReduceGraphMode 底层处理

    // Direct 路径: crack 地址 = 用户 buffer 基址 + crackOffset（inputOffset=0）
    // 对应 InfoStore GetCrackParamsInfoFromTaskInfo: crackAddr[i] = crackOffset[i] - inputOffset + inputAddr
    std::vector<void*> crackAddrs;
    std::vector<uint64_t> crackSizes;
    for (size_t i = 0; i < st.crackOffsets.size(); i++) {
        crackAddrs.push_back(static_cast<char*>(st.inputPtrs[0]) + st.crackOffsets[i]);
        crackSizes.push_back(st.crackSizes[i]);
    }
    HCCL_GE_CHK_RET(crackCleaner_.Clean(st, crackAddrs, crackSizes));

    HcclResult ret = HcclReduceGraphMode(
        st.inputPtrs[0], st.outputPtrs[0], st.count, st.dataType, st.reduceOp, st.root, st.group,
        static_cast<aclrtStream>(st.stream), st.tags[0].c_str(),
        st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
        st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcomReduceOp][LaunchDirect] HcclReduceGraphMode failed, ret=%d.", static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// LaunchRefresh 循环上下文: CCL buf、args 字段 device 地址与 crack 清理器
// (devXxx 为 args struct 各字段的 device 侧地址, D2D ADDR 经其二级指针间接寻址)
struct HcomReduceRefreshCtx {
    hccl::CrackCleaner* crackCleaner = nullptr;
    void* commInputPtr = nullptr;
    uint64_t commInputSize = 0;
    void* commOutputPtr = nullptr;
    void* devCclInputPtr = nullptr;
    void* devInputAddr = nullptr;
    void* devCclOutputPtr = nullptr;
    void* devOutputAddr = nullptr;
    void* devArgsData = nullptr;
    uint64_t outputMaxSize = 0;
    uint64_t maxCountPerLoop = 0;
};

// LaunchRefresh 单迭代游标
struct HcomReduceLoopCursor {
    uint64_t loopTime = 0;
    uint64_t inputOffset = 0;
    uint64_t outputOffset = 0;
    uint64_t curCount = 0;
    bool isLastLoop = false; // curCount == 剩余量, 末迭代才做缝隙清理
};

// 刷新上下文的 args 部分: maxCountPerLoop 计算、args struct 分配与字段地址换算
// (args 经 MallocReadOnlyDevArgs 分配 host+device 双份, GE 统一 H2D 同步 — 对应 InfoStore
// RefreshInputAddr/RefreshOutputAddr 的 ①④ 次拷贝; CCL buf 由调用方经 GetOrCreate 获取后填入 ctx)
static ge::graphStatus InitRefreshArgs(hccl::HcclOpState& st, HcomReduceRefreshCtx& ctx)
{
    uint64_t unitSize = hccl::GetHcclDataTypeSize(st.dataType);
    // maxCountPerLoop 与 GenerateTagVec 的 loopMaxTime 同源 (st.cclBuffSize 单一来源), tag 数恒等于迭代数
    ctx.maxCountPerLoop = st.cclBuffSize / unitSize;
    if (ctx.maxCountPerLoop == 0) {
        HCCL_ERROR(
            "[HcomReduceOp][LaunchRefresh] cclBuffSize=%lu cannot hold one element (unitSize=%lu).", st.cclBuffSize,
            unitSize);
        return ge::GRAPH_FAILED;
    }
    // ⑤ 侧 dstMax 按正确语义 count*unitSize (InfoStore HcomReduceLoop 的 count*unitSize/rankSize
    // 为疑似笔误, 不复刻)
    ctx.outputMaxSize = st.count * unitSize;

    HcomReduceArgs args = {ctx.commInputPtr, ctx.commOutputPtr, st.inputPtrs[0], st.outputPtrs[0]};
    const gert::KernelArgs* devArgs = st.ctx->MallocReadOnlyDevArgs(&args, sizeof(args));
    if (devArgs == nullptr || devArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomReduceOp][LaunchRefresh] MallocReadOnlyDevArgs failed.");
        return ge::GRAPH_FAILED;
    }
    // devArgs->args_data 是 device 侧 args 起始地址; 字段偏移: cclInputPtr=0, cclOutputPtr=8, inputAddr=16,
    // outputAddr=24
    char* devArgsBase = static_cast<char*>(devArgs->args_data);
    ctx.devArgsData = devArgs->args_data;
    ctx.devCclInputPtr = devArgsBase + offsetof(HcomReduceArgs, cclInputPtr);
    ctx.devCclOutputPtr = devArgsBase + offsetof(HcomReduceArgs, cclOutputPtr);
    ctx.devInputAddr = devArgsBase + offsetof(HcomReduceArgs, inputAddr);
    ctx.devOutputAddr = devArgsBase + offsetof(HcomReduceArgs, outputAddr);
    return ge::GRAPH_SUCCESS;
}

// ② D2D ADDR 带偏移: 用户 input → CCL input buf (srcOffset=inputOffset, dstOffset=0)
//   src = devInputAddr    (二级指针 → 解引用得用户 input 地址)
//   dst = devCclInputPtr  (二级指针 → 解引用得 CCL buf 地址)
static ge::graphStatus
CopyInputToCCL(hccl::HcclOpState& st, const HcomReduceRefreshCtx& ctx, const HcomReduceLoopCursor& cursor)
{
    const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);
    HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
        ctx.devCclInputPtr, ctx.commInputSize, 0, ctx.devInputAddr, curSize, cursor.inputOffset,
        static_cast<aclrtStream>(st.stream));
    if (hrtRet != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceOp][RefreshOneLoop] D2D ADDR input->CCL failed, loopTime=%lu, ret=%d.", cursor.loopTime,
            static_cast<int>(hrtRet));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 末迭代窗口缝隙清理: RefreshOneLoop 仅末迭代调用 (isLastLoop 门控, 对照 InfoStore
// HcomReduceLoop 的 curCount == countLeft), 清与末窗口 [inputOffset, inputOffset + curSize)
// 有交集的 crack, 地址 = commInputPtr + (交集起点 - inputOffset)。
// 对照 InfoStore 差异: 其用末迭代 inputOffset 无过滤平移全部 crack — 非末窗口 crack 平移出
// 负地址, CleanInterMemory 不检查地址直接 D2D 清零 (缓冲区前越界); 本实现的交集过滤跳过
// 非末窗口 crack, 无该问题; 钳位把写入约束在窗口内属防御性 (末窗口缝隙按几何恒不越窗口尾:
// totalSize 为 512 对齐块之和, 末窗口尾恰为 totalSize)。
// 实现与 AllReduce CleanCracksInWindow 同款; 非末窗口缝隙两侧均不清 — 与 InfoStore 一致
static ge::graphStatus
CleanCracksInWindow(hccl::HcclOpState& st, const HcomReduceRefreshCtx& ctx, uint64_t inputOffset, uint64_t curSize)
{
    std::vector<void*> crackAddrs;
    std::vector<uint64_t> crackSizes;
    const uint64_t windowEnd = inputOffset + curSize;
    for (size_t i = 0; i < st.crackOffsets.size(); i++) {
        const uint64_t gapStart = st.crackOffsets[i];
        const uint64_t gapEnd = gapStart + st.crackSizes[i];
        if (gapStart >= windowEnd || gapEnd <= inputOffset) {
            continue; // 与本窗口无交集
        }
        const uint64_t cleanStart = gapStart > inputOffset ? gapStart : inputOffset;
        const uint64_t cleanEnd = gapEnd < windowEnd ? gapEnd : windowEnd;
        crackAddrs.push_back(static_cast<char*>(ctx.commInputPtr) + (cleanStart - inputOffset));
        crackSizes.push_back(cleanEnd - cleanStart);
    }
    return ctx.crackCleaner->Clean(st, crackAddrs, crackSizes);
}

// LaunchRefresh 单迭代: ② D2D ADDR 搬入 → 末迭代窗口内 crack 清理 → ③ 通信 (root 第 6 参) →
// ⑤ D2D ADDR 回写 (仅 root rank 下发, 对照 InfoStore HcomReduceLoop 的 rankId == root 判定;
// 非 root 跳过 ⑤, args struct 仍 4 字段, outputAddr 字段闲置)
static ge::graphStatus
RefreshOneLoop(hccl::HcclOpState& st, const HcomReduceRefreshCtx& ctx, HcomReduceLoopCursor& cursor)
{
    const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);

    // tag 越界守卫, 对照 InfoStore HcomReduceLoop
    if (cursor.loopTime >= st.tags.size()) {
        HCCL_ERROR(
            "[HcomReduceOp][RefreshOneLoop] tagVec access out-of-bounds, loopTime=%lu tags.size=%zu.", cursor.loopTime,
            st.tags.size());
        return ge::GRAPH_FAILED;
    }

    HCCL_GE_CHK_RET(CopyInputToCCL(st, ctx, cursor));

    // 缝隙清理: 仅末迭代, 窗口内存在 crack 才清
    if (cursor.isLastLoop && !st.crackOffsets.empty()) {
        HCCL_GE_CHK_RET(CleanCracksInWindow(st, ctx, cursor.inputOffset, curSize));
    }

    // ③ HcclReduceGraphMode (每迭代独立 tag: _looptime_N; root 第 6 参)
    HcclResult ret = HcclReduceGraphMode(
        ctx.commInputPtr, ctx.commOutputPtr, cursor.curCount, st.dataType, st.reduceOp, st.root, st.group,
        static_cast<aclrtStream>(st.stream), st.tags[cursor.loopTime].c_str(),
        st.followStreams.empty() ? nullptr : st.followStreams.data(), st.followStreams.size(), st.scratchMem,
        st.scratchMemSize);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HcomReduceOp][RefreshOneLoop] HcclReduceGraphMode failed, loopTime=%lu, ret=%d.", cursor.loopTime,
            static_cast<int>(ret));
        return ge::GRAPH_FAILED;
    }

    // ⑤ D2D ADDR 带偏移: CCL output buf → 用户 output (dstOffset=outputOffset, srcOffset=0)
    //   src = devCclOutputPtr  (二级指针 → 解引用得 CCL buf 地址)
    //   dst = devOutputAddr    (二级指针 → 解引用得用户 output 地址)
    if (st.rankId == st.root) {
        HcclResult hrtRet = hccl::HcclRtMemAddrAsyncCopyWithOffset(
            ctx.devOutputAddr, ctx.outputMaxSize, cursor.outputOffset, ctx.devCclOutputPtr, curSize, 0,
            static_cast<aclrtStream>(st.stream));
        if (hrtRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcomReduceOp][RefreshOneLoop] D2D ADDR CCL->output failed, loopTime=%lu, ret=%d.", cursor.loopTime,
                static_cast<int>(hrtRet));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceOp::LaunchRefresh(hccl::HcclOpState& st)
{
    // 地址刷新路径: args struct + ArgsUpdater, GE 统一 H2D
    // 多迭代 loop: 结构对照 InfoStore HcomReduceLoop, 超限数据分迭代执行 (等长膨胀系数 1)
    // 单迭代 (数据不超 CCL buf) 时 inputOffset=0、curSize=全量, ②③⑤ 参数与单迭代语义逐字节等价
    HcomReduceRefreshCtx ctx;
    ctx.crackCleaner = &crackCleaner_;
    HCCL_GE_CHK_RET(GetOrCreateInCCLbuf(st.group, ctx.commInputPtr, ctx.commInputSize));
    uint64_t outCclBufCapacity = 0;
    HCCL_GE_CHK_RET(GetOrCreateOutCCLbuf(st.group, ctx.commOutputPtr, outCclBufCapacity));
    HCCL_GE_CHK_RET(InitRefreshArgs(st, ctx));

    // 大于 cclbuffer 时 count 取最大支持 count, 否则取剩余 count (等长, 膨胀系数 1)
    HcomReduceLoopCursor cursor;
    for (uint64_t countLeft = st.count; countLeft > 0; countLeft -= cursor.curCount) {
        cursor.curCount = (countLeft > ctx.maxCountPerLoop) ? ctx.maxCountPerLoop : countLeft;
        cursor.isLastLoop = (cursor.curCount == countLeft);
        HCCL_INFO(
            "[HcomReduceOp][LaunchRefresh] loopTime=%lu inputOffset=%lu outputOffset=%lu countLeft=%lu "
            "maxCountPerLoop=%lu.",
            cursor.loopTime, cursor.inputOffset, cursor.outputOffset, countLeft, ctx.maxCountPerLoop);

        HCCL_GE_CHK_RET(RefreshOneLoop(st, ctx, cursor));

        const uint64_t curSize = cursor.curCount * hccl::GetHcclDataTypeSize(st.dataType);
        cursor.inputOffset += curSize;
        cursor.outputOffset += curSize;
        cursor.loopTime++;
    }

    HCCL_INFO(
        "[HcomReduceOp][LaunchRefresh] devArgs=%p count=%lu maxCountPerLoop=%lu cclInput=%p cclOutput=%p "
        "inputAddr=%p outputAddr=%p root=%u rankId=%u.",
        ctx.devArgsData, st.count, ctx.maxCountPerLoop, ctx.commInputPtr, ctx.commOutputPtr, st.inputPtrs[0],
        st.outputPtrs[0], st.root, st.rankId);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceOp::UpdateHostArgs(gert::UpdateArgsContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReduceOp][UpdateHostArgs] update args context is null.");
        return ge::GRAPH_FAILED;
    }
    // GE 在 sink 模型每次执行前调: 刷新 args struct 中的 inputAddr/outputAddr
    // GE 统一 H2D 同步 host → device, D2D ADDR task 自动 dereference 新地址
    // (非 root rank 的 outputAddr 字段闲置, 刷新无副作用)
    const gert::Tensor* inputTensor = ctx->GetInputTensor(INPUT_INDEX);
    const gert::Tensor* outputTensor = ctx->GetOutputTensor(OUTPUT_INDEX);
    if (inputTensor == nullptr || outputTensor == nullptr) {
        HCCL_ERROR("[HcomReduceOp][UpdateHostArgs] input=%p output=%p.", inputTensor, outputTensor);
        return ge::GRAPH_FAILED;
    }

    // [DFX] 回调入口: 本次从 ctx 获取到的输入输出地址 (函数尾有 旧→新 对照)
    HCCL_INFO(
        "[HcomReduceOp][UpdateHostArgs] enter input=%p output=%p.", inputTensor->GetAddr(), outputTensor->GetAddr());

    const gert::KernelArgs* hostArgs = ctx->GetKernelArgs(gert::Placement::kPlacementHost, 0);
    if (hostArgs == nullptr || hostArgs->args_data == nullptr) {
        HCCL_ERROR("[HcomReduceOp][UpdateHostArgs] hostArgs is null.");
        return ge::GRAPH_FAILED;
    }

    auto* args = static_cast<HcomReduceArgs*>(hostArgs->args_data);
    void* prevIn = args->inputAddr; // Execute 阶段填入或上次刷新的地址
    void* prevOut = args->outputAddr;
    args->inputAddr = const_cast<void*>(inputTensor->GetAddr());
    args->outputAddr = const_cast<void*>(outputTensor->GetAddr());

    // DFX: 地址刷新链路对照 — 旧值 = Execute 阶段获取的地址(或上次刷新值), 新值 = 本次回调从 ctx 获取的地址
    // (非 root rank 的 outputAddr 字段闲置, 刷新无副作用)
    HCCL_INFO(
        "[HcomReduceOp][UpdateHostArgs] input %p -> %p, output %p -> %p.", prevIn, args->inputAddr, prevOut,
        args->outputAddr);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceOp::InferShape(gert::InferShapeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReduceOp][InferShape] infer shape context is null.");
        return ge::GRAPH_FAILED;
    }
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();
    if (inputNum != outputNum) {
        HCCL_ERROR("[HcomReduceOp][InferShape] inputNum=%zu != outputNum=%zu.", inputNum, outputNum);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < inputNum; i++) {
        const gert::Shape* inputShape = ctx->GetInputShape(i);
        if (inputShape == nullptr) {
            HCCL_ERROR("[HcomReduceOp][InferShape] input shape[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }
        gert::Shape* outputShape = ctx->GetOutputShape(i);
        if (outputShape == nullptr) {
            HCCL_ERROR("[HcomReduceOp][InferShape] output shape[%zu] is null.", i);
            return ge::GRAPH_FAILED;
        }
        *outputShape = *inputShape;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus HcomReduceOp::InferDataType(gert::InferDataTypeContext* ctx)
{
    if (ctx == nullptr) {
        HCCL_ERROR("[HcomReduceOp][InferDataType] infer datatype context is null.");
        return ge::GRAPH_FAILED;
    }
    size_t inputNum = ctx->GetComputeNodeInputNum();
    size_t outputNum = ctx->GetComputeNodeOutputNum();
    if (inputNum != outputNum) {
        HCCL_ERROR("[HcomReduceOp][InferDataType] inputNum=%zu != outputNum=%zu.", inputNum, outputNum);
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < inputNum; i++) {
        ge::DataType dataType = ctx->GetInputDataType(i);
        if (dataType == ge::DT_UNDEFINED) {
            HCCL_ERROR("[HcomReduceOp][InferDataType] input dtype[%zu] is undefined.", i);
            return ge::GRAPH_FAILED;
        }
        ge::graphStatus ret = ctx->SetOutputDataType(i, dataType);
        if (ret != ge::GRAPH_SUCCESS) {
            HCCL_ERROR("[HcomReduceOp][InferDataType] SetOutputDataType[%zu] failed.", i);
            return ret;
        }
    }
    return ge::GRAPH_SUCCESS;
}

const char* HcomReduceOp::GetOpTypeName() const { return HCCL_KERNEL_OP_TYPE_REDUCE.c_str(); }

static ge::BaseCustomOp* CreateHcomReduceOp() { return new HcomReduceOp(); }
// 条件注册: 目标平台 (950/960) 且 GE/RTS 新接口在位才注册接管, 否则回退原 hccl_engine 路径
// (hccl_custom_op.h CreateSocGatedRegister: SoC 判定 + GE/RTS 接口在位判定)
// 5 元接管注册 (kBottom/kHccl)
static ge::CustomOpCreatorRegister* g_hcom_reduce_register
    = hccl::CreateSocGatedRegister(HCCL_KERNEL_OP_TYPE_REDUCE.c_str(), CreateHcomReduceOp);
