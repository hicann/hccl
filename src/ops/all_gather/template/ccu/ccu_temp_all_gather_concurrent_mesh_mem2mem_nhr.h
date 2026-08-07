/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TEMP_ALL_GATHER_CONCURRENT_MESH_MEM2MEM_NHR_H
#define HCCL_CCU_TEMP_ALL_GATHER_CONCURRENT_MESH_MEM2MEM_NHR_H

#include "utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_temp_all_gather_mesh_1D_mem2mem.h"
#include "ccu_temp_all_gather_nhr_1D_mem2mem.h"

namespace ops_hccl {

// 注意:以下两组 ArgLayout 常量是 kernel LoadArgs 顺序的镜像,不是 kernel 自己声明的。
// 修改 kernel 的 LoadArgs 时必须同步修改本文件。每个常量注释标明对应 kernel .cc 的 LoadArg 调用。
// 不改 kernel 的约束下,漂移只能靠注释 + 人工对齐控制,无编译期约束。

// 镜像 ccu_kernel_all_gather_mesh1d_mem2mem.cc 的 LoadArgs
struct CcuAllGatherMesh1DMem2MemArgLayout {
    static constexpr uint32_t INPUT = 0;                   // LoadArg(ctx.input)
    static constexpr uint32_t OUTPUT = 1;                  // LoadArg(ctx.output[rankId])
    static constexpr uint32_t TOKEN = 2;                   // LoadArg(ctx.token[rankId])
    static constexpr uint32_t CUR_RANK_SLICE_IN_OFF = 3;   // LoadArg(ctx.currentRankSliceInputOffset)
    static constexpr uint32_t CUR_RANK_SLICE_OUT_OFF = 4;  // LoadArg(ctx.currentRankSliceOutputOffset)
    static constexpr uint32_t TMP_REPEAT_NUM = 5;          // LoadArg(ctx.tmpRepeatNum)
    static constexpr uint32_t INPUT_REPEAT_STRIDE = 6;     // LoadArg(ctx.inputRepeatStride)
    static constexpr uint32_t OUTPUT_REPEAT_STRIDE = 7;    // LoadArg(ctx.outputRepeatStride)
    static constexpr uint32_t NORMAL_SLICE_SIZE = 8;       // LoadArg(ctx.normalSliceSize)
    static constexpr uint32_t LAST_SLICE_SIZE = 9;         // LoadArg(ctx.lastSliceSize)
    static constexpr uint32_t IS_INPUT_OUTPUT_EQUAL = 10;  // LoadArg(ctx.isInputOutputEqual)
    static constexpr uint32_t GO_SIZE_ADDR_OFFSET = 11;    // LoadArg(ctx.goSize.addrOffset)
    static constexpr uint32_t GO_SIZE_LOOP_PARAM = 12;     // LoadArg(ctx.goSize.loopParam)
    static constexpr uint32_t GO_SIZE_PARALLEL_PARAM = 13; // LoadArg(ctx.goSize.parallelParam)
    static constexpr uint32_t GO_SIZE_RESIDUAL = 14;       // LoadArg(ctx.goSize.residual)
    static constexpr uint32_t ARG_SIZE = 15;               // 实际下发给 kernel 的 arg 个数
    // 以下两个不在 kernel LoadArgs 内,仅供 FastLaunch 回放时计算地址用,由 PrepareLaunchArgs 额外缓存
    static constexpr uint32_t IN_BUFF_BASE_OFF = 15;  // buffInfo_.inBuffBaseOff
    static constexpr uint32_t OUT_BUFF_BASE_OFF = 16; // buffInfo_.outBuffBaseOff
};

// 镜像 ccu_kernel_all_gather_nhr1d_mem2mem.cc 的 LoadArgs
struct CcuAllGatherNHR1DMem2MemArgLayout {
    static constexpr uint32_t INPUT = 0;                  // LoadArg(ctx.input)
    static constexpr uint32_t OUTPUT = 1;                 // LoadArg(ctx.output[myRankIdx])
    static constexpr uint32_t TOKEN = 2;                  // LoadArg(ctx.token[myRankIdx])
    static constexpr uint32_t DIE0_SIZE = 3;              // LoadArg(ctx.die0Size)
    static constexpr uint32_t DIE1_SIZE = 4;              // LoadArg(ctx.die1Size)
    static constexpr uint32_t REPEAT_NUM = 5;             // LoadArg(ctx.repeatNum)
    static constexpr uint32_t INPUT_SLICE_STRIDE = 6;     // LoadArg(ctx.inputSliceStride)
    static constexpr uint32_t OUTPUT_SLICE_STRIDE = 7;    // LoadArg(ctx.outputSliceStride)
    static constexpr uint32_t INPUT_REPEAT_STRIDE = 8;    // LoadArg(ctx.inputRepeatStride)
    static constexpr uint32_t OUTPUT_REPEAT_STRIDE = 9;   // LoadArg(ctx.outputRepeatStride)
    static constexpr uint32_t IS_INPUT_OUTPUT_EQUAL = 10; // LoadArg(ctx.isInputOutputEqual)
    static constexpr uint32_t DIE0_LAST_SIZE = 11;        // LoadArg(ctx.die0LastSize)
    static constexpr uint32_t DIE1_LAST_SIZE = 12;        // LoadArg(ctx.die1LastSize)
    static constexpr uint32_t ARG_SIZE = 13;              // 实际下发给 kernel 的 arg 个数
    // 以下三个不在 kernel LoadArgs 内,仅供 FastLaunch 回放时计算地址用,由 PrepareLaunchArgs 额外缓存
    static constexpr uint32_t IN_BUFF_BASE_OFF = 13;  // buffInfo_.inBuffBaseOff
    static constexpr uint32_t OUT_BUFF_BASE_OFF = 14; // buffInfo_.outBuffBaseOff
    static constexpr uint32_t MY_SUB_COMM_RANK = 15;  // mySubCommRank_
};

// Concurrent template: mesh 路径 + NHR(CLOS) 路径并发执行，数据按带宽比切分。
// subCommRanks_[0] 给 mesh 子 template, subCommRanks_[1] 给 NHR 子 template。
// 线程分配: threads[0] -> mesh 主流, threads[1] -> NHR 主流, threads[2] -> NHR 从流。
// kernel 分配: ccuKernels[0] -> mesh, ccuKernels[1] -> NHR。
class CcuTempAllGatherConcurrentMeshMem2MemNHR : public CcuAlgTemplateBase {
public:
    CcuTempAllGatherConcurrentMeshMem2MemNHR() = default;
    explicit CcuTempAllGatherConcurrentMeshMem2MemNHR(
        const OpParam& param, const u32 rankId, const std::vector<std::vector<u32>>& subCommRanks);
    ~CcuTempAllGatherConcurrentMeshMem2MemNHR() override = default;

    std::string Describe() const override
    {
        return StringFormat(
            "Template of AllGather ccu mesh1d+nhr1d concurrent mem2mem with tempRankSize [%u].", templateRankSize_);
    }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& templateDataParams,
        TemplateResource& templateResource) override;
    HcclResult FastLaunch(const OpParam& param, const TemplateFastLaunchCtx& tempFastLaunchCtx) override;
    u64 GetThreadNum() const override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;

private:
    void CalcDataSplit(u64 totalCount, u64 dataTypeSize, u64& meshCount, u64& closCount) const;
    void CalcNhrDieSplit(u64 sliceSize, u64 typeSize, u64& die0Size, u64& die1Size) const;
    HcclResult BuildMeshTaskArgs(
        const OpParam& param, const TemplateDataParams& templateDataParams, u64 meshSize, u64 meshTailSize,
        std::vector<uint64_t>& meshTaskArgs);
    HcclResult BuildNhrTaskArgs(
        const OpParam& param, const TemplateDataParams& templateDataParams, u64 closSize, u64 closTailSize,
        u64 meshSize, u32 nhrKernelNum, std::vector<uint64_t>& nhrTaskArgs);
    HcclResult LaunchMeshKernel(TemplateResource& templateResource, const std::vector<uint64_t>& meshTaskArgs);
    HcclResult LaunchNhrKernels(
        TemplateResource& templateResource, const std::vector<uint64_t>& nhrTaskArgs, u32 meshKernelNum,
        u32 nhrKernelNum);
    HcclResult LaunchConcurrentKernels(
        TemplateResource& templateResource, u32 meshKernelNum, u32 nhrKernelNum, bool hasMesh, bool hasNhr,
        const std::vector<uint64_t>& meshTaskArgs, const std::vector<uint64_t>& nhrTaskArgs);
    HcclResult SaveSubmitInfos(
        TemplateResource& templateResource, const std::vector<uint64_t>& meshTaskArgs,
        const std::vector<uint64_t>& nhrTaskArgs, u64 meshSize, u32 meshKernelNum, u32 nhrKernelNum, bool hasMesh,
        bool hasNhr, const TemplateDataParams& templateDataParams);
    HcclResult PatchMeshArgs(const TemplateFastLaunchCtx& ctx);
    HcclResult PatchNhrArgs(const TemplateFastLaunchCtx& ctx, u32 meshKernelNum);

    uint32_t mySubCommRank_ = 0;
};

} // namespace ops_hccl

#endif // HCCL_CCU_TEMP_ALL_GATHER_CONCURRENT_MESH_MEM2MEM_NHR_H
