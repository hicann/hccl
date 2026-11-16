/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_RHD_DPU_INTER_H
#define INS_TEMP_REDUCE_SCATTER_RHD_DPU_INTER_H

#include <map>
#include <vector>
#include "alg_v2_template_base.h"
#include "alg_v2_template_register.h"
#include "alg_param.h"
#include "executor_base.h"
#include "dpu_alg_data_trans_wrapper.h"
#include "dpu_alg_nhr_opt_wrapper.h" // DpuTransferCtx（BuildDpuStepCtx 签名）
#include "alg_data_trans_wrapper.h"

namespace ops_hccl {

// DPU 单步五相位切片构造入参（自 DPUKernelRun 抽出，仅数据传递无逻辑）
struct DpuStepCtxParam {
    const RhdStepInfo* step{nullptr};
    const RhdPart1Info* info{nullptr};
    const std::vector<u64>* unitSizes{nullptr};
    u64 dataTypeSize{0};
    void* localCclBuffAddr{nullptr};
    void* remoteCclBuffAddr{nullptr};
    u64 accRegionBytes{0};
    u64 rptNum{1};
    u64 scratchStrideBase{0};
};

// ReduceScatter RHD（递归折半倍增，支持非 2 次幂）host-DPU inter 模板（host RoCE 逐对端链路）。
// 参数契约：allRank* 变长分片（需求 5.2）：allRankSliceSize[j]=rank j 份额大小（须全 rank 一致
// 全局表）、allRankDispls[j]=输入片位移（本地布局）、allRankProcessedDataCount[j]；模板不消费
// sliceSize/sliceStride 寻址（均匀退化：size[j]≡sliceSize、displs[j]=j·ISS）。σ unit 尺寸经
// RhdBuildUnitSizes 反查、段偏移/长度经 RhdSegToBytesAllRank 推导（需求 5.1.1）。count==0 份额
// 跳过通信（零字节段不产 WQE，握手仍完成）。
// 交换机制——DPU 只搬数：scratch 为 acc+recv 双区（2×Σunit），PRE/MAIN 对端 plain write 落 recv、
// 每步后 AICPU LocalReduce 累加进 acc；POST 例外：代表覆盖写直达被合并方 acc shareSlot。
// 老路径步进协议（新路径 DPU Thread 四步握手预留门控分支位，待 hcomm 线程级 notify 原语）：
// 镜像 → batch 切换（End+Sync）→ 每步【DPURunInfo 序列化（携步骤游标）→ HcommSendRequest → DPU 侧
// 按名分发重算步骤表并校验 → DpuBatchTransfer（STEP_SYNC → 单条大块写 + DATA_SIGNAL → Drain/Fence）
// → HcommWaitResponse → LocalReduce → 完成屏障】→ 终态回搬（rpt 递减序）→ 回 batch。
// 每步每对端 1 条 WQE（合并发送硬约束）；hostdpu 组网每对端 1 通道取 channels.at(peer)[0]；建链走
// CalcChannelRequestNhrHighestHostRoce（只认 ROCE+HOST）。preferred-only：topoCustomCheck 恒 false +
// 占位 cost 恒非空。绑定：二级 L1 / 三级 L2（Dpu 前缀，L0 均为 MeshConcur 绕路）；executor fill 提供
// stride（device RHD 消费），allRank* 表由本模板在自身 KernelRun 入口自推导（inputSliceStride 口径），
// 与共享 executor 的 NhrDpu L1 各管各口径、互不污染，均匀退化下两套逐字节等价。
// 复用开关：TemplateDataParams.rhdSkipPost 置位时 POST 不入扁平步骤表、被合并方跳过终态回搬
// ——供 Reduce/AllReduce 等上层算子复用 RS 相位（两侧同值构建，rhdStepCnt 校验不受影响）。
// skipPost 口径：本模板（DPU inter 步骤表 + 逐步游标协议）「POST 不入表」与 device 模板「POST
// 动作门控」机制不同但行为等价（均跳过 POST 交换 + 被合并方终态回搬，见 template_utils.h
// rhdSkipPost 字段注释）。当前无 executor 置位，skipPost 分支为预留不可达代码。
class InsTempReduceScatterRhdDpuInter : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::RHD};
    // L2 scratch 为 acc+recv 双区（CalcScratchMultiple 返回 2·templateRankSize_）：DPU plain write 无
    // 随路归约，对端数据必须落独立 recv 区、不能与 L0/L1 原地规约区重叠。executor 据此在编排期给
    // L2 分派 "+m2" 隔离预算与 hcclBuffBaseOff（R·C 基址、全 rank 同值）。
    static constexpr bool kNeedsAccRecvScratchIsolation = true;
    InsTempReduceScatterRhdDpuInter() = default;
    explicit InsTempReduceScatterRhdDpuInter(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempReduceScatterRhdDpuInter() override;

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter RHD (host-DPU inter) with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBufferType, BufferType outBufferType) override;
    HcclResult DPUKernelRun(
        const TemplateDataParams& tempAlgParams, const std::map<u32, std::vector<ChannelInfo>>& channels,
        const u32 myRank, const std::vector<std::vector<uint32_t>>& subCommRanks) override;

    // 仅主流（slaveThreadNum=0、无 notify）：主线程完成 AICPU 侧镜像/归约/回搬与 DPU 请求/响应
    void GetNotifyIdxMainToSub(std::vector<u32>&) override {} // 去未用形参名
    void GetNotifyIdxSubToMain(std::vector<u32>&) override {}

private:
    HcclResult HandleSingleRank(const std::vector<ThreadHandle>& threads);
    HcclResult SingleRankCopy(const std::vector<ThreadHandle>& threads, u64 rpt);
    HcclResult MirrorToScratch(const std::vector<ThreadHandle>& threads); // 阶段一：全量镜像（σ 布局，升序）
    HcclResult
    RunStepsOldPath(const OpParam& param, const TemplateResource& templateResource); // 阶段二~四：老路径步进交替
    HcclResult
    StepLocalReduce(const std::vector<ThreadHandle>& threads, const RhdStepInfo& step); // 步间 AICPU 归约（recv → acc）
    HcclResult PostLocalCopy(const std::vector<ThreadHandle>& threads); // 阶段五：终态回搬（rpt 递减序）
    u64 RptScratchBase(u64 rpt) const; // 每 repeat 块基址（2p·SS 跨度，acc+recv 双区）
    // DPU 单步切片构造（每相位每 rpt 至多各一条 tx/rx 大块；零字节段不产 WQE）
    // 自 DPUKernelRun 抽出
    HcclResult BuildDpuStepCtx(DpuTransferCtx& ctx, const DpuStepCtxParam& p);

    // 扁平步骤表（本 rank 视角）：[PRE（仅 inPart1）， MAIN×log₂P（被合并方跳过）， POST（仅
    // inPart1 且 !skipPost）]。AICPU/DPU 两侧同源重算（template_utils RHD 函数族），DPU 侧以
    // rhdStepCnt 校验两表一致；skipPost 由 TemplateDataParams.rhdSkipPost 透传（两侧同值，
    // Reduce/AllReduce 复用 RS 相位时置位）。
    static HcclResult
    BuildRhdFlatSteps(u32 rankSize, u32 myAlgRank, bool skipPost, std::vector<RhdStepInfo>& flatSteps);

    // allRank* 契约辅助：σ unit 尺寸反查构建（KernelRun 期一次）与段字节推导（RhdSegToBytesAllRank，
    // 需求 5.1.1——不做模板内私有换算）；accRegionBytes_ = Σ unitSizes_（recv 区基址偏移）
    HcclResult SegBytesOfUnits(u32 unitStart, u32 unitEnd, RhdSegBytes& bytes) const;

    TemplateDataParams tempAlgParams_;
    std::map<u32, std::vector<ChannelInfo>> channels_;
    RhdPart1Info part1Info_{};
    std::vector<RhdStepInfo> flatSteps_;
    std::vector<u64> unitSizes_; // σ unit 尺寸（allRankSliceSize 反查；均匀退化时全等）
    u32 myAlgIdx_{0};
    u64 dataTypeSize_{0};
    u64 accRegionBytes_{0}; // acc 区总字节 = Σ unitSizes_（RptScratchBase 跨度 = 2×该值）
};

} // namespace ops_hccl

#endif // INS_TEMP_REDUCE_SCATTER_RHD_DPU_INTER_H
