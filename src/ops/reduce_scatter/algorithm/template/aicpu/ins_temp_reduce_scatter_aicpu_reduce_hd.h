/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INS_TEMP_REDUCE_SCATTER_AICPU_REDUCE_HD_H
#define INS_TEMP_REDUCE_SCATTER_AICPU_REDUCE_HD_H

#include <map>
#include <vector>
#include "alg_v2_template_base.h"
#include "executor_v2_base.h"
#include "alg_data_trans_wrapper.h"
#include "cost_model.h"

namespace ops_hccl {

// ReduceScatter 递归折半（halving-doubling）AICPU 模板（双绑定，多通道）：
// log2(N) 步 XOR 配对交换，步内归约由 DMA WRITE_REDUCE（PCIe 链路 READ_REDUCE）在传输侧完成，
// 无本地归约、无步间 barrier（每步收发 + ACK/DATA_SIGNAL 通知在同一线程队列内定序，
// ACK 链跨 rank 传递先行步完成关系）；rank 数要求为 2 的幂（非 2 幂由 CalcCostCoeff 返回空拦截）。
// 通道按 channelsPerRank_ 多流展开（对齐 InsTempReduceScatterHD 的已验证模式）：每通道一个线程，
// 数据按 CalcDataSplitByPortGroup 水平切分，镜像/交换/回搬全部按通道独立执行——单流 SQE/DbSqe
// 速率摊薄 NCH 倍，避免连跑场景单流 RTSQ 完成速率超过 ci 守护线程匹配能力导致的 SQ 塞满。
// 绑定一（Sole 单层）：InsV2ReduceScatterSoleExecutor + TopoMatchOneLevel，注册名
// AicpuReduceScatterSoleHDAicpuReduce，N=通信域 rank 数，输入片按 algIdx 索引
// （Sole 语境 localRanks 升序恒等，algIdx==userRank），输出 OSS=0 退化。
// 绑定二（Parallel level1-inter）：InsReduceScatterParallelExecutor + TopoMatchTwoLevel，注册名
// AicpuReduceScatterParallelMeshHD（level0 机内 Mesh1D + level1 跨机 HD），N=level1 组大小（机器数），
// inter1（RPT=N0，输出 OSS=SS1 写 scratch 供 intra1 消费）/inter0（RPT=1，输入为 intra0 的
// HCCL_BUFFER 输出，SCR 使用专用区）两语境。
// 寻址统一按通用五参数契约：IN[base+rpt*IRS+i*ISS]（i=algIdx）、OUT[base+rpt*ORS+myAlgIdx*OSS]。
// scratch 布局为 N 槽单缓冲（每槽 sliceSize）：镜像初始化后槽内即归约态；各步交换对为本端放弃区
// [sentStart,+dist) 与保留区 [keepStart,+dist)，二者在双端同号槽位上对称（sentStart_me==keepStart_peer），
// 发送侧 write-reduce 直接 RMW 对端保留槽。交换原语 SendRecvBatchWriteReduce 把完成通知融合进
// 最后一个数据 WQE（无独立通知 WQE）；write 模式首步发送区（放弃半区）从不被本地归约写入，
// 跳过镜像、首步交换直接从输入区读取。
class InsTempReduceScatterAicpuReduceHD : public InsAlgTemplateBase {
public:
    static constexpr TemplateProp props = {.algoType = AlgoType::HD_AICPU_REDUCE};
    InsTempReduceScatterAicpuReduceHD() = default;
    explicit InsTempReduceScatterAicpuReduceHD(
        const OpParam& param, const u32 rankId, // 传通信域的rankId，userRank
        const std::vector<std::vector<u32>>& subCommRanks);
    ~InsTempReduceScatterAicpuReduceHD() override;

    static std::vector<CostModelParam> CalcCostCoeff(CalcCostCoeffParam param);

    std::string Describe() const override
    {
        std::string info = "Template of reduce scatter HDAicpuReduce with tempRankSize ";
        info += std::to_string(templateRankSize_);
        return info;
    }

    HcclResult CalcRes(
        HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
        AlgResourceRequest& resourceRequest) override;
    HcclResult GetRes(AlgResourceRequest& resourceRequest) const override;
    HcclResult KernelRun(
        const OpParam& param, const TemplateDataParams& tempAlgParams, TemplateResource& templateResource) override;
    u64 CalcScratchMultiple(BufferType inBuffType, BufferType outBuffType) override;
    u64 GetThreadNum() const override;

    void GetNotifyIdxMainToSub(std::vector<u32>& notifyIdxMianToSub) override;
    void GetNotifyIdxSubToMain(std::vector<u32>& notifyIdxSubToMain) override;

private:
    HcclResult HandleSingleRank(const std::vector<ThreadHandle>& threads);
    HcclResult SingleRankCopy(const std::vector<ThreadHandle>& threads, u64 rpt);
    HcclResult LocalDataCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx);
    HcclResult RunHD(const std::vector<ThreadHandle>& threads, u32 channelIdx);
    HcclResult PostLocalCopy(const std::vector<ThreadHandle>& threads, u32 channelIdx);
    u64 RptScratchBase(u64 rpt) const;

    TemplateDataParams tempAlgParams_;
    std::map<u32, std::vector<ChannelInfo>> channels_;
    std::vector<u64> sizeOut_;
    std::vector<u64> elemOffset_;
    std::vector<u32> notifyIdxMainToSub_;
    std::vector<u32> notifyIdxSubToMain_;
    u32 threadNum_{0};
    u64 dataTypeSize_{0};
    bool isDmaRead_{false};
};

} // namespace ops_hccl

#endif // INS_TEMP_REDUCE_SCATTER_AICPU_REDUCE_HD_H
