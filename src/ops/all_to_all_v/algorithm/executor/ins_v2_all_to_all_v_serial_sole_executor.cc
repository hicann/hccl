/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include "ins_v2_all_to_all_v_serial_sole_executor.h"

#include <algorithm>
#include "alg_data_trans_wrapper.h"
#include "alg_attrs_registry.h"
#include "coll_alg_v2_exec_registry.h"
#include "ins_temp_dpu_alltoall_mesh_nda.h"
#include "topo_match_one_level.h"

namespace ops_hccl {

namespace {
    template <typename InsAlgTemplate>
    class SerialAlltoAllWindow {
    public:
        SerialAlltoAllWindow(const OpParam& param, const AlgResourceCtxSerializable& context)
            : param_(param),
              context_(context),
              rank_(context.topoInfo.userRank),
              size_(context.topoInfo.userRankSize)
        {}

        HcclResult Prepare(bool& handled)
        {
            handled = false;
            if (size_ < 2 || context_.channels.size() != 1 || context_.threads.empty()) {
                return HCCL_SUCCESS;
            }
            if (param_.opType == HcclCMDType::HCCL_CMD_ALLTOALL) {
                CHK_PTR_NULL(param_.all2AllVDataDes.sendCounts);
                const u64 fixedCount = *static_cast<const u64*>(param_.all2AllVDataDes.sendCounts);
                const u64 elementBytes = DATATYPE_SIZE_TABLE[param_.all2AllVDataDes.sendType];
                const u64 meshSlot = std::min<u64>(
                    context_.cclMem.size / 2 / size_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN, UB_MAX_DATA_SIZE);
                const bool uniform = std::all_of(
                    context_.channels[0].begin(), context_.channels[0].end(), [this](const ChannelInfo& channel) {
                        return channel.remoteRank == rank_ || channel.remoteCclMem.size == context_.cclMem.size;
                    });
                if (uniform && fixedCount <= meshSlot / elementBytes) {
                    HCCL_DEBUG(
                        "[SerialAlltoAllWindow] route[Mesh] count[%llu] slotBytes[%llu] ranks[%u] ccl[%llu]",
                        fixedCount, meshSlot, size_, context_.cclMem.size);
                    return HCCL_SUCCESS;
                }
            }
            for (const auto& channel : context_.channels[0]) {
                channels_.emplace(channel.remoteRank, &channel);
            }
            bufferBytes_ = context_.cclMem.size;
            for (u32 peer = 0; peer < size_; ++peer) {
                if (peer == rank_) {
                    continue;
                }
                auto found = channels_.find(peer);
                CHK_PRT_RET(
                    found == channels_.end(), HCCL_ERROR("[SerialAlltoAllWindow] missing rank[%u] channel", peer),
                    HCCL_E_PARA);
                bufferBytes_ = std::min<u64>(bufferBytes_, found->second->remoteCclMem.size);
            }
            halfBytes_ = bufferBytes_ / 2;
            slotBytes_ = std::min<u64>(halfBytes_ / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN, UB_MAX_DATA_SIZE);
            if (slotBytes_ == 0) {
                return HCCL_SUCCESS;
            }
            CHK_PTR_NULL(context_.cclMem.addr);
            CHK_PTR_NULL(param_.all2AllVDataDes.sendCounts);
            CHK_PTR_NULL(param_.all2AllVDataDes.recvCounts);
            CHK_PTR_NULL(param_.all2AllVDataDes.sdispls);
            CHK_PTR_NULL(param_.all2AllVDataDes.rdispls);
            sendCounts_ = static_cast<const u64*>(param_.all2AllVDataDes.sendCounts);
            recvCounts_ = static_cast<const u64*>(param_.all2AllVDataDes.recvCounts);
            sendDispls_ = static_cast<const u64*>(param_.all2AllVDataDes.sdispls);
            recvDispls_ = static_cast<const u64*>(param_.all2AllVDataDes.rdispls);
            dataType_ = param_.all2AllVDataDes.sendType;
            elementBytes_ = DATATYPE_SIZE_TABLE[dataType_];
            thread_ = context_.threads[0];
            cap_ = slotBytes_ / elementBytes_;
            if (cap_ == 0) {
                return HCCL_SUCCESS;
            }
            handled = true;
            HCCL_INFO(
                "[SerialAlltoAllWindow] rank[%u] ranks[%u] windowPeers[1] capElements[%llu] ccl[%llu]", rank_, size_,
                cap_, bufferBytes_);
            return HCCL_SUCCESS;
        }

        HcclResult Run()
        {
            CHK_RET(static_cast<HcclResult>(HcommBatchModeEnd(param_.algTag)));
            HcclResult ret = RunWindow();
            HcclResult restore = static_cast<HcclResult>(HcommBatchModeStart(param_.algTag));
            if (ret != HCCL_SUCCESS) {
                if (restore != HCCL_SUCCESS) {
                    HCCL_ERROR("[SerialAlltoAllWindow] batch mode restore failed[%d]", restore);
                }
                return ret;
            }
            return restore;
        }

    private:
        HcclResult RunSelf()
        {
            // 本端收发数量应对称；取小避免 sendCounts 大于 recvCounts 时越过 output 的 recv 槽位
            const u64 selfCount = std::min(sendCounts_[rank_], recvCounts_[rank_]);
            for (u64 processed = 0; processed < selfCount;) {
                u64 count = std::min(cap_, selfCount - processed);
                CHK_RET(LocalCopy(
                    thread_,
                    DataSlice(
                        param_.inputPtr, (sendDispls_[rank_] + processed) * elementBytes_, count * elementBytes_,
                        count),
                    DataSlice(
                        param_.outputPtr, (recvDispls_[rank_] + processed) * elementBytes_, count * elementBytes_,
                        count)));
                processed += count;
            }
            return HCCL_SUCCESS;
        }

        HcclResult RunPeer(u32 peer)
        {
            const u64 maximum = std::max(sendCounts_[peer], recvCounts_[peer]);
            for (u64 processed = 0; processed < maximum;) {
                u64 advance = std::min(cap_, maximum - processed);
                u64 send = sendCounts_[peer] > processed ? std::min(advance, sendCounts_[peer] - processed) : 0;
                u64 recv = recvCounts_[peer] > processed ? std::min(advance, recvCounts_[peer] - processed) : 0;
                const DataSlice input(
                    param_.inputPtr, (sendDispls_[peer] + std::min(processed, sendCounts_[peer])) * elementBytes_,
                    send * elementBytes_, send);
                const DataSlice output(
                    param_.outputPtr, (recvDispls_[peer] + std::min(processed, recvCounts_[peer])) * elementBytes_,
                    recv * elementBytes_, recv);
                const DataSlice sendBuffer(context_.cclMem.addr, 0, send * elementBytes_, send);
                const DataSlice recvBuffer(context_.cclMem.addr, halfBytes_, recv * elementBytes_, recv);
                CHK_RET(InsAlgTemplate::RunWindowChunk(
                    *channels_.at(peer), thread_, dataType_,
                    RunWindowChunkSlices{input, output, sendBuffer, recvBuffer}));
                processed += advance;
            }
            return HCCL_SUCCESS;
        }

        HcclResult RunWindow()
        {
            CHK_RET(RunSelf());
            u64 limit = 1;
            while (limit < size_) {
                limit <<= 1;
            }
            for (u64 step = 1; step < limit; ++step) {
                u64 peer = rank_ ^ step;
                if (peer < size_) {
                    CHK_RET(RunPeer(static_cast<u32>(peer)));
                }
            }
            return HCCL_SUCCESS;
        }

        const OpParam& param_;
        const AlgResourceCtxSerializable& context_;
        u32 rank_;
        u32 size_;
        std::map<u32, const ChannelInfo*> channels_;
        const u64* sendCounts_ = nullptr;
        const u64* recvCounts_ = nullptr;
        const u64* sendDispls_ = nullptr;
        const u64* recvDispls_ = nullptr;
        HcclDataType dataType_ = HCCL_DATA_TYPE_RESERVED;
        u64 elementBytes_ = 0;
        ThreadHandle thread_ = 0;
        u64 bufferBytes_ = 0;
        u64 halfBytes_ = 0;
        u64 slotBytes_ = 0;
        u64 cap_ = 0;
    };
} // namespace

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSerialSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    SerialAlltoAllWindow<InsAlgTemplate> operation(param, resCtx);
    bool useWindow = false;
    CHK_RET(operation.Prepare(useWindow));
    if (useWindow) {
        return operation.Run();
    }
    return RunMesh(param, resCtx);
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSerialSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    return topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs);
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSerialSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    CHK_PTR_NULL(topoInfo);
    CHK_PRT_RET(
        algHierarchyInfo.infos.empty(), HCCL_ERROR("[InsV2AlltoAllVSerialSoleExecutor] algorithm hierarchy is empty"),
        HCCL_E_PARA);
    InsAlgTemplate algTemplate(param, topoInfo->userRank, algHierarchyInfo.infos[0]);
    return algTemplate.CalcRes(comm, param, topoInfo, resourceRequest);
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSerialSoleExecutor<AlgTopoMatch, InsAlgTemplate>::PrepareMesh(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx, TemplateResource& resources,
    TemplateDataParams& data)
{
    const u32 rankSize = resCtx.topoInfo.userRankSize;
    const u64 minVectorNum = ALL_TO_ALL_V_VECTOR_NUM;
    const u64 maxVectorNum
        = param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC ? ALL_TO_ALL_VC_VECTOR_NUM : ALL_TO_ALL_V_VECTOR_NUM;
    CHK_PRT_RET(
        param.varMemSize < minVectorNum * rankSize * sizeof(u64)
            || param.varMemSize > maxVectorNum * rankSize * sizeof(u64),
        HCCL_ERROR("[InsV2AlltoAllVSerialSoleExecutor] invalid varMemSize[%llu]", param.varMemSize), HCCL_E_PARA);
    CHK_PTR_NULL(param.all2AllVDataDes.sendCounts);
    CHK_PTR_NULL(param.all2AllVDataDes.recvCounts);
    CHK_PTR_NULL(param.all2AllVDataDes.sdispls);
    CHK_PTR_NULL(param.all2AllVDataDes.rdispls);
    CHK_PRT_RET(
        resCtx.algHierarchyInfo.infos.empty() || resCtx.channels.empty(),
        HCCL_ERROR("[InsV2AlltoAllVSerialSoleExecutor] missing Mesh resources"), HCCL_E_PARA);

    resources.threads = resCtx.threads;
    for (auto channel : resCtx.channels[0]) {
        if (param.supportSymmetricMemory) {
            CHK_RET(FillChannelSymWinPeerAddrs(
                param.inputSymWindow, param.inputOffset, param.outputSymWindow, param.outputOffset, channel));
        }
        resources.channels[channel.remoteRank].push_back(channel);
    }

    data.buffInfo.inBuffType = BufferType::INPUT;
    data.buffInfo.outBuffType = BufferType::OUTPUT;
    data.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    data.buffInfo.inputPtr = param.inputPtr;
    data.buffInfo.outputPtr = param.outputPtr;
    data.buffInfo.inputSize = param.inputSize;
    data.buffInfo.outputSize = param.outputSize;
    data.buffInfo.hcclBuff = resCtx.cclMem;
    data.dataType = param.all2AllVDataDes.sendType;
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSerialSoleExecutor<AlgTopoMatch, InsAlgTemplate>::RunMesh(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    TemplateResource resources;
    TemplateDataParams data;
    CHK_RET(PrepareMesh(param, resCtx, resources, data));
    const u32 rankSize = resCtx.topoInfo.userRankSize;
    const u64 elementBytes = DATATYPE_SIZE_TABLE[data.dataType];
    InsAlgTemplate algTemplate(param, resCtx.topoInfo.userRank, resCtx.algHierarchyInfo.infos[0]);
    const u64 multiplier = algTemplate.CalcScratchMultiple(data.buffInfo.inBuffType, data.buffInfo.outBuffType);
    const u64 slotBytes
        = multiplier == 0 ?
              UB_MAX_DATA_SIZE :
              std::min<u64>(
                  UB_MAX_DATA_SIZE, resCtx.cclMem.size / multiplier / HCCL_MIN_SLICE_ALIGN * HCCL_MIN_SLICE_ALIGN);
    const u64 countPerLoop = slotBytes / elementBytes;
    CHK_PRT_RET(
        countPerLoop == 0, HCCL_ERROR("[InsV2AlltoAllVSerialSoleExecutor] Mesh slot is empty"), HCCL_E_INTERNAL);

    const auto* sendCounts = static_cast<const u64*>(param.all2AllVDataDes.sendCounts);
    const auto* recvCounts = static_cast<const u64*>(param.all2AllVDataDes.recvCounts);
    const auto* sendDispls = static_cast<const u64*>(param.all2AllVDataDes.sdispls);
    const auto* recvDispls = static_cast<const u64*>(param.all2AllVDataDes.rdispls);
    u64 maximum = 0;
    for (u32 peer = 0; peer < rankSize; ++peer) {
        maximum = std::max(maximum, std::max(sendCounts[peer], recvCounts[peer]));
    }
    CHK_PRT_RET(
        param.supportSymmetricMemory && rankSize > 1 && maximum > countPerLoop,
        HCCL_ERROR(
            "[InsV2AlltoAllVSerialSoleExecutor] symmetric Mesh count[%llu] exceeds CCL slot capacity[%llu]", maximum,
            countPerLoop),
        HCCL_E_PARA);
    const u64 loops = param.supportSymmetricMemory ? 1 : maximum / countPerLoop + (maximum % countPerLoop != 0);
    // 对称内存单轮搬运时 data.count = maximum，块大小须与 count 对齐，避免按 countPerLoop 分块导致块区间互相覆盖
    data.inputSliceStride = (param.supportSymmetricMemory ? maximum : countPerLoop) * elementBytes;
    data.outputSliceStride = maximum * elementBytes;
    data.enableRemoteMemAccess = false;
    data.repeatNum = 1;
    data.sendCounts.resize(rankSize);
    data.recvCounts.resize(rankSize);
    data.sdispls.resize(rankSize);
    data.rdispls.resize(rankSize);
    if (param.supportSymmetricMemory && param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        CHK_PTR_NULL(param.all2AllVDataDes.peerRdispls);
        const auto* peerDispls = static_cast<const u64*>(param.all2AllVDataDes.peerRdispls);
        data.peerRdispls.assign(peerDispls, peerDispls + rankSize);
    }
    u64 processed = 0;
    for (u64 loop = 0; loop < loops; ++loop) {
        data.count = loop == loops - 1 ? maximum - processed : countPerLoop;
        data.sliceSize = data.count * elementBytes;
        data.tailSize = data.sliceSize;
        data.processedDataCount = processed;
        data.buffInfo.inBuffBaseOff = processed * elementBytes;
        data.buffInfo.outBuffBaseOff = processed * elementBytes;
        for (u32 peer = 0; peer < rankSize; ++peer) {
            const u64 sent = std::min(processed, sendCounts[peer]);
            const u64 received = std::min(processed, recvCounts[peer]);
            data.sendCounts[peer] = std::min(data.count, sendCounts[peer] - sent);
            data.recvCounts[peer] = std::min(data.count, recvCounts[peer] - received);
            data.sdispls[peer] = sendDispls[peer] + sent;
            data.rdispls[peer] = recvDispls[peer] + received;
        }
        CHK_RET(algTemplate.KernelRun(param, data, resources));
        processed += data.count;
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
std::vector<CostModelParam> InsV2AlltoAllVSerialSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcCostCoeff(
    HcclComm, TopoInfoWithNetLayerDetails*, const char*, const OpParam&)
{
    return {{0.0f, 0.0f, 1.0f, 0.0f}};
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
AlgNetMeta InsV2AlltoAllVSerialSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails*, const OpParam&, const char*) const
{
    AlgNetMeta meta;
    meta.netTypes = {CommTopo::COMM_TOPO_1DMESH};
    meta.groupSizes = {1};
    meta.dataRatios = {0.0f};
    meta.rankSizes = {0};
    return meta;
}

#if CANN_VERSION_NUM >= CANN_VERSION(9, 0, 0)
REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLTOALL, DpuAllToAllSoleMeshNDA, InsV2AlltoAllVSerialSoleExecutor, TopoMatchOneLevel,
    InsTempDpuAlltoAllMeshNda);
REGISTER_ALG_ATTRS(DpuAllToAllSoleMeshNDA, topo.isSupportLevel0PcieMix = true; topo.minTopoLevelNum = TOPO_LEVEL_NUM_2;
                   topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.isHostDpuOnly = true; topo.isSupportLevel1Nhr = true;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
                   op.isSupportInplace = false;);

REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLTOALLV, DpuAllToAllVSoleMeshNDA, InsV2AlltoAllVSerialSoleExecutor, TopoMatchOneLevel,
    InsTempDpuAlltoAllMeshNda);
REGISTER_ALG_ATTRS(DpuAllToAllVSoleMeshNDA, topo.isSupportLevel0PcieMix = true; topo.minTopoLevelNum = TOPO_LEVEL_NUM_2;
                   topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3; topo.isHostDpuOnly = true; topo.isSupportLevel1Nhr = true;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
                   op.isSupportInplace = false;);

REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_ALLTOALLVC, DpuAllToAllVCSoleMeshNDA, InsV2AlltoAllVSerialSoleExecutor, TopoMatchOneLevel,
    InsTempDpuAlltoAllMeshNda);
REGISTER_ALG_ATTRS(DpuAllToAllVCSoleMeshNDA, topo.isSupportLevel0PcieMix = true;
                   topo.minTopoLevelNum = TOPO_LEVEL_NUM_2; topo.maxTopoLevelNum = TOPO_LEVEL_NUM_3;
                   topo.isHostDpuOnly = true; topo.isSupportLevel1Nhr = true;
                   topo.supportLevel0Topos = LEVEL0_TOPO_MESH_1D | LEVEL0_TOPO_MESH_1D_CLOS | LEVEL0_TOPO_CLOS;
                   op.isSupportInplace = false;);
#endif

} // namespace ops_hccl
