/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "ins_v2_aiv_batch_send_recv_sole_executor.h"

#include <algorithm>
#include <map>

#include "adapter_acl.h"
#include "aiv_temp_batch_send_recv_mesh_1D.h"
#include "alg_attrs_registry.h"
#include "coll_alg_v2_exec_registry.h"
#include "hccl_common.h"

namespace ops_hccl {

namespace {
    constexpr u32 AIV_BSR_SEND_TYPE = 0;
    constexpr u32 AIV_BSR_RECV_TYPE = 1;
    // 自收发拆成发/收两条 item，走与跨 rank 相同的切片与分核流水，只有数据路径不同：
    // SELF_SEND 由发方向核直接 CpGM2GM 到目的地址，SELF_RECV 只等握手 flag、不搬数据。
    constexpr u32 AIV_BSR_SELF_SEND_TYPE = 2;
    constexpr u32 AIV_BSR_SELF_RECV_TYPE = 3;
    // 桶容量与桶内片起点都按 32B 对齐（尾片长度由 item 剩余字节决定，允许非对齐）
    constexpr u64 AIV_BSR_BUCKET_ALIGN = 32;

    // tag 完全由轮次序号决定：host 侧 template 在 KernelRun 里算 sliceId = AIV_BSR_TAG_BASE + 1 + roundIdx % SPAN
    // （即自 1025 起、越界按 AIV_BSR_TAG_SPAN 回绕），kernel 只接收、不自己推导。正因如此 host 既不下发 D2D
    // 拷贝，也不手抄 512K 偏移，失败路径更无需还原计数器。
} // namespace

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfoV2(
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
{
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(topoInfo, algHierarchyInfo, algAttrs));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo,
    const AlgHierarchyInfoForAllLevel& algHierarchyInfo, AlgResourceRequest& resourceRequest)
{
    CHK_PTR_NULL(topoInfo);
    if (algHierarchyInfo.infos.empty()) {
        HCCL_ERROR("[InsV2AivBatchSendRecvSoleExecutor] algHierarchyInfo level num is zero!");
        return HCCL_E_PARA;
    }
    std::vector<std::vector<u32>> subCommRanks = algHierarchyInfo.infos[0];
    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, subCommRanks);
    CHK_RET(algTemplate->CalcRes(comm, param, topoInfo, resourceRequest));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
std::vector<CostModelParam> InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcCostCoeff(
    HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, const char* algName, const OpParam& param)
{
    (void)comm;
    (void)topoInfo;
    (void)algName;
    (void)param;
    // BatchSendRecv 无法获取其他 rank 间的通信量，不实现 costmodel，仅给出可被选中的系数
    return {{0.0f, 0.0f, 1.0f, 0.0f}};
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
AlgNetMeta InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::GetAlgNetMeta(
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& param, const char* algName) const
{
    (void)param;
    (void)algName;
    return BuildDefaultMeshNetMeta(topoInfo);
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
void InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::ResetTaskState()
{
    sendDeque_.clear();
    recvDeque_.clear();
    selfDst_.clear();
    roundItems_.clear();
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
bool InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::SortTasks(
    const HcclSendRecvItem* a, const HcclSendRecvItem* b) const
{
    if (a->remoteRank != b->remoteRank) {
        return a->remoteRank < b->remoteRank;
    }
    // 同一 pair 内按 (count 降序, dataType 降序)：两侧任务的多重集相同，排序后逐条对齐，
    // 切片时才会得到一致的 sliceIdx（收发的 flag 格号靠它匹配）。stable_sort 再兜住完全相同的两条。
    if (a->count != b->count) {
        return a->count > b->count;
    }
    return a->dataType > b->dataType;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::BuildTaskDeques(
    const HcclSendRecvItem* sendRecvInfo, u32 itemNum)
{
    HCCL_INFO("[InsV2AivBatchSendRecvSoleExecutor][BuildTaskDeques] Start split and sort the batchSendRecv tasklist.");
    CHK_PTR_NULL(sendRecvInfo);

    for (u32 i = 0; i < itemNum; i++) {
        HCCL_DEBUG(
            "[InsV2AivBatchSendRecvSoleExecutor][BuildTaskDeques] index is %u, itemNum is %u,"
            "localRankID is %u, remoteRank is %u, dataCount is %llu, sendRecvType is %u, rankSize is %u.",
            i, itemNum, myRank_, sendRecvInfo->remoteRank, sendRecvInfo->count,
            static_cast<u32>(sendRecvInfo->sendRecvType), rankSize_);
        if (sendRecvInfo->count > 0) {
            CHK_PTR_NULL(sendRecvInfo->buf);
            if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_SEND) {
                sendDeque_.push_back(sendRecvInfo);
            } else if (sendRecvInfo->sendRecvType == HcclSendRecvType::HCCL_RECV) {
                recvDeque_.push_back(sendRecvInfo);
            } else {
                HCCL_ERROR(
                    "[InsV2AivBatchSendRecvSoleExecutor][BuildTaskDeques] sendRecvType wrong sendRecvType is %d, "
                    "rankID is %u, remoteRank is %u.",
                    sendRecvInfo->sendRecvType, myRank_, sendRecvInfo->remoteRank);
                return HcclResult::HCCL_E_PARA;
            }
        }
        sendRecvInfo++;
    }

    /* 排序只需两个性质：同一 remoteRank 连续（切换时重置本轮字节数）、同 pair 组内收发顺序一致
       （两侧切出的轮次与 sliceIdx 才能对齐）。不沿用 AICPU 的 pair-wise 相位错开：AIV 收发全并发。*/
    auto taskCompare = [this](const HcclSendRecvItem* a, const HcclSendRecvItem* b) {
        return this->SortTasks(a, b);
    };

    std::stable_sort(sendDeque_.begin(), sendDeque_.end(), taskCompare);
    std::stable_sort(recvDeque_.begin(), recvDeque_.end(), taskCompare);

    // 自收发项不单独摘出：留在各自队列里与跨 rank 任务一同切片、分核、走 flag 握手，只有类型与数据路径不同。
    HCCL_INFO("[InsV2AivBatchSendRecvSoleExecutor][BuildTaskDeques] End sort the batchSendRecv tasklist.");
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CheckSelfSendRecvPairs()
{
    // 自收发不在 host 侧拷贝（haclrtMemcpy 同步、不进 stream、图捕获期失败）：打成 SELF_SEND/SELF_RECV
    // 两条 item 交给 kernel，同一条流内流序天然正确。这里只做配对校验并记下 SELF_SEND 的目的地址。
    const u32 myRank = static_cast<u32>(myRank_);
    // self 项排序后位于队列中段而非队首，故直接收集全部 self 项，组内顺序收发一致，可按下标一一配对。
    auto collectSelfItems = [myRank](const std::deque<const HcclSendRecvItem*>& dq) {
        std::vector<const HcclSendRecvItem*> selfItems;
        for (const auto* item : dq) {
            if (item->remoteRank == myRank) {
                selfItems.push_back(item);
            }
        }
        return selfItems;
    };
    const std::vector<const HcclSendRecvItem*> selfSends = collectSelfItems(sendDeque_);
    const std::vector<const HcclSendRecvItem*> selfRecvs = collectSelfItems(recvDeque_);
    CHK_PRT_RET(
        selfSends.size() != selfRecvs.size(),
        HCCL_ERROR(
            "[InsV2AivBatchSendRecvSoleExecutor][CheckSelfSendRecvPairs] self send num[%zu] != self recv num[%zu].",
            selfSends.size(), selfRecvs.size()),
        HCCL_E_PARA);
    for (size_t i = 0; i < selfSends.size(); i++) {
        const HcclSendRecvItem* sendItem = selfSends[i];
        const HcclSendRecvItem* recvItem = selfRecvs[i];
        if (sendItem->count != recvItem->count || sendItem->dataType != recvItem->dataType) {
            HCCL_ERROR("[InsV2AivBatchSendRecvSoleExecutor][CheckSelfSendRecvPairs] Send task and recv task to "
                       "self : count or dataType is not equal, please check the task list.");
            return HCCL_E_PARA;
        }
        selfDst_[sendItem] = reinterpret_cast<u64>(recvItem->buf);
    }
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcSlicesAndRounds(u64 cclCapacity)
{
    roundItems_.clear();
    // 中转区用通信域 cclBuffer，按 rankSize 均分成桶（每对端一个）。每桶上界沿用同类 AIV 算子的约定：
    // 一轮总量不超过 UB_MAX_DATA_SIZE(256MB)，即 min(256MB, cclCapacity)/rankSize，再按 32B 向下对齐。
    static_assert(
        UB_MAX_DATA_SIZE < (1ULL << 32),
        "AivBsrItemArgs uses u32 byteCount/cclOffset, UB_MAX_DATA_SIZE must fit in u32");
    // 先落到 u64 局部变量再取 min：UB_MAX_DATA_SIZE 是 uint64_t，与 u64(unsigned long long) 在 LP64 下
    // 是两个不同类型，直接 std::min 会推导失败。同类算子（AG/A2AV）也是这个写法。
    const u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    const u64 usableCapacity = std::min(cclCapacity, transportBoundDataSize);
    u64 bucketCap = usableCapacity / static_cast<u64>(rankSize_);
    // 每一个按rankSize分片的bucket 对齐32B
    bucketCap = bucketCap / AIV_BSR_BUCKET_ALIGN * AIV_BSR_BUCKET_ALIGN;
    if (usableCapacity != cclCapacity) {
        HCCL_DEBUG(
            "[InsV2AivBatchSendRecvSoleExecutor][CalcSlicesAndRounds] ccl buffer size[%llu] exceeds "
            "UB_MAX_DATA_SIZE[%llu], only the first [%llu] bytes are used, bucketCap[%llu].",
            cclCapacity, transportBoundDataSize, usableCapacity, bucketCap);
    }
    // 分桶后若不足一个对齐单位，连一片都放不下，继续跑会退化成整桶按字节搬运，直接报错退出
    CHK_PRT_RET(
        bucketCap < AIV_BSR_BUCKET_ALIGN,
        HCCL_ERROR(
            "[InsV2AivBatchSendRecvSoleExecutor][CalcSlicesAndRounds] ccl buffer size[%llu] is too small for "
            "rankSize[%u].",
            cclCapacity, rankSize_),
        HCCL_E_PARA);
    // 当前AIV_BSR_ITEM_MAX设置为512，由send和recv任务共享，且当前BatchSendRecv算子支持的最大的rankSize为256
    // 为了保证不发生rank间忙等的死锁问题，要保证每一轮每一个rank至少有一个收发任务（如果当前还有任务的话）
    // 所以针对每一个 remoteRank、每一个 itemType 都有 item 条数上限，即 AIV_BSR_ITEM_MAX / (2 * rankSize)
    const u64 itemCapPerPair = std::max<u64>(AIV_BSR_ITEM_MAX / (2 * static_cast<u64>(rankSize_)), 1);
    // flag同步区域按照rankSize分区，由于每一个任务最大item数量固定，flag区每一个remoteRank占用的格子数量为当前itemSize
    flagBlock_ = itemCapPerPair;

    // 收发任务分片，每一个收发任务分片的限制有两个，一个是bucketCap，一个是itemCapPerPair，分别对应内存和item数量的限制
    auto sliceOneSide = [&](std::deque<const HcclSendRecvItem*>& taskDeque, bool isSendSide) -> HcclResult {
        std::map<u32, std::vector<std::vector<AivBsrItemArgs>>> pairRounds;
        u32 curRemoteRank = INVALID_VALUE_RANKID;
        // curRoundBytes 必须在本 pair 内跨 item 累积，仅在切换 remoteRank 或开新轮时归零
        u64 curRoundBytes = 0;
        while (!taskDeque.empty()) {
            const HcclSendRecvItem* item = taskDeque.front();
            taskDeque.pop_front();
            // remoteRank 会窄化成 u16 且参与 cclOffset 计算，越界会读写本 rank cclBuffer 之外的地址；
            // AIV 侧没有 AICPU 那层隐式反查，必须在切片前显式拦住。
            CHK_PRT_RET(
                static_cast<u32>(item->remoteRank) >= rankSize_,
                HCCL_ERROR(
                    "[InsV2AivBatchSendRecvSoleExecutor][CalcSlicesAndRounds] invalid remoteRank[%u], myRank[%u], "
                    "rankSize[%u].",
                    static_cast<u32>(item->remoteRank), myRank_, rankSize_),
                HCCL_E_PARA);
            const bool isSelf = (item->remoteRank == static_cast<u32>(myRank_));
            // 自收发是直拷，发方向必须手里有目的地址；收方向不搬数据，不需要。
            u64 selfDstBase = 0;
            if (isSelf && isSendSide) {
                auto it = selfDst_.find(item);
                CHK_PRT_RET(
                    it == selfDst_.end(),
                    HCCL_ERROR(
                        "[InsV2AivBatchSendRecvSoleExecutor][CalcSlicesAndRounds] self send item has no paired "
                        "recv item, remoteRank[%u], myRank[%u].",
                        static_cast<u32>(item->remoteRank), myRank_),
                    HCCL_E_INTERNAL);
                selfDstBase = it->second;
            }
            // 不同remoteRank，新的一个rank分轮任务开始
            if (item->remoteRank != curRemoteRank) {
                curRemoteRank = item->remoteRank;
                curRoundBytes = 0;
                pairRounds[curRemoteRank].emplace_back();
            }
            u64 remainBytes = item->count * DATATYPE_SIZE_TABLE[item->dataType];
            u64 bufBaseOffset = 0;
            // 收发item分轮
            while (remainBytes > 0) {
                // 本remoteRank的item达到分轮要求，同上的两个限制
                if (curRoundBytes >= bucketCap || pairRounds[curRemoteRank].back().size() >= itemCapPerPair) {
                    curRoundBytes = 0;
                    pairRounds[curRemoteRank].emplace_back();
                }
                u64 room = bucketCap - curRoundBytes;
                // 本轮中本item搬运的数据
                u64 sliceBytes = std::min(remainBytes, room);
                AivBsrItemArgs frag{};
                frag.bufAddr = reinterpret_cast<u64>(item->buf) + bufBaseOffset;
                // 窄化安全：sliceBytes <= bucketCap、cclOffset <= usableCapacity-1 <= 256MB-1，均落在 u32 内。
                frag.byteCount = static_cast<u32>(sliceBytes);
                frag.sliceIdx = static_cast<u16>(pairRounds[curRemoteRank].back().size());
                if (isSelf) {
                    frag.sendRecvType = static_cast<u8>(isSendSide ? AIV_BSR_SELF_SEND_TYPE : AIV_BSR_SELF_RECV_TYPE);
                    // 只有 SELF_SEND 需要目的地址。SELF_RECV 不搬数据，dstAddr/peer 两个视图都不激活，
                    // 保持 frag{} 的零初始化即可。
                    if (isSendSide) {
                        frag.dstAddr = selfDstBase + bufBaseOffset;
                    }
                } else {
                    frag.sendRecvType = static_cast<u8>(isSendSide ? AIV_BSR_SEND_TYPE : AIV_BSR_RECV_TYPE);
                    // send 侧按对端编号分桶，recv 侧读的就是对端按本 rank 编号分好的那一桶
                    u32 bucketOwner = isSendSide ? curRemoteRank : static_cast<u32>(myRank_);
                    frag.peer.cclOffset = static_cast<u32>(static_cast<u64>(bucketOwner) * bucketCap + curRoundBytes);
                    // 窄化安全：remoteRank < rankSize <= 256（u16 足够）
                    frag.peer.remoteRank = static_cast<u16>(curRemoteRank);
                }
                frag.rsv = 0;
                pairRounds[curRemoteRank].back().push_back(frag);
                // 片起点对齐：bucketCap 与 curRoundBytes 恒为 32B 倍数，故非尾片长度（取满 room 时）必然对齐；
                // 尾片长度是 item 剩余字节，允许非对齐（数据本身就不是 32B 倍数），但落位必须抬到对齐边界，
                // 否则同一桶里后续片（可能属于下一个 item）的 cclOffset 会落在非对齐地址上。
                curRoundBytes += sliceBytes;
                curRoundBytes
                    = (curRoundBytes + AIV_BSR_BUCKET_ALIGN - 1) / AIV_BSR_BUCKET_ALIGN * AIV_BSR_BUCKET_ALIGN;
                bufBaseOffset += sliceBytes;
                remainBytes -= sliceBytes;
            }
        }
        // 同一个 pair 的第 k 轮固定落在 roundItems_[k]，收发两侧切片一致才能对齐。
        // 任务表已扩到 AIV_BSR_ITEM_MAX(512)，rankSize<=256 时单轮最坏 512 条（含自收发片），无需相位调度。
        for (auto& pr : pairRounds) {
            u64 roundIdx = 0;
            for (auto& roundList : pr.second) {
                if (roundItems_.size() <= roundIdx) {
                    roundItems_.resize(static_cast<size_t>(roundIdx) + 1);
                }
                roundItems_[roundIdx].insert(roundItems_[roundIdx].end(), roundList.begin(), roundList.end());
                roundIdx++;
            }
        }
        return HCCL_SUCCESS;
    };

    CHK_RET(sliceOneSide(sendDeque_, true));
    CHK_RET(sliceOneSide(recvDeque_, false));

    // rankSize<=256 时上面的不等式恒成立，这里只是防御性检查；与模板 KernelRun 用同一错误码，便于上层判断能否重试。
    for (size_t r = 0; r < roundItems_.size(); r++) {
        if (roundItems_[r].size() > AIV_BSR_ITEM_MAX) {
            HCCL_ERROR(
                "[InsV2AivBatchSendRecvSoleExecutor][CalcSlicesAndRounds] round[%zu] item num[%zu] exceeds "
                "limit[%u], rankSize[%u], itemCapPerPair[%llu], bucketCap[%llu].",
                r, roundItems_[r].size(), AIV_BSR_ITEM_MAX, rankSize_, itemCapPerPair, bucketCap);
            return HCCL_E_PARA;
        }
    }

    HCCL_INFO(
        "[InsV2AivBatchSendRecvSoleExecutor][CalcSlicesAndRounds] bucketCap[%llu], itemCapPerPair[%llu], "
        "flagBlock[%llu], self send num[%zu], total rounds[%zu].",
        bucketCap, itemCapPerPair, flagBlock_, selfDst_.size(), roundItems_.size());
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::DispatchOneRound(
    u64 roundIdx, const OpParam& param, const TemplateResource& templateAlgRes,
    std::shared_ptr<InsAlgTemplate>& algTemplate)
{
    TemplateDataParams tempAlgParams;
    // 本算子不使用 input/output：每条任务的用户地址都在任务表 items[] 里，中转区只有通信域的 cclBuffer。
    // 因此只声明实际使用的 hcclBuff，inBuffType/outBuffType 保持默认 DEFAULT、两个指针保持 nullptr，
    // 不出现"类型声明成 INPUT/OUTPUT 却配空地址"的不一致。
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;
    tempAlgParams.buffInfo.hcclBuff = cclMem_;
    tempAlgParams.count = cclMem_.size;

    ExtraArgsBsr extraArgs;
    extraArgs.itemNum = roundItems_[roundIdx].size();
    extraArgs.flagBlock = flagBlock_;
    // DFX 上报本轮真正发出的字节数（totalDataCount 只喂 taskInfo.size，不参与 launch）。只算发方向：
    // 自收发是 SELF_SEND + SELF_RECV 一对，全表累加会把同一份数据算两遍，而 SELF_RECV 并不搬数据。
    u64 dfxBytes = 0;
    for (u64 i = 0; i < extraArgs.itemNum; i++) {
        extraArgs.items[i] = roundItems_[roundIdx][i];
        const u32 type = extraArgs.items[i].sendRecvType;
        if (type == AIV_BSR_SEND_TYPE || type == AIV_BSR_SELF_SEND_TYPE) {
            dfxBytes += extraArgs.items[i].byteCount;
        }
    }
    algTemplate->SetBatchSendRecvItems(extraArgs);
    // 轮次序号（0 起）传给模板，由模板换算成 tag（BASE+1+roundIdx，越界回绕）随 sliceId 下发；
    // 各 rank 由同一套切片规则算出同一个值。
    algTemplate->SetDfxTotalBytes(dfxBytes);
    algTemplate->SetRoundIdx(static_cast<u32>(roundIdx));

    CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AivBatchSendRecvSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    HCCL_INFO("[InsV2AivBatchSendRecvSoleExecutor][Orchestrate] Start.");

    ResetTaskState();

    threads_ = resCtx.threads;
    myRank_ = resCtx.topoInfo.userRank;
    rankSize_ = resCtx.topoInfo.userRankSize;
    cclMem_ = resCtx.cclMem;
    algHierarchyInfo_ = resCtx.algHierarchyInfo;

    itemNum_ = param.batchSendRecvDataDes.itemNum;
    itemPtr_ = param.batchSendRecvDataDes.sendRecvItemsPtr;
    CHK_PRT_RET(
        rankSize_ == 0, HCCL_ERROR("[InsV2AivBatchSendRecvSoleExecutor][Orchestrate] rankSize is 0."), HCCL_E_PARA);
    CHK_PTR_NULL(itemPtr_);
    CHK_PRT_RET(
        algHierarchyInfo_.infos.empty(),
        HCCL_ERROR("[InsV2AivBatchSendRecvSoleExecutor][Orchestrate] algHierarchyInfo is empty."), HCCL_E_PARA);

    CHK_RET(BuildTaskDeques(itemPtr_, itemNum_));
    CHK_RET(CheckSelfSendRecvPairs());
    CHK_RET(CalcSlicesAndRounds(cclMem_.size));

    // itemNum>0 但每条 count 都为 0 时切片器不产生任何片：不下发 kernel、也不推进 tag，直接返回。
    if (roundItems_.empty()) {
        HCCL_INFO("[InsV2AivBatchSendRecvSoleExecutor][Orchestrate] empty task list, skip dispatch.");
        return HCCL_SUCCESS;
    }

    TemplateResource templateAlgRes;
    templateAlgRes.threads = threads_;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;

    std::shared_ptr<InsAlgTemplate> algTemplate
        = std::make_shared<InsAlgTemplate>(param, myRank_, algHierarchyInfo_.infos[0]);

    // tag 全在 kernel 内按轮次算出，host 既不下发 D2D 拷贝、失败路径也无需还原任何计数器，
    // 算子仍是"返回即下发完成"。
    HcclResult dispatchRet = HCCL_SUCCESS;
    for (u64 r = 0; r < roundItems_.size(); r++) {
        dispatchRet = DispatchOneRound(r, param, templateAlgRes, algTemplate);
        if (dispatchRet != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[InsV2AivBatchSendRecvSoleExecutor][Orchestrate] round[%llu] dispatch failed[%d].", r, dispatchRet);
            return dispatchRet;
        }
    }
    // 对通信域 tag 序列完全透明：kernel 既不读也不写那个共享计数器，本调用不会让后续算子（含本算子的
    // 下一次调用）看到被推进过的 tag；残留 flag 由 kernel 读方清 0。
    // 失败语义：轮次 tag 在不同调用间会重复，若某轮失败留下未清的 flag（对端永远等不到），下一次调用的同号
    // 轮次可能把它误判成"已到达"。因此下发失败后须重建通信域再重试，不能直接在原通信域上重下发。

    HCCL_INFO("[InsV2AivBatchSendRecvSoleExecutor][Orchestrate] End, rounds[%zu].", roundItems_.size());
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2(
    HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, AivBatchSendRecvSoleMesh, InsV2AivBatchSendRecvSoleExecutor,
    TopoMatchOneLevel, AivTempBatchSendRecvMesh1D);
REGISTER_ALG_ATTRS(
    AivBatchSendRecvSoleMesh, topo.maxSupportRankSize = MAX_RANK_SIZE_V; topo.supportLevel0Topos = LEVEL0_TOPO_ANY;
    topo.maxTopoLevelNum = TOPO_LEVEL_NUM_2; topo.isSupportLevel0PcieMix = true;
    topo.topoCustomCheck = [](const TopoInfoWithNetLayerDetails* topo) -> bool {
        return !topo->level2UbRtp && !topo->topLevelUboe;
    };);

} // namespace ops_hccl
