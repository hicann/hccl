/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "scatter_mesh.h"

namespace ops_hccl {
ScatterMesh::ScatterMesh() : AlgTemplateBase()
{
}

ScatterMesh::~ScatterMesh()
{
}

HcclResult ScatterMesh::Prepare(u32 interRank, u32 interRankSize)
{
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    return HCCL_SUCCESS;
}

void ScatterMesh::PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const
{
    slices_.resize(rankSize);
    u64 sliceSize = (totalCount / rankSize) * unitSize;

    for (u32 i = 0; i < rankSize; i++) {
        slices_[i].offset = i * sliceSize;
        slices_[i].size = sliceSize;
        HCCL_DEBUG(" default slice[%u]: offset: [%llu] size[%llu]", i, i * sliceSize, sliceSize);
    }
}

// scatter的入口函数
HcclResult ScatterMesh::RunAsync(const u32 rank, const u32 rankSize, std::vector<ChannelInfo> &channels)
{
    HCCL_INFO("ScatterMesh run: rank[%u] rankSize[%u] inputMem[%p] to outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.addr, outputMem_.addr, count_);
    // ranksize ==1 的处理
    if (rankSize == 1) {
        if (inputMem_.addr != outputMem_.addr) {
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, outputMem_.addr, inputMem_.addr, inputMem_.size)));
        }
        return HCCL_SUCCESS;
    }

    if (channels.size() < rankSize) {
        HCCL_ERROR("[ScatterMesh][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank, channels.size(), rankSize);
        return HCCL_E_INTERNAL;
    }

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[ScatterMesh][RunAsync]rank[%u] unit data size is zero", rank);
        return HCCL_E_INTERNAL;
    }
    if (slices_.size() == 0) {
        PrepareSlicesData(unitSize, count_, rankSize);
    }
    // rank存放scatter 结果的偏移
    u64 scatterOffset = slices_[rank].offset;
    HCCL_DEBUG("rank[%u] scatter_offset is [%llu] reslutsize[%llu]", rank, \
        scatterOffset, slices_[rank].size);

    // root rank向其他rank发送数据
    if (rank == root_) {
        for (u32 dstRank = 0; dstRank < rankSize; dstRank++) {
            HcclResult ret = RunSendScatter(dstRank, slices_[dstRank], channels);
            CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[ScatterMesh][RunAsync]root rank[%u] send scatter to "\
                "dstrank[%u] run failed", rank, dstRank), ret);
        }
    } else { // 非rootrank 从rootrank接收数据
        CHK_RET(RunRecvScatter(root_, slices_[rank], channels));
    }

    // Barrier确保数据收发完成
    if (barrierSwitchOn_) {
        for (u32 dstRank = 0; dstRank < rankSize; dstRank++) {
            if (dstRank != interRank_) {
                CHK_RET(ExecuteBarrier(channels[dstRank], thread_));
            }
        }
    }

    HCCL_INFO("ScatterMesh finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

HcclResult ScatterMesh::RunSendScatter(const u32 dstRank, const Slice &slice, std::vector<ChannelInfo> &channels)
{
    (void) slice;
    // 本rank是root rank，直接进行数据的拷贝，从input拷贝到output
    if (dstRank == interRank_) {
        if (inputMem_.addr != outputMem_.addr) {
             // root rank给自身拷贝时候不需要同步信号，拷贝到outputmem的偏移不同
            void* src = static_cast<void *>(static_cast<u8 *>(inputMem_.addr) + slices_[interRank_].offset);
            void* dst = static_cast<void *>(static_cast<u8 *>(outputMem_.addr) + slices_[interRank_].offset);
            HCCL_DEBUG("root rank copy from input[%p] range[%llu] to output[%p] range[%llu], size[%llu]", src,
                slices_[interRank_].offset, dst, slices_[interRank_].offset, slices_[interRank_].size);
            CHK_RET(static_cast<HcclResult>(HcommLocalCopyOnThread(thread_, dst, src, slices_[interRank_].size)));
        }
    } else { // root rank给其他rank进行数据发送
        // 接收目的rank的同步信号，便可进行下一轮发送
        CHK_RET(static_cast<HcclResult>(HcommNotifyWaitOnThread(thread_, channels[dstRank].handle, NOTIFY_IDX_ACK, CUSTOM_TIMEOUT)));
        CHK_RET(static_cast<HcclResult>(HcommNotifyRecordOnThread(thread_, channels[dstRank].handle, NOTIFY_IDX_DATA_SIGNAL)));
    }
    return HCCL_SUCCESS;
}

// 只有非root节点才接收数据
HcclResult ScatterMesh::RunRecvScatter(const u32 srcRank, const Slice &slice, std::vector<ChannelInfo> &channels)
{
    // 判断数据是否需要分片
    if (srcRank >= channels.size()) {
        HCCL_ERROR("[Run][RecvScatter]SrcRank[%u] is out of range, linkSize[%llu]", srcRank, channels.size());
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("rank[%u] will rcv with ouput's offset[%llu], size[%llu]", interRank_, slice.offset, slice.size);

    // 向root节点发送tx同步,rxmem可用
    CHK_RET(static_cast<HcclResult>(HcommNotifyRecordOnThread(thread_, channels[srcRank].handle, NOTIFY_IDX_ACK)));
    CHK_RET(static_cast<HcclResult>(HcommNotifyWaitOnThread(thread_, channels[srcRank].handle, NOTIFY_IDX_DATA_SIGNAL, CUSTOM_TIMEOUT)));

    void* src = static_cast<void *>(static_cast<s8 *>(channels[srcRank].remoteInput.addr) + slice.offset + baseOffset_);
    void* dst = static_cast<void *>(static_cast<s8 *>(outputMem_.addr) + slice.offset);
    CHK_RET(static_cast<HcclResult>(HcommReadOnThread(thread_, channels[srcRank].handle, dst, src, slice.size)));

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_SCATTER_MESH, ScatterMesh);
}
