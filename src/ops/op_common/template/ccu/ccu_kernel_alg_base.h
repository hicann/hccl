/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_KERNEL_ALG_BASE
#define CCU_KERNEL_ALG_BASE

#include <vector>
#include <map>
#include <array>
#include <memory>

#include "log.h"
#include "ccu_kernel_utils.h"
#include "ccu_primitives_dl.hpp"
#include "ccu_log.h"
namespace ccu = ::AscendC::ccu;

namespace ops_hccl {

constexpr uint64_t CCU_MS_INTERLEAVE         = 8;
constexpr uint64_t CCU_MS_DEFAULT_LOOP_COUNT = 128;
constexpr uint64_t CCU_MS_SIZE               = 4096;
constexpr uint64_t NUM_TWO                   = 2;
constexpr uint32_t LOCAL_COPY_MS_PER_LOOP = 8;
constexpr uint32_t CCU_MS_LOCAL_COPY_LOOP_COUNT = 8;
constexpr uint32_t CCU_M2M_LOCAL_COPY_LOOP_COUNT = 16;

constexpr uint64_t CCU_LOOP_CKE_NUM_BCAST_V2 = 2; 
constexpr uint64_t CCU_LOOP_CKE_NUM_REDUCE_V2 = 3;
constexpr uint64_t CCU_LOOP_CKE_NUM_COPY_V2 = 2;
constexpr uint64_t CCU_LOOP_CKE_NUM_REDUCE_LOOP_V2 = 3;

struct LoopGroupConfig {
    uint32_t msInterleave;  // loop使用的ms步长，即与前一个loop间的间距
    uint32_t loopCount;     // loop的并行次数
    uint64_t memSlice;      // 单个loop内使用的ms总字节大小
};

struct LoopGroupResource {
    ccu::Array<ccu::Event>     completedEvent{0};
    ccu::Array<ccu::CcuBuffer> ccuBuf{0};
    uint32_t  eventCount;
    uint32_t  bufCount;
};

struct GroupOpSizeVars {
    ccu::Variable addrOffset;        // 第二个loopGroup搬运的起始偏移
    ccu::Variable loopParam;         // loop串行重复执行次数
    ccu::Variable parallelParam;     // loopgroup展开参数，包括展开次数、从第几个loop开始展开、共有几个loop
    ccu::Variable residual;          // 尾块数据size
};

struct GroupCopyVar {
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopDst[2];
    ccu::Variable  loopLen[2];
};

struct GroupReduceVar {
    ccu::LocalAddr loopDst[2];
    std::array<std::vector<ccu::RemoteAddr>, NUM_TWO> loopRemoteSrc;
    ccu::LocalAddr loopLocalSrc[2];
    ccu::Variable  loopLen[2];
    ccu::Variable  loopLenExp[2];
};

struct GroupBroadcastVar {
    ccu::LocalAddr loopSrc[2];
    ccu::LocalAddr loopLocalDst[2];
    std::array<std::vector<ccu::RemoteAddr>, NUM_TWO> loopRemoteDst;
    ccu::Variable  loopLen[2];
};

struct GroupLocalReduceVar {
    ccu::LocalAddr loopDst[2];
    std::array<std::vector<ccu::LocalAddr>, NUM_TWO> loopScratch;
    ccu::Variable  loopLen[2];
    ccu::Variable  loopLenExp[2];
};

struct CcuKernelCtxBase {
    struct CcuLoopEntity {
        std::unique_ptr<ccu::Func> body[2];
        std::unique_ptr<ccu::Loop> loops[2];
        ccu::Variable              loopParam[2];
        ccu::Variable              addrOffset[2];
    };

    LoopGroupConfig  moConfig;
    LoopGroupResource moRes;
    bool resourceAllocated;

    std::map<std::string, CcuLoopEntity> loopMap;
    CcuLoopExecutors enginePool;

    // GroupCopyVar 延迟分配：仅使用 GroupCopy 的 kernel 才会创建，避免资源浪费
    std::unique_ptr<GroupCopyVar> gcVarPtr;
    GroupCopyVar& GetGcVar() {
        if (!gcVarPtr) {
            gcVarPtr.reset(new GroupCopyVar());
        }
        return *gcVarPtr;
    }

    void CreateLoopEntity(std::string loopStr) {
        loopMap.emplace(loopStr, CcuLoopEntity());
    }

    bool IsLoopEntityRegistered(std::string loopStr) {
        return loopMap.count(loopStr) != 0;
    }
};

std::vector<uint64_t> CalGoSize(uint64_t size);
std::vector<uint64_t> CalGoSize(uint64_t size, CcuVersion ccuVersion);
std::vector<uint64_t> CalGoSize(uint64_t size, const LoopGroupConfig &config, CcuVersion ccuVersion = CcuVersion::CCU_V1);
CcuResult AllocGoResource(LoopGroupConfig &config, LoopGroupResource &res,
    bool &allocated, uint32_t parallelDim = CCU_MS_DEFAULT_LOOP_COUNT, uint32_t msPerLoop = 1);

CcuResult GroupBroadcastWithoutMyRank(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount,
                        std::vector<ccu::RemoteAddr> dst, ccu::LocalAddr src, GroupOpSizeVars goSize);

CcuResult GroupReduceWithoutMyRank(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount,
                        ccu::LocalAddr dst, std::vector<ccu::RemoteAddr> src, GroupOpSizeVars goSize,
                        HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);

CcuResult CreateMultiOpCopyV1(CcuKernelCtxBase &ctx, GroupCopyVar &var);
CcuResult GroupCopy(CcuKernelCtxBase &ctx, ccu::LocalAddr dst, ccu::LocalAddr src, GroupOpSizeVars goSize,
                    CcuVersion ccuVersion = CcuVersion::CCU_V1);

CcuResult CreateReduceLoop(CcuKernelCtxBase &ctx, GroupLocalReduceVar &var, uint32_t size,
    HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);
CcuResult GroupLocalReduce(CcuKernelCtxBase &ctx, ccu::LocalAddr outDstOrg, std::vector<ccu::LocalAddr> &scratchOrg,
    GroupOpSizeVars goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType);

CcuResult CreateMultiOpBroadcastWithoutMyRank(CcuKernelCtxBase &ctx, GroupBroadcastVar &var,
                                 const size_t channels[], uint32_t channelCount);

CcuResult GroupReduce(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount,
                        ccu::LocalAddr dst, std::vector<ccu::RemoteAddr> src, ccu::LocalAddr localSrc,
                        GroupOpSizeVars goSize, HcclDataType dataType, HcclDataType outputDataType, HcclReduceOp opType,
                        CcuVersion ccuVersion = CcuVersion::CCU_V1);

CcuResult CreateMultiOpReduceV1(CcuKernelCtxBase &ctx, GroupReduceVar &var,
                               const size_t channels[], uint32_t channelCount, HcclDataType dataType,
                               HcclDataType outputDataType, HcclReduceOp opType);

CcuResult CreateMultiOpBroadcastV1(CcuKernelCtxBase &ctx, GroupBroadcastVar &var,
                                 const size_t channels[], uint32_t channelCount);

CcuResult GroupBroadcast(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount,
                         ccu::LocalAddr localDst, std::vector<ccu::RemoteAddr> dst, ccu::LocalAddr src, GroupOpSizeVars goSize,
                         CcuVersion ccuVersion = CcuVersion::CCU_V1);

CcuResult CreateMultiOpReduceWithoutMyRank(CcuKernelCtxBase &ctx, GroupReduceVar &var,
                                 const size_t channels[], uint32_t channelCount, HcclDataType dataType,
                                 HcclDataType outputDataType, HcclReduceOp opType);
CcuResult GroupReduceV1(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount, ccu::LocalAddr dst,
                        std::vector<ccu::RemoteAddr> src, ccu::LocalAddr localSrc, GroupOpSizeVars goSize, HcclDataType dataType,
                        HcclDataType outputDataType, HcclReduceOp opType);
CcuResult GroupBroadcastV1(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount,
                         ccu::LocalAddr localDst, std::vector<ccu::RemoteAddr> dst, ccu::LocalAddr src, GroupOpSizeVars goSize);
CcuResult GroupCopyV1(CcuKernelCtxBase &ctx, ccu::LocalAddr dst, ccu::LocalAddr src, GroupOpSizeVars goSize);
CcuResult GroupReduceV2(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount, ccu::LocalAddr dst,
                        std::vector<ccu::RemoteAddr> src, ccu::LocalAddr localSrc, GroupOpSizeVars goSize, HcclDataType dataType,
                        HcclDataType outputDataType, HcclReduceOp opType);
CcuResult GroupBroadcastV2(CcuKernelCtxBase &ctx, const size_t channels[], uint32_t channelCount,
                        ccu::LocalAddr localDst, std::vector<ccu::RemoteAddr> dst, ccu::LocalAddr src, GroupOpSizeVars goSize);
CcuResult GroupCopyV2(CcuKernelCtxBase &ctx, ccu::LocalAddr dst, ccu::LocalAddr src, GroupOpSizeVars goSize);
}

#endif // !CCU_KERNEL_ALG_BASE
