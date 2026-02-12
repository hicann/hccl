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

#include "log.h"
#include "ccu_kernel.h"

namespace ops_hccl {
using namespace hcomm;

/* hccl仓CcuKernel基类，提供group高阶操作接口 */
class CcuKernelAlgBase : public CcuKernel {
public:
    // 继承所有构造函数
    using CcuKernel::CcuKernel;

protected:
    // 编程接口
    struct GroupOpConfig {
        uint32_t msInterleave;  // loop使用的ms步长，即与前一个loop间的间距
        uint32_t loopCount;     // loop的并行次数
        uint64_t memSlice;      // 单个loop内使用的ms总字节大小
    };

    struct GroupOpSizeResource {
        std::vector<CcuRep::CompletedEvent> completedEvent;
        std::vector<CcuRep::CcuBuf>         ccuBuf;
        std::vector<CcuRep::Executor>       executor;
    };

    struct GroupOpSize {
        CcuRep::Variable addrOffset;        // loop展开时的addr步长
        CcuRep::Variable loopParam;         // loop串行重复执行次数
        CcuRep::Variable parallelParam;     // loopgroup展开参数，包括展开次数、从第几个loop开始展开、共有几个loop
        CcuRep::Variable residual;          // 尾块数据size
    };
    
    // 用于n和p部分数据loopgroup的参数
    GroupOpConfig       moConfig{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF};
    GroupOpSizeResource moRes;

    const uint32_t LOCAL_COPY_MS_PER_LOOP = 8;
    const uint32_t CCU_MS_LOCAL_COPY_LOOP_COUNT = 8;

    // 引入基类Load函数，防止名称遮蔽
    using CcuKernel::LocalReduceNb;
    // 封装vector接口
    HcclResult LocalReduceNb(const std::vector<CcuRep::CcuBuf> &bufs, uint32_t count, HcclDataType dataType,
                     HcclDataType outputDataType, HcclReduceOp opType,
                     const CcuRep::Variable &len, CcuRep::CompletedEvent event);

    // loopgroup
    GroupOpSize CreateGroupOpSize();
    std::vector<CcuRep::CcuBuf> CreateBlockCcuBuf(uint32_t count);
    std::vector<CcuRep::Executor> CreateBlockExecutor(uint32_t count);
    std::vector<CcuRep::CompletedEvent> CreateBlockCompletedEvent(uint32_t count);
  
    void LoopGroup(const std::vector<CcuRep::LoopCall> &loops, const std::vector<CcuRep::Variable> &loopCfg,
        const CcuRep::Variable &paraCfg, const CcuRep::Variable &offsetCfg);
    
    // 高阶操作
    std::vector<uint64_t> CalGoSize(uint64_t size);

    void AllocGoResource(uint32_t parallelDim = CcuRep::CCU_MS_DEFAULT_LOOP_COUNT, uint32_t msPerLoop = 1);
    // 引入基类Load函数，防止名称遮蔽
    using CcuKernel::Load;
    void Load(GroupOpSize moSize);

    HcclResult GroupBroadcast(const std::vector<ChannelHandle>& channels, std::vector<CcuRep::RemoteAddr> dst,
                              CcuRep::LocalAddr src, GroupOpSize goSize);

    HcclResult GroupReduce(const std::vector<ChannelHandle>& channels, CcuRep::LocalAddr dst,
                           std::vector<CcuRep::RemoteAddr> src, GroupOpSize goSize, HcclDataType dataType,
                           HcclDataType outputDataType, HcclReduceOp opType);

    HcclResult GroupCopy(CcuRep::LocalAddr dst, CcuRep::LocalAddr src, GroupOpSize goSize);

private:
    HcclResult CreateMultiOpCopy();
    HcclResult CreateMultiOpBroadcast(const std::vector<ChannelHandle> &channels);
    HcclResult CreateMultiOpReduce(const std::vector<ChannelHandle>& channels, HcclDataType dataType,
                                   HcclDataType outputDataType, HcclReduceOp opType);
};

}

#endif // !CCU_KERNEL_ALG_BASE