/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_REDUCE_SCATTER_OP_H
#define AIV_REDUCE_SCATTER_OP_H

#include "aiv_communication_base_v2.h"
#include "aiv_reduce_scatter_mesh_1d.h"
#include "aiv_reduce_scatter_mesh_1d_corectrl.h"
#include "aiv_reduce_scatter_mesh_1d_bigdata.h"
#include "aiv_reduce_scatter_mesh_1d_bigdata_write.h"
#include "aiv_reduce_scatter_local_tree.h"
#include "aiv_reduce_scatter_local_tree_write.h"
#include "aiv_reduce_scatter_local_tree_corectrl.h"
#include "aiv_reduce_scatter_mesh_1d_hif8.h" // hif8 两阶段升精度归约 kernel（搬运 hif8/本地升 fp32 归约）

using namespace AscendC;

// 对称内存使能探测：host 按窗口布局算好对端 input 对称 VA 数组后下发，本 rank 条目非 0 表示可用。
// ConsumerSym 按输出连续区间切分全部核、不依赖「生产者/消费者」两段式，低核数（numBlocks <= 2*rankSize）
// 也走对称分支，故入口不按 rankSize / 核数设限（host 侧 CalNumBlocks 只压核数上限，不会抬到 2 核）。
__aicore__ inline bool AivReduceScatterSymInEnabled(GM_ADDR buffIn, uint32_t rank, uint32_t rankSize)
{
    (void)rankSize;
    __gm__ uint64_t* base = reinterpret_cast<__gm__ uint64_t*>(buffIn);
    return base[AIV_SYM_PEER_IN_ADDR_OFFSET / sizeof(uint64_t) + rank] != 0;
}

#if defined(HCCL_AIV_ACLGRAPH_SK)
#define AIV_REDUCE_SCATTER_KERNEL_DECL(type) \
    extern "C" __aicore__ void aiv_reduce_scatter_##type##_inner(KERNEL_ARGS_DEF);

// isWrite为host下发的运行期参数：按核数选择大数据写语义或LocalTreeWrite，少核沿用读语义。
#define AIV_REDUCE_SCATTER_KERNEL_DEF(type)                                       \
    extern "C" __aicore__ void aiv_reduce_scatter_##type##_inner(KERNEL_ARGS_DEF) \
    {                                                                             \
        AIV_INFO_HINT;                                                            \
        if (AivReduceScatterSymInEnabled(buffIn, rank, rankSize)) {               \
            AivReduceScatterV2Mesh1DBigData<type>(KERNEL_ARGS_CALL);              \
            return;                                                               \
        }                                                                         \
        if (isWrite && numBlocks > 2 * rankSize) {                                \
            AivReduceScatterV2Mesh1DBigDataWrite<type>(KERNEL_ARGS_CALL);         \
            return;                                                               \
        }                                                                         \
        if (isWrite && numBlocks >= rankSize) {                                   \
            AivReduceScatterV2LocalTreeWrite<type>(KERNEL_ARGS_CALL);             \
            return;                                                               \
        }                                                                         \
        if (numBlocks > 2 * rankSize) {                                           \
            AivReduceScatterV2Mesh1DBigData<type>(KERNEL_ARGS_CALL);              \
        } else if (numBlocks >= rankSize) {                                       \
            AivReduceScatterV2LocalTree<type>(KERNEL_ARGS_CALL);                  \
        } else {                                                                  \
            AivReduceScatterV2LocalTreeCoreCtrl<type>(KERNEL_ARGS_CALL);          \
        }                                                                         \
    }

#if defined(BUILD_SK_FUNC) && defined(SK_FUNC_ID)
#define AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF(type) \
    AIV_REDUCE_SCATTER_KERNEL_DECL(type);         \
    SK_BIND_FUNC_DEF(aiv_reduce_scatter_##type, SK_FUNC_ID)
#else
#define AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF(type) \
    AIV_REDUCE_SCATTER_KERNEL_DEF(type);          \
    GLOBAL_FUNC_DEF(aiv_reduce_scatter_##type);   \
    SuperKernelBind(aiv_reduce_scatter_##type)
#endif
#else // 未启用 aclgraph SK（960 等）：保持原 __global__ 入口导出
#define AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF(type)                                    \
    extern "C" __global__ __aicore__ void aiv_reduce_scatter_##type(KERNEL_ARGS_DEF) \
    {                                                                                \
        AIV_INFO_HINT;                                                               \
        if (AivReduceScatterSymInEnabled(buffIn, rank, rankSize)) {                  \
            AivReduceScatterV2Mesh1DBigData<type>(KERNEL_ARGS_CALL);                 \
            return;                                                                  \
        }                                                                            \
        if (isWrite && numBlocks > 2 * rankSize) {                                   \
            AivReduceScatterV2Mesh1DBigDataWrite<type>(KERNEL_ARGS_CALL);            \
            return;                                                                  \
        }                                                                            \
        if (isWrite && numBlocks >= rankSize) {                                      \
            AivReduceScatterV2LocalTreeWrite<type>(KERNEL_ARGS_CALL);                \
            return;                                                                  \
        }                                                                            \
        if (numBlocks > 2 * rankSize) {                                              \
            AivReduceScatterV2Mesh1DBigData<type>(KERNEL_ARGS_CALL);                 \
        } else if (numBlocks >= rankSize) {                                          \
            AivReduceScatterV2LocalTree<type>(KERNEL_ARGS_CALL);                     \
        } else {                                                                     \
            AivReduceScatterV2LocalTreeCoreCtrl<type>(KERNEL_ARGS_CALL);             \
        }                                                                            \
    }                                                                                \
    EXPORT_AIV_META_INFO(aiv_reduce_scatter_##type)
#endif

// 定义各算子各数据类型Kernel入口
AIV_ATOMIC_DATA_TYPE_DEF(AIV_REDUCE_SCATTER_KERNEL_BATCH_DEF);

extern "C" __global__ __aicore__ void aiv_reduce_scatter_hifloat8_t(KERNEL_ARGS_DEF)
{
    AIV_INFO_HINT;
    AivReduceScatterV2Mesh1DHif8<hifloat8_t>(KERNEL_ARGS_CALL);
}
EXPORT_AIV_META_INFO(aiv_reduce_scatter_hifloat8_t);

#endif // AIV_REDUCE_SCATTER_OP_H
