/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PAIRWISE_TESTCASE_COMMON_H
#define PAIRWISE_TESTCASE_COMMON_H

#include "sim_common.h"
#include <vector>

namespace HcclSim {
// AllToAll/AllToAllV/AllToAllVC Pairwise 算子 ST 用例矩阵生成 helper
// 模板内部按 2 套流集合（stream set）划分板空间，与物理框数无关
// 目标拓扑示例: 2 superpod × 8 server × 8 rank = 128 rank（128 为性能回归基线）
constexpr u32 PAIRWISE_RANK_SIZE = 128;

// 每 rank 总发送量预算（字节），per-pair 均摊，控制大规模仿真的内存与耗时
constexpr u64 PAIRWISE_RANK_BUDGET_BYTES = 8 * 1024 * 1024;

// 按 rank 总量预算均摊出每 pair 元素数（向下取整到 64B 对齐）
inline u64 GenPairwisePerPairCount(u32 rankSize, u32 dtypeSize)
{
    u64 perPairBytes = PAIRWISE_RANK_BUDGET_BYTES / (rankSize > 1 ? (rankSize - 1) : 1);
    return perPairBytes / dtypeSize / 64 * 64;
}

// 生成非对称矩阵: count[i][j] = baseCount + ((i + j) % 8) * stepCount，8 级变化
// caller 传 baseCount = stepCount = perPair / 8，峰值恰为均摊值，保证每 rank 总量不超预算
inline std::vector<u64> GenPairwiseAsymmetricMatrix(u32 rankSize, u64 baseCount, u64 stepCount)
{
    std::vector<u64> matrix(rankSize * rankSize, 0);
    for (u32 i = 0; i < rankSize; ++i) {
        for (u32 j = 0; j < rankSize; ++j) {
            matrix[i * rankSize + j] = baseCount + (i + j) % 8 * stepCount;
        }
    }
    return matrix;
}

// 生成部分零矩阵: 对角线为 0 (不自发自收), 其余等量 count
inline std::vector<u64> GenPairwiseZeroDiagMatrix(u32 rankSize, u64 count)
{
    std::vector<u64> matrix(rankSize * rankSize, count);
    for (u32 i = 0; i < rankSize; ++i) {
        matrix[i * rankSize + i] = 0;
    }
    return matrix;
}

// 生成对齐边界矩阵: count 对齐 alignBytes 字节 (按 dtypeSize 倍数对齐)
inline std::vector<u64> GenPairwiseAlignedMatrix(u32 rankSize, u64 count, u32 alignBytes)
{
    u64 aligned = (count + alignBytes - 1) / alignBytes * alignBytes;
    aligned = aligned == 0 ? alignBytes : aligned; // 至少 1 对齐单元
    return std::vector<u64>(rankSize * rankSize, aligned);
}

// 生成稀疏多通道矩阵: 每 rank 仅向 (i+1) 邻居发 largeCount（超 2MB 多通道阈值），其余 pair 发 smallCount
// 多通道路径的最小代价覆盖：每 rank 总量 = largeCount + (rankSize - 2) × smallCount，可控制在 8MB 预算内
// smallCount=0 时退化为稀疏环：单向有数 pair 覆盖纯发/纯收象限，非相邻 pair 双向 0 覆盖双零跳过象限
inline std::vector<u64> GenPairwiseSparseMatrix(u32 rankSize, u64 largeCount, u64 smallCount)
{
    std::vector<u64> matrix(rankSize * rankSize, smallCount);
    for (u32 i = 0; i < rankSize; ++i) {
        matrix[i * rankSize + i] = 0; // 不自发自收
        matrix[i * rankSize + (i + 1) % rankSize] = largeCount;
    }
    return matrix;
}
} // namespace HcclSim

#endif // PAIRWISE_TESTCASE_COMMON_H
