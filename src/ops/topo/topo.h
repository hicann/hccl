/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_TOPO
#define OPS_HCCL_SRC_OPS_TOPO

#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "log.h"
#include "alg_param.h"
#include "hccl_rank_graph.h"
#include "hccl_rankgraph.h"
#include "hccl_res.h"
#include "hcomm_primitives.h"


using namespace ops_hccl;

namespace ops_hccl {

constexpr s32 DEVICE_PER_MODULE_A2 = 8;

HcclResult InitRankInfo(HcclComm comm, TopoInfo* topoInfo);

HcclResult CalcMyRankInfo(HcclComm comm, const std::vector<struct GraphRankInfo> &rankList, TopoInfo* topoInfo);
HcclResult SetServerModuleInfo(const std::vector<struct GraphRankInfo> &rankList, TopoInfo* topoInfo);
HcclResult SetSuperPodInfo(const std::vector<struct GraphRankInfo> &rankList, TopoInfo* topoInfo);
bool IsDiffDeviceModule(const std::vector<struct GraphRankInfo> &rankList, TopoInfo* topoInfo);
HcclResult GetModuleIdx(const struct GraphRankInfo &rankInfo, TopoInfo* topoInfo, u32& moduleIdx);

HcclResult CalcLinkInfo(HcclComm comm, const std::vector<struct GraphRankInfo> &rankList, TopoInfo* topoInfo);

HcclResult CalcGeneralTopoInfoForA2(HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo);
HcclResult CalcGeneralTopoInfoForA3(HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo);
HcclResult CalcGeneralTopoInfoForComm(HcclComm comm, TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo);

HcclResult GetUserRankBySubCommRank(u32 subCommRank, u32 curLevel, AlgHierarchyInfo& algHierarchyInfo, u32 &userRank);
HcclResult GetSubCommRankByUserRank(u32 userRank, u32 curLevel, AlgHierarchyInfo& algHierarchyInfo, u32 &subCommRank);

}

#endif

