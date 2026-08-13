/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_TOPO_EXPERIMENTAL
#define OPS_HCCL_SRC_OPS_TOPO_EXPERIMENTAL

#include "topo.h"

namespace ops_hccl_experimental {
using ops_hccl::AlgHierarchyInfo;
using ops_hccl::TopoInfo;

HcclResult
CalcGeneralTopoInfoInterServer(const HcclComm comm, const TopoInfo* topoInfo, AlgHierarchyInfo& algHierarchyInfo);

} // namespace ops_hccl_experimental

#endif
