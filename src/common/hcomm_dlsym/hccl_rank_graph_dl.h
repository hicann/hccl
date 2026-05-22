/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_RANK_GRAPH_DL_H
#define HCCL_RANK_GRAPH_DL_H

#include "dlsym_common.h"
#include "hccl_rank_graph.h"   // 原头文件，包含所有类型和 inline 函数

/* 8.5.0 桩: hccl_rank_graph.h 中 9.0.0 新增类型 */
#if CANN_VERSION_NUM < 90000000
typedef enum {
    ENDPOINT_ATTR_INVALID = -1,
    ENDPOINT_ATTR_BW_COEFF = 0,
    ENDPOINT_ATTR_DIE_ID = 1,
    ENDPOINT_ATTR_LOCATION = 2
} EndpointAttr;

typedef uint32_t EndpointAttrBwCoeff;
typedef uint32_t EndpointAttrDieId;
typedef uint32_t EndpointAttrLocation;

#define COMM_TOPO_A2AXSERVER ((CommTopo)4)
#define COMM_TOPO_CUSTOM     ((CommTopo)5)
#endif /* CANN_VERSION_NUM < 90000000 */

#ifdef __cplusplus
extern "C" {
#endif

DECL_WEAK_FUNC(HcclResult, HcclRankGraphGetTopoInstsByLayer, HcclComm comm, uint32_t netLayer,
    uint32_t** topoInsts, uint32_t* topoInstNum);
DECL_WEAK_FUNC(HcclResult, HcclRankGraphGetTopoType, HcclComm comm, uint32_t netLayer,
    uint32_t topoInstId, CommTopo* topoType);
DECL_WEAK_FUNC(HcclResult, HcclRankGraphGetRanksByTopoInst, HcclComm comm, uint32_t netLayer,
    uint32_t topoInstId, uint32_t** ranks, uint32_t* rankNum);
DECL_WEAK_FUNC(HcclResult, HcclRankGraphGetEndpointNum, HcclComm comm, uint32_t layer,
    uint32_t topoInstId, uint32_t* num);
DECL_WEAK_FUNC(HcclResult, HcclRankGraphGetEndpointDesc, HcclComm comm, uint32_t layer,
    uint32_t topoInstId, uint32_t* descNum, EndpointDesc* endpointDesc);
DECL_WEAK_FUNC(HcclResult, HcclRankGraphGetEndpointInfo, HcclComm comm, uint32_t rankId,
    const EndpointDesc* endpointDesc, EndpointAttr endpointAttr, uint32_t infoLen, void* info);

void HcclRankGraphDlInit(void* libHcommHandle);

#ifdef __cplusplus
}
#endif

#endif // HCCL_RANK_GRAPH_DL_H