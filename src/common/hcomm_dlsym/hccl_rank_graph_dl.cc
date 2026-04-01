/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_rank_graph_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

DEFINE_WEAK_FUNC(HcclResult, HcclRankGraphGetTopoInstsByLayer, HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum);
DEFINE_WEAK_FUNC(HcclResult, HcclRankGraphGetTopoType, HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType);
DEFINE_WEAK_FUNC(HcclResult, HcclRankGraphGetRanksByTopoInst, HcclComm comm, uint32_t netLayer, uint32_t topoInstId,
                                                      uint32_t** ranks, uint32_t* rankNum);
DEFINE_WEAK_FUNC(HcclResult, HcclRankGraphGetEndpointNum, HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num);
DEFINE_WEAK_FUNC(HcclResult, HcclRankGraphGetEndpointDesc, HcclComm comm, uint32_t layer, uint32_t topoInstId,
                                                   uint32_t* descNum, EndpointDesc* endpointDesc);
DEFINE_WEAK_FUNC(HcclResult, HcclRankGraphGetEndpointInfo, HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc,
                                                   EndpointAttr endpointAttr, uint32_t infoLen, void* info);

void HcclRankGraphDlInit(void* libHcommHandle) {
    INIT_SUPPORT_FLAG(libHcommHandle, HcclRankGraphGetTopoInstsByLayer);
    INIT_SUPPORT_FLAG(libHcommHandle, HcclRankGraphGetTopoType);
    INIT_SUPPORT_FLAG(libHcommHandle, HcclRankGraphGetRanksByTopoInst);
    INIT_SUPPORT_FLAG(libHcommHandle, HcclRankGraphGetEndpointNum);
    INIT_SUPPORT_FLAG(libHcommHandle, HcclRankGraphGetEndpointDesc);
    INIT_SUPPORT_FLAG(libHcommHandle, HcclRankGraphGetEndpointInfo);
}