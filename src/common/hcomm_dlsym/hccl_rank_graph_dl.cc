/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log.h"
#include "hccl_rank_graph_dl.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

// 定义全局函数指针（小驼峰）
HcclResult (*hcclRankGraphGetLayersPtr)(HcclComm, uint32_t**, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetRanksByLayerPtr)(HcclComm, uint32_t, uint32_t**, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetRankSizeByLayerPtr)(HcclComm, uint32_t, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetTopoTypeByLayerPtr)(HcclComm, uint32_t, CommTopo*) = NULL;
HcclResult (*hcclRankGraphGetInstSizeListByLayerPtr)(HcclComm, uint32_t, uint32_t**, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetLinksPtr)(HcclComm, uint32_t, uint32_t, uint32_t, CommLink**, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetTopoInstsByLayerPtr)(HcclComm, uint32_t, uint32_t**, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetTopoTypePtr)(HcclComm, uint32_t, uint32_t, CommTopo*) = NULL;
HcclResult (*hcclRankGraphGetRanksByTopoInstPtr)(HcclComm, uint32_t, uint32_t, uint32_t**, uint32_t*) = NULL;
HcclResult (*hcclGetHeterogModePtr)(HcclComm, HcclHeterogMode*) = NULL;
HcclResult (*hcclRankGraphGetEndpointNumPtr)(HcclComm, uint32_t, uint32_t, uint32_t*) = NULL;
HcclResult (*hcclRankGraphGetEndpointDescPtr)(HcclComm, uint32_t, uint32_t, uint32_t*, EndpointDesc*) = NULL;
HcclResult (*hcclRankGraphGetEndpointInfoPtr)(HcclComm, uint32_t, const EndpointDesc*, EndpointAttr, uint32_t, void*) = NULL;

// 添加支持标志（静态，默认 false，初始化时根据 dlsym 结果设置）
static bool g_hcclRankGraphGetLayersSupported = false;
static bool g_hcclRankGraphGetRanksByLayerSupported = false;
static bool g_hcclRankGraphGetRankSizeByLayerSupported = false;
static bool g_hcclRankGraphGetTopoTypeByLayerSupported = false;
static bool g_hcclRankGraphGetInstSizeListByLayerSupported = false;
static bool g_hcclRankGraphGetLinksSupported = false;
static bool g_hcclRankGraphGetTopoInstsByLayerSupported = false;
static bool g_hcclRankGraphGetTopoTypeSupported = false;
static bool g_hcclRankGraphGetRanksByTopoInstSupported = false;
static bool g_hcclGetHeterogModeSupported = false;
static bool g_hcclRankGraphGetEndpointNumSupported = false;
static bool g_hcclRankGraphGetEndpointDescSupported = false;
static bool g_hcclRankGraphGetEndpointInfoSupported = false;

// ---------- 桩函数定义（签名与真实API完全一致）----------
static HcclResult StubHcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum) {
    (void)comm; (void)netLayers; (void)netLayerNum;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetLayers not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum) {
    (void)comm; (void)netLayer; (void)ranks; (void)rankNum;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetRanksByLayer not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum) {
    (void)comm; (void)netLayer; (void)rankNum;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetRankSizeByLayer not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, CommTopo* topoType) {
    (void)comm; (void)netLayer; (void)topoType;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetTopoTypeByLayer not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize) {
    (void)comm; (void)netLayer; (void)instSizeList; (void)listSize;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetInstSizeListByLayer not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank,
                                            CommLink** links, uint32_t* linkNum) {
    (void)comm; (void)netLayer; (void)srcRank; (void)dstRank; (void)links; (void)linkNum;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetLinks not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum) {
    (void)comm; (void)netLayer; (void)topoInsts; (void)topoInstNum;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetTopoInstsByLayer not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType) {
    (void)comm; (void)netLayer; (void)topoInstId; (void)topoType;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetTopoType not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId,
                                                      uint32_t** ranks, uint32_t* rankNum) {
    (void)comm; (void)netLayer; (void)topoInstId; (void)ranks; (void)rankNum;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetRanksByTopoInst not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclGetHeterogMode(HcclComm comm, HcclHeterogMode* mode) {
    (void)comm; (void)mode;
    HCCL_ERROR("[HcclWrapper] HcclGetHeterogMode not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetEndpointNum(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num) {
    (void)comm; (void)layer; (void)topoInstId; (void)num;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetEndpointNum not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t topoInstId,
                                                   uint32_t* descNum, EndpointDesc* endpointDesc) {
    (void)comm; (void)layer; (void)topoInstId; (void)descNum; (void)endpointDesc;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetEndpointDesc not supported");
    return HCCL_E_NOT_SUPPORTED;
}

static HcclResult StubHcclRankGraphGetEndpointInfo(HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc,
                                                   EndpointAttr endpointAttr, uint32_t infoLen, void* info) {
    (void)comm; (void)rankId; (void)endpointDesc; (void)endpointAttr; (void)infoLen; (void)info;
    HCCL_ERROR("[HcclWrapper] HcclRankGraphGetEndpointInfo not supported");
    return HCCL_E_NOT_SUPPORTED;
}

void HcclRankGraphDlInit(void* libHcommHandle) {
    // 辅助宏：解析符号，失败则指向对应桩函数，同时设置支持标志
    #define SET_PTR(ptr, name, stub, support_flag) \
        do { \
            ptr = (decltype(ptr))dlsym(libHcommHandle, name); \
            if (ptr == NULL) { \
                ptr = stub; \
                support_flag = false; \
                HCCL_DEBUG("[HcclWrapper] %s not supported", name); \
            } else { \
                support_flag = true; \
            } \
        } while(0)

    SET_PTR(hcclRankGraphGetLayersPtr, "HcclRankGraphGetLayers", StubHcclRankGraphGetLayers, g_hcclRankGraphGetLayersSupported);
    SET_PTR(hcclRankGraphGetRanksByLayerPtr, "HcclRankGraphGetRanksByLayer", StubHcclRankGraphGetRanksByLayer, g_hcclRankGraphGetRanksByLayerSupported);
    SET_PTR(hcclRankGraphGetRankSizeByLayerPtr, "HcclRankGraphGetRankSizeByLayer", StubHcclRankGraphGetRankSizeByLayer, g_hcclRankGraphGetRankSizeByLayerSupported);
    SET_PTR(hcclRankGraphGetTopoTypeByLayerPtr, "HcclRankGraphGetTopoTypeByLayer", StubHcclRankGraphGetTopoTypeByLayer, g_hcclRankGraphGetTopoTypeByLayerSupported);
    SET_PTR(hcclRankGraphGetInstSizeListByLayerPtr, "HcclRankGraphGetInstSizeListByLayer", StubHcclRankGraphGetInstSizeListByLayer, g_hcclRankGraphGetInstSizeListByLayerSupported);
    SET_PTR(hcclRankGraphGetLinksPtr, "HcclRankGraphGetLinks", StubHcclRankGraphGetLinks, g_hcclRankGraphGetLinksSupported);
    SET_PTR(hcclRankGraphGetTopoInstsByLayerPtr, "HcclRankGraphGetTopoInstsByLayer", StubHcclRankGraphGetTopoInstsByLayer, g_hcclRankGraphGetTopoInstsByLayerSupported);
    SET_PTR(hcclRankGraphGetTopoTypePtr, "HcclRankGraphGetTopoType", StubHcclRankGraphGetTopoType, g_hcclRankGraphGetTopoTypeSupported);
    SET_PTR(hcclRankGraphGetRanksByTopoInstPtr, "HcclRankGraphGetRanksByTopoInst", StubHcclRankGraphGetRanksByTopoInst, g_hcclRankGraphGetRanksByTopoInstSupported);
    SET_PTR(hcclGetHeterogModePtr, "HcclGetHeterogMode", StubHcclGetHeterogMode, g_hcclGetHeterogModeSupported);
    SET_PTR(hcclRankGraphGetEndpointNumPtr, "HcclRankGraphGetEndpointNum", StubHcclRankGraphGetEndpointNum, g_hcclRankGraphGetEndpointNumSupported);
    SET_PTR(hcclRankGraphGetEndpointDescPtr, "HcclRankGraphGetEndpointDesc", StubHcclRankGraphGetEndpointDesc, g_hcclRankGraphGetEndpointDescSupported);
    SET_PTR(hcclRankGraphGetEndpointInfoPtr, "HcclRankGraphGetEndpointInfo", StubHcclRankGraphGetEndpointInfo, g_hcclRankGraphGetEndpointInfoSupported);

    #undef SET_PTR
}

// 销毁函数：将指针重置为桩函数（可选，与 HcclResDlFini 配合使用）
void HcclRankGraphDlFini(void) {
    hcclRankGraphGetLayersPtr = StubHcclRankGraphGetLayers;
    hcclRankGraphGetRanksByLayerPtr = StubHcclRankGraphGetRanksByLayer;
    hcclRankGraphGetRankSizeByLayerPtr = StubHcclRankGraphGetRankSizeByLayer;
    hcclRankGraphGetTopoTypeByLayerPtr = StubHcclRankGraphGetTopoTypeByLayer;
    hcclRankGraphGetInstSizeListByLayerPtr = StubHcclRankGraphGetInstSizeListByLayer;
    hcclRankGraphGetLinksPtr = StubHcclRankGraphGetLinks;
    hcclRankGraphGetTopoInstsByLayerPtr = StubHcclRankGraphGetTopoInstsByLayer;
    hcclRankGraphGetTopoTypePtr = StubHcclRankGraphGetTopoType;
    hcclRankGraphGetRanksByTopoInstPtr = StubHcclRankGraphGetRanksByTopoInst;
    hcclGetHeterogModePtr = StubHcclGetHeterogMode;
    hcclRankGraphGetEndpointNumPtr = StubHcclRankGraphGetEndpointNum;
    hcclRankGraphGetEndpointDescPtr = StubHcclRankGraphGetEndpointDesc;
    hcclRankGraphGetEndpointInfoPtr = StubHcclRankGraphGetEndpointInfo;
}

// ---------- 对外API实现（通过函数指针转发）----------
HcclResult HcclRankGraphGetLayers(HcclComm comm, uint32_t** netLayers, uint32_t* netLayerNum) {
    return hcclRankGraphGetLayersPtr(comm, netLayers, netLayerNum);
}
HcclResult HcclRankGraphGetRanksByLayer(HcclComm comm, uint32_t netLayer, uint32_t** ranks, uint32_t* rankNum) {
    return hcclRankGraphGetRanksByLayerPtr(comm, netLayer, ranks, rankNum);
}
HcclResult HcclRankGraphGetRankSizeByLayer(HcclComm comm, uint32_t netLayer, uint32_t* rankNum) {
    return hcclRankGraphGetRankSizeByLayerPtr(comm, netLayer, rankNum);
}
HcclResult HcclRankGraphGetTopoTypeByLayer(HcclComm comm, uint32_t netLayer, CommTopo* topoType) {
    return hcclRankGraphGetTopoTypeByLayerPtr(comm, netLayer, topoType);
}
HcclResult HcclRankGraphGetInstSizeListByLayer(HcclComm comm, uint32_t netLayer, uint32_t** instSizeList, uint32_t* listSize) {
    return hcclRankGraphGetInstSizeListByLayerPtr(comm, netLayer, instSizeList, listSize);
}
HcclResult HcclRankGraphGetLinks(HcclComm comm, uint32_t netLayer, uint32_t srcRank, uint32_t dstRank,
                                 CommLink** links, uint32_t* linkNum) {
    return hcclRankGraphGetLinksPtr(comm, netLayer, srcRank, dstRank, links, linkNum);
}
HcclResult HcclRankGraphGetTopoInstsByLayer(HcclComm comm, uint32_t netLayer, uint32_t** topoInsts, uint32_t* topoInstNum) {
    return hcclRankGraphGetTopoInstsByLayerPtr(comm, netLayer, topoInsts, topoInstNum);
}
HcclResult HcclRankGraphGetTopoType(HcclComm comm, uint32_t netLayer, uint32_t topoInstId, CommTopo* topoType) {
    return hcclRankGraphGetTopoTypePtr(comm, netLayer, topoInstId, topoType);
}
HcclResult HcclRankGraphGetRanksByTopoInst(HcclComm comm, uint32_t netLayer, uint32_t topoInstId,
                                          uint32_t** ranks, uint32_t* rankNum) {
    return hcclRankGraphGetRanksByTopoInstPtr(comm, netLayer, topoInstId, ranks, rankNum);
}
HcclResult HcclGetHeterogMode(HcclComm comm, HcclHeterogMode* mode) {
    return hcclGetHeterogModePtr(comm, mode);
}
HcclResult HcclRankGraphGetEndpointNum(HcclComm comm, uint32_t layer, uint32_t topoInstId, uint32_t* num) {
    return hcclRankGraphGetEndpointNumPtr(comm, layer, topoInstId, num);
}
HcclResult HcclRankGraphGetEndpointDesc(HcclComm comm, uint32_t layer, uint32_t topoInstId,
                                        uint32_t* descNum, EndpointDesc* endpointDesc) {
    return hcclRankGraphGetEndpointDescPtr(comm, layer, topoInstId, descNum, endpointDesc);
}
HcclResult HcclRankGraphGetEndpointInfo(HcclComm comm, uint32_t rankId, const EndpointDesc* endpointDesc,
                                        EndpointAttr endpointAttr, uint32_t infoLen, void* info) {
    return hcclRankGraphGetEndpointInfoPtr(comm, rankId, endpointDesc, endpointAttr, infoLen, info);
}

// ---------- 对外提供的查询接口（判断函数是否存在）----------
extern "C" bool HcommIsSupportHcclRankGraphGetLayers(void) {
    return g_hcclRankGraphGetLayersSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetRanksByLayer(void) {
    return g_hcclRankGraphGetRanksByLayerSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetRankSizeByLayer(void) {
    return g_hcclRankGraphGetRankSizeByLayerSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetTopoTypeByLayer(void) {
    return g_hcclRankGraphGetTopoTypeByLayerSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetInstSizeListByLayer(void) {
    return g_hcclRankGraphGetInstSizeListByLayerSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetLinks(void) {
    return g_hcclRankGraphGetLinksSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetTopoInstsByLayer(void) {
    return g_hcclRankGraphGetTopoInstsByLayerSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetTopoType(void) {
    return g_hcclRankGraphGetTopoTypeSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetRanksByTopoInst(void) {
    return g_hcclRankGraphGetRanksByTopoInstSupported;
}
extern "C" bool HcommIsSupportHcclGetHeterogMode(void) {
    return g_hcclGetHeterogModeSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetEndpointNum(void) {
    return g_hcclRankGraphGetEndpointNumSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetEndpointDesc(void) {
    return g_hcclRankGraphGetEndpointDescSupported;
}
extern "C" bool HcommIsSupportHcclRankGraphGetEndpointInfo(void) {
    return g_hcclRankGraphGetEndpointInfoSupported;
}