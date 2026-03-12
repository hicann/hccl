/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ins_v2_all_to_all_v_sole_executor.h"
#include "ins_temp_all_to_all_v_mesh_1D.h"
#ifndef AICPU_COMPILE
#include "aiv_temp_all_to_all_mesh_1D.h"
#include "aiv_temp_all_to_all_v_mesh_1D.h"
#include "ccu_temp_all_to_all_mesh_1D.h"
#include "ccu_temp_all_to_all_mesh2die.h"
#include "ccu_temp_all_to_all_mesh1d_multi_jetty.h"
#endif

#define CONST_ZERO 0
#define CONST_ONE 1
#define CONST_TWO 2
#define CONST_THREE 3
namespace ops_hccl {

template <typename AlgTopoMatch, typename InsAlgTemplate>
InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::InsV2AlltoAllVSoleExecutor()
{}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcAlgHierarchyInfo(HcclComm comm,
    TopoInfoWithNetLayerDetails* topoInfo,
    AlgHierarchyInfoForAllLevel& algHierarchyInfo)
{
    // 使用topo match计算AlgHierarchyInfoForAllLevel
    AlgTopoMatch topoMatch;
    CHK_RET(topoMatch.MatchTopo(comm, topoInfo, algHierarchyInfo));
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::CalcRes(
    HcclComm comm, const OpParam& param,
    const TopoInfoWithNetLayerDetails* topoInfo, const AlgHierarchyInfoForAllLevel& algHierarchyInfo,
    AlgResourceRequest& resourceRequest)
{
    std::vector<std::vector<u32>> tempAlgHierachyInfo;
    if (topoInfo->level0Topo == Level0Shape::MESH_1D_CLOS) {
        tempAlgHierachyInfo.push_back(algHierarchyInfo.infos[0][1]);    // clos拓扑，包含所有rank
    } else {
        tempAlgHierachyInfo = algHierarchyInfo.infos[0];
    }
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, topoInfo->userRank, tempAlgHierachyInfo);
    // 调用计算资源的函数
    algTemplate->CalcRes(comm, param, topoInfo, resourceRequest);
    return HCCL_SUCCESS;
}

template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::Orchestrate(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][Orchestrate] Orchestrate Start");

    // maxTmpMemSize_设定为cclIn的大小，op中将申请的HcclBuff全给了cclIn
    maxTmpMemSize_ = resCtx.cclMem.size;
    // 给channels_和threads_赋值
    threads_ = resCtx.threads;
    if (param.engine != CommEngine::COMM_ENGINE_AIV && param.engine != CommEngine::COMM_ENGINE_CCU) {
        CHK_RET(RestoreChannelMap(resCtx, remoteRankToChannelInfo_));
    }

    dataType_ = param.all2AllVDataDes.sendType;
    dataTypeSize_ = DATATYPE_SIZE_TABLE[dataType_];
    rankSize_ = resCtx.topoInfo.userRankSize;
    sendTypeSize_ = DATATYPE_SIZE_TABLE[param.all2AllVDataDes.sendType];
    recvTypeSize_ = DATATYPE_SIZE_TABLE[param.all2AllVDataDes.recvType];
    dataSize_ = dataCount_ * dataTypeSize_;

    // Init sendRevc data for alltoall/alltoallV/alltoallVC algorithm
    CHK_PRT_RET(param.varMemSize != ALL_TO_ALL_V_VECTOR_NUM * rankSize_ * sizeof(u64),
        HCCL_ERROR("[CalcAlltoAllVSendRecvInfo] param.varMemSize [%llu] is invalid", param.varMemSize),
        HCCL_E_PARA);
    localSendRecvInfo_.sendCounts.resize(rankSize_, 0);
    localSendRecvInfo_.sendDispls.resize(rankSize_, 0);
    localSendRecvInfo_.recvCounts.resize(rankSize_, 0);
    localSendRecvInfo_.recvDispls.resize(rankSize_, 0);
    if (param.engine == COMM_ENGINE_CCU) {
        const u64* data = reinterpret_cast<const u64*>(param.varData);
        // 从varData把值取出来
        for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize_; i++) {
            HCCL_INFO("OrchestrateLoop, param.varData[%u] is [%u]", i, data[i]);
        }
        for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize_; i++) {
            u64 val = i / rankSize_;
            u64 curRank = i % rankSize_;
            switch(val) {
                case CONST_ZERO:
                    localSendRecvInfo_.sendCounts[curRank] = data[i];
                    break;
                case CONST_ONE:
                    localSendRecvInfo_.recvCounts[curRank] = data[i];
                    break;
                case CONST_TWO:
                    localSendRecvInfo_.sendDispls[curRank] = data[i];
                    break;
                case CONST_THREE:
                    localSendRecvInfo_.recvDispls[curRank] = data[i];
                    break;
                default:
                    break;
            }
        }
    } else {
        for (u32 j = 0; j < rankSize_; j++) {
            // Send info
            u64 curSendCounts = *(static_cast<const u64 *>(param.all2AllVDataDes.sendCounts) + j);
            u64 curSendDispls = *(static_cast<const u64 *>(param.all2AllVDataDes.sdispls) + j);
            localSendRecvInfo_.sendCounts[j] = curSendCounts;
            localSendRecvInfo_.sendDispls[j] = curSendDispls;

            // Recv info
            u64 curRecvCounts = *(static_cast<const u64 *>(param.all2AllVDataDes.recvCounts) + j);
            u64 curRecvDispls = *(static_cast<const u64 *>(param.all2AllVDataDes.rdispls) + j);
            localSendRecvInfo_.recvCounts[j] = curRecvCounts;
            localSendRecvInfo_.recvDispls[j] = curRecvDispls;
        }
    }

    HcclResult ret = OrchestrateLoop(param, resCtx);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][Orchestrate]errNo[0x%016llx] AlltoAll excutor kernel run failed",
            HCCL_ERROR_CODE(ret)), ret);
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][Orchestrate] Orchestrate End.");
    return HCCL_SUCCESS;
}

// 切分数据并调用 template
template <typename AlgTopoMatch, typename InsAlgTemplate>
HcclResult InsV2AlltoAllVSoleExecutor<AlgTopoMatch, InsAlgTemplate>::OrchestrateLoop(
    const OpParam &param, const AlgResourceCtxSerializable &resCtx)
{
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][OrchestrateLoop] Start");

    TemplateResource templateAlgRes;
    if (remoteRankToChannelInfo_.size() > 0) {
        templateAlgRes.channels = remoteRankToChannelInfo_[0];
    }
    if (param.engine == COMM_ENGINE_CCU) {
        templateAlgRes.ccuKernels = resCtx.ccuKernels;
    }

    templateAlgRes.threads = resCtx.threads;
    templateAlgRes.aivCommInfoPtr = resCtx.aivCommInfoPtr;
    // 准备数据
    TemplateDataParams tempAlgParams;
    tempAlgParams.buffInfo.inputPtr = param.inputPtr;
    tempAlgParams.buffInfo.outputPtr = param.outputPtr;
    tempAlgParams.buffInfo.inputSize = param.inputSize;
    tempAlgParams.buffInfo.outputSize = param.outputSize;
    tempAlgParams.buffInfo.hcclBuff = resCtx.cclMem;
    tempAlgParams.buffInfo.inBuffType = BufferType::INPUT;
    tempAlgParams.buffInfo.outBuffType = BufferType::OUTPUT;
    tempAlgParams.buffInfo.hcclBuffType = BufferType::HCCL_BUFFER;

    // RestoreVarDataAlltoAllV 已经将数据放到对应的指针
    std::vector<u64> sendCounts(rankSize_, 0);
    std::vector<u64> recvCounts(rankSize_, 0);
    std::vector<u64> sdispls(rankSize_, 0);
    std::vector<u64> rdispls(rankSize_, 0);
    if (param.engine == COMM_ENGINE_CCU) {
        const u64* data = reinterpret_cast<const u64*>(param.varData);
        // 从varData把值取出来
        for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize_; i++) {
            HCCL_INFO("OrchestrateLoop, param.varData[%u] is [%u]", i, data[i]);
        }
        for (u64 i = 0; i < ALL_TO_ALL_V_VECTOR_NUM * rankSize_; i++) {
            u64 val = i / rankSize_;
            u64 curRank = i % rankSize_;
            switch(val) {
                case CONST_ZERO:
                    sendCounts[curRank] = data[i];
                    break;
                case CONST_ONE:
                    recvCounts[curRank] = data[i];
                    break;
                case CONST_TWO:
                    sdispls[curRank] = data[i];
                    break;
                case CONST_THREE:
                    rdispls[curRank] = data[i];
                    break;
                default:
                    break;
            }
        }
    } else {
        for (u64 i = 0; i < rankSize_; i++) {
            sendCounts[i] = reinterpret_cast<u64*>(param.all2AllVDataDes.sendCounts)[i];
            recvCounts[i] = reinterpret_cast<u64*>(param.all2AllVDataDes.recvCounts)[i];
            sdispls[i] = reinterpret_cast<u64*>(param.all2AllVDataDes.sdispls)[i];
            rdispls[i] = reinterpret_cast<u64*>(param.all2AllVDataDes.rdispls)[i];
        }
    }

    std::vector<std::vector<u32>> tempAlgHierachyInfo;
    if (resCtx.topoInfo.level0Topo == Level0Shape::MESH_1D_CLOS) {
        tempAlgHierachyInfo.push_back(resCtx.algHierarchyInfo.infos[0][1]);
    } else {
        tempAlgHierachyInfo = resCtx.algHierarchyInfo.infos[0];
    }
    // 构建template
    std::shared_ptr<InsAlgTemplate> algTemplate =
        std::make_shared<InsAlgTemplate>(param, resCtx.topoInfo.userRank, tempAlgHierachyInfo);
    u32 templateScratchMultiplier = algTemplate->CalcScratchMultiple(tempAlgParams.buffInfo.inBuffType,
                                                                     tempAlgParams.buffInfo.outBuffType);

    // 计算最小传输大小
    u64 maxDataSizePerLoop = 0;
    maxTmpMemSize_ = tempAlgParams.buffInfo.hcclBuff.size;
    u64 transportBoundDataSize = UB_MAX_DATA_SIZE;
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor]maxTmpMemSize_ [%u]", maxTmpMemSize_);
    if (templateScratchMultiplier != 0) {
        u64 scratchBoundDataSize = maxTmpMemSize_ / templateScratchMultiplier / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN;
        maxDataSizePerLoop = std::min(transportBoundDataSize, scratchBoundDataSize);
    } else {
        maxDataSizePerLoop = transportBoundDataSize;
    }
    // 单次循环处理的数据量大小
    u64 maxDataCountPerLoop = maxDataSizePerLoop / dataTypeSize_ / rankSize_; // 发往单卡的数据量
    HCCL_INFO(
        "[InsV2AlltoAllVSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop[%llu], maxDataSizePerLoop[%llu], "
        "transportBoundDataSize[%llu], templateScratchMultiplier[%llu]",
        maxDataCountPerLoop, maxDataSizePerLoop, transportBoundDataSize, templateScratchMultiplier);
    CHK_PRT_RET(maxDataCountPerLoop == 0,
        HCCL_ERROR("[InsV2AlltoAllVSoleExecutor][OrchestrateOpbase] maxDataCountPerLoop is 0"), HCCL_E_INTERNAL);

    u64 maxSendOrRecvDataCount = 0;
    for (u64 i = 0; i < rankSize_; i++) {
        maxSendOrRecvDataCount = std::max(maxSendOrRecvDataCount, sendCounts[i]);
        maxSendOrRecvDataCount = std::max(maxSendOrRecvDataCount, recvCounts[i]);
    }
    HCCL_INFO("[InsV2AlltoAllVSoleExecutor] maxSendOrRecvDataCount[%u]", maxSendOrRecvDataCount);

    // 计算loopTimes，alltoallv的时候，有些算子的loopTimes可能是0
    u64 loopTimes = maxSendOrRecvDataCount / maxDataCountPerLoop +
        static_cast<u64>(maxSendOrRecvDataCount % maxDataCountPerLoop != 0);
    u64 processedDataCount = 0;
    for (u64 loop = 0; loop < loopTimes; loop++) {
        u64 currDataCount = (loop == loopTimes - 1) ? maxSendOrRecvDataCount - processedDataCount : maxDataCountPerLoop;

        tempAlgParams.count = currDataCount;
        tempAlgParams.buffInfo.inBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.outBuffBaseOff = processedDataCount * dataTypeSize_;
        tempAlgParams.buffInfo.hcclBuffBaseOff = 0;
        tempAlgParams.processedDataCount = processedDataCount;

        tempAlgParams.sliceSize = currDataCount * dataTypeSize_; // 这是每次循环处理的数据大小
        tempAlgParams.tailSize = tempAlgParams.sliceSize;
        // 这里的stride当成传统意义上的sreide 间隔
        tempAlgParams.inputSliceStride = 0; // 变长算子不涉及,这里是每一块数据的大小，这个值被sendCounts代替了
        // 这里用来放每张卡可以用的cclBuffer的大小，数据从ureIn到cclBuffer的时候，以这个量来分隔
        tempAlgParams.outputSliceStride = maxDataCountPerLoop * dataTypeSize_;

        HCCL_INFO("[InsV2AlltoAllVSoleExecutor] loop [%u] tempAlgParams.inputSliceStride [%u],"
            "tempAlgParams.outputSliceStride [%u] tempAlgParams.sliceSize [%u]",
            loop, tempAlgParams.inputSliceStride, tempAlgParams.outputSliceStride, tempAlgParams.sliceSize);
        HCCL_INFO("[InsV2AlltoAllVSoleExecutor] loop [%u] tempAlgParams.buffInfo.inBuffBaseOff [%u],"
            "tempAlgParams.buffInfo.outBuffBaseOff [%u]",
            loop, tempAlgParams.buffInfo.inBuffBaseOff, tempAlgParams.buffInfo.outBuffBaseOff);
        // 不需要重复
        tempAlgParams.repeatNum = 1;
        tempAlgParams.inputRepeatStride = 0;
        tempAlgParams.outputRepeatStride = 0;

        tempAlgParams.sendCounts.resize(rankSize_, 0);
        tempAlgParams.recvCounts.resize(rankSize_, 0);
        tempAlgParams.sdispls.resize(rankSize_, 0);
        tempAlgParams.rdispls.resize(rankSize_, 0);

        for (u64 i = 0; i < rankSize_; i++) {
            if (sendCounts[i] > processedDataCount) {
                tempAlgParams.sendCounts[i] = std::min(currDataCount, sendCounts[i] - processedDataCount);
                tempAlgParams.sdispls[i] = sdispls[i] + processedDataCount;
            } else {
                tempAlgParams.sendCounts[i] = 0;
                tempAlgParams.sdispls[i] = sdispls[i] + sendCounts[i];
            }

            if (recvCounts[i] > processedDataCount) {
                tempAlgParams.recvCounts[i] = std::min(currDataCount, recvCounts[i] - processedDataCount);
                tempAlgParams.rdispls[i] = rdispls[i] + processedDataCount;
            } else {
                tempAlgParams.recvCounts[i] = 0;
                tempAlgParams.rdispls[i] = rdispls[i] + recvCounts[i];
            }
        }

        // 因为只考虑执行0级算法，所以传进template里面的channels就是channels_的第一个vector
        CHK_RET(algTemplate->KernelRun(param, tempAlgParams, templateAlgRes));
        processedDataCount += currDataCount;
    }

    HCCL_INFO("[InsV2AlltoAllVSoleExecutor][OrchestrateLoop] End.");
    return HCCL_SUCCESS;
}

REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALL, InsAlltoAllMesh1D, InsV2AlltoAllVSoleExecutor, TopoMatch1D,
    InsTempAlltoAllVMesh1D);
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLV, InsAlltoAllVMesh1D, InsV2AlltoAllVSoleExecutor, TopoMatch1D,
    InsTempAlltoAllVMesh1D);
REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLVC, InsAlltoAllVCMesh1D, InsV2AlltoAllVSoleExecutor, TopoMatch1D,
    InsTempAlltoAllVMesh1D);
#ifndef AICPU_COMPILE
    REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALL, CcuAlltoAllMesh1D, InsV2AlltoAllVSoleExecutor, TopoMatch1D,
        CcuTempAlltoAllMesh1D);
    REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALL, AivAlltoAllMesh1D, InsV2AlltoAllVSoleExecutor, TopoMatch1D,
                     AivTempAlltoAllMesh1D);
    REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALLV, AivAlltoAllVMesh1D, InsV2AlltoAllVSoleExecutor, TopoMatch1D,
                     AivTempAlltoAllVMesh1D);
    REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALL, CcuAllToAllMesh2Die, InsV2AlltoAllVSoleExecutor, TopoMatch1D,
    CcuTempAllToAllMesh2Die);
    REGISTER_EXEC_V2(HcclCMDType::HCCL_CMD_ALLTOALL, CcuAlltoAllMesh1DMultiJetty, InsV2AlltoAllVSoleExecutor,
                    TopoMatchUBX, CcuTempAllToAllMesh1dMultiJetty);
#endif
}  // namespace Hccl