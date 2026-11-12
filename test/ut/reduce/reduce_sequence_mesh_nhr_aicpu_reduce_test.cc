/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Reduce 分层算法 AicpuReduceSequenceMeshNHRAicpuReduce（PCIE 标卡 fp64/uint64）UT：
// 覆盖名解析、注册 attrs 契约、selector 选路、CalcCostCoeff 标定完备性与空段守卫、
// Orchestrate 编排（含 in-place 时序安全）。各用例组由 TEST 名自述。
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include "alg_attrs_registry.h"
#include "alg_param.h"
#include "alg_parse.h"
#include "coll_alg_v2_exec_registry.h"
#include "reduce_auto_selector.h"
#include "topo_match_test_common.h"

namespace ops_hccl {
// 桩内拓扑全互连开关（stub 文件定义）
extern bool g_utStubLayerAllConnected;
} // namespace ops_hccl

namespace {

using ops_hccl::AlgAttrs;
using ops_hccl::AlgAttrsRegistry;
using ops_hccl::AlgoType;
using ops_hccl::AlgResourceCtxSerializable;
using ops_hccl::ChannelInfo;
using ops_hccl::CollAlgExecRegistryV2;
using ops_hccl::InsCollAlgBase;
using ops_hccl::Level0Shape;
using ops_hccl::OpExecuteConfig;
using ops_hccl::OpParam;
using ops_hccl::ReduceAutoSelector;
using ops_hccl::SelectorStatus;
using ops_hccl::TopoInfoWithNetLayerDetails;

using ops_hccl::LEVEL0_TOPO_MESH_1D_CLOS;
using ops_hccl::TOPO_LEVEL_NUM_1;

const std::string SEQUENCE_MESH_NHR_AICPU_REDUCE_ALG = "AicpuReduceSequenceMeshNHRAicpuReduce";
const std::string SOLE_NHR_AICPU_REDUCE_ALG = "AicpuReduceSoleNHRAicpuReduce";
// 同执行器在产算法，作 costmodel 正控
const std::string CONCUR_NHR_ALG = "AicpuReduceSequenceMeshConcurNHR";

// 目标机型夹具：level0=LOCAL 视图卡内组（1DMESH），level1=GLOBAL 全域（CLOS 即 PCIE 混连面），
// 单网络层（topoLevelNums=1），为新算法注册域内形态。
TopoInfoWithNetLayerDetails MakeTwoLevelCostProbeTopo(u32 level0GroupSize = 4, u32 userRankSize = 16)
{
    using ops_hccl::ut_helper::MakeLevel;
    using ops_hccl::ut_helper::MakeTopoInfo;
    using ops_hccl::ut_helper::Range;
    auto topo = MakeTopoInfo(
        0, userRankSize,
        {MakeLevel(
             ops_hccl::ut_helper::RangeFrom(0, level0GroupSize), ops_hccl::PhysicalLevelView::LOCAL,
             std::vector<u32>(userRankSize / level0GroupSize, level0GroupSize)),
         MakeLevel(Range(userRankSize), ops_hccl::PhysicalLevelView::GLOBAL, {userRankSize})});
    topo.topoLevelNums = 1; // 单网络层机型（注册域 maxTopoLevelNum=1）
    topo.physicalLevels[0].topoType = CommTopo::COMM_TOPO_1DMESH;
    topo.physicalLevels[0].portNums = {8};
    topo.physicalLevels[1].topoType = CommTopo::COMM_TOPO_CLOS;
    topo.physicalLevels[1].portNums = {8};
    topo.isPod = false;
    return topo;
}

} // namespace

TEST(ReduceMeshNhrAicpuReduceNameParse, SequenceMeshNhrAicpuReduceAttrs)
{
    AlgAttrs attrs;
    AlgAttrsRegistry::ParseAlgName(SEQUENCE_MESH_NHR_AICPU_REDUCE_ALG, attrs);

    ASSERT_EQ(attrs.name, SEQUENCE_MESH_NHR_AICPU_REDUCE_ALG);
    ASSERT_EQ(attrs.algoTypes.size(), 2u);
    EXPECT_EQ(attrs.algoTypes[0], AlgoType::MESH);
    EXPECT_EQ(attrs.algoTypes[1], AlgoType::NHR_AICPU_REDUCE);
    EXPECT_EQ(attrs.engine, OpExecuteConfig::AICPU_TS);
    EXPECT_EQ(attrs.opType, HcclCMDType::HCCL_CMD_REDUCE);
}

TEST(ReduceMeshNhrAicpuReduceNameParse, SoleNhrAicpuReduceAttrs)
{
    AlgAttrs attrs;
    AlgAttrsRegistry::ParseAlgName(SOLE_NHR_AICPU_REDUCE_ALG, attrs);

    ASSERT_EQ(attrs.algoTypes.size(), 1u);
    EXPECT_EQ(attrs.algoTypes[0], AlgoType::NHR_AICPU_REDUCE);
    EXPECT_EQ(attrs.engine, OpExecuteConfig::AICPU_TS);
    EXPECT_EQ(attrs.opType, HcclCMDType::HCCL_CMD_REDUCE);
}

TEST(ReduceMeshNhrAicpuReduceExecRegistry, SequenceMeshNhrAicpuReduceIsRegistered)
{
    std::unique_ptr<ops_hccl::InsCollAlgBase> exec = CollAlgExecRegistryV2::Instance().GetAlgExec(
        HcclCMDType::HCCL_CMD_REDUCE, SEQUENCE_MESH_NHR_AICPU_REDUCE_ALG);
    EXPECT_NE(exec.get(), nullptr);
}

TEST(ReduceMeshNhrAicpuReduceAttrsRegistry, SequenceMeshNhrAicpuReduceTopoAttrsContract)
{
    const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(SEQUENCE_MESH_NHR_AICPU_REDUCE_ALG);
    ASSERT_NE(attrs, nullptr);

    EXPECT_EQ(attrs->topo.minTopoLevelNum, TOPO_LEVEL_NUM_1);
    EXPECT_EQ(attrs->topo.maxTopoLevelNum, TOPO_LEVEL_NUM_1);
    // 框间为 NHR 系模板，L1 报 NHR 形态的拓扑不应被拦
    EXPECT_EQ(attrs->topo.supportLevel0Topos, LEVEL0_TOPO_MESH_1D_CLOS);
    EXPECT_TRUE(attrs->topo.isSupportLevel0PcieMix);
    EXPECT_TRUE(attrs->topo.isSupportLevel1Nhr);
    EXPECT_TRUE(attrs->op.isSupportProd);
    EXPECT_TRUE(attrs->op.isSupportInplace);
    EXPECT_TRUE(attrs->op.unsupportedDataTypes.empty());
    EXPECT_TRUE(attrs->op.supportedDataTypes.empty());
    ASSERT_NE(attrs->op.opCustomCheck, nullptr);
    auto admit = [&attrs](HcclDataType dtype, HcclReduceOp reduceOp) {
        ops_hccl::OpParam param{};
        param.DataDes.dataType = dtype;
        param.reduceType = reduceOp;
        TopoInfoWithNetLayerDetails topo;
        return attrs->op.opCustomCheck(param, &topo);
    };
    EXPECT_TRUE(admit(HcclDataType::HCCL_DATA_TYPE_INT64, HcclReduceOp::HCCL_REDUCE_SUM));
    EXPECT_TRUE(admit(HcclDataType::HCCL_DATA_TYPE_UINT64, HcclReduceOp::HCCL_REDUCE_SUM));
    EXPECT_TRUE(admit(HcclDataType::HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM));
    EXPECT_TRUE(admit(HcclDataType::HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_PROD));
    EXPECT_TRUE(admit(HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_PROD)); // 非 64bit PROD 不被挡
    EXPECT_FALSE(admit(HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM)); // 32bit SUM 域外

    // topoCustomCheck：仅 CLOS+pcieMix+非全互连为 true
    TopoInfoWithNetLayerDetails topo;
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo.level0PcieMix = true;
    ops_hccl::g_utStubLayerAllConnected = false; // 非全互连
    EXPECT_TRUE(attrs->topo.topoCustomCheck(&topo));
    ops_hccl::g_utStubLayerAllConnected = true; // 全互连
    EXPECT_FALSE(attrs->topo.topoCustomCheck(&topo));
    ops_hccl::g_utStubLayerAllConnected = false;
    topo.level0PcieMix = false; // 非 pcieMix CLOS：域外
    EXPECT_FALSE(attrs->topo.topoCustomCheck(&topo));
    topo.level0PcieMix = true;
    topo.level0Topo = Level0Shape::MESH_1D; // 非 CLOS：域外
    EXPECT_FALSE(attrs->topo.topoCustomCheck(&topo));
}

TEST(ReduceMeshNhrAicpuReduceSelector, MeshAlgoAicpuRoutesSpecialToNewAlgorithm)
{
    // 自初始化桩开关（桩默认 true=全互连），不依赖其它用例状态
    ops_hccl::g_utStubLayerAllConnected = false;
    TopoInfoWithNetLayerDetails topo;
    topo.topoLevelNums = 1; // 单网络层机型
    topo.level0Topo = Level0Shape::MESH_1D_CLOS;
    topo.level0PcieMix = true;

    ReduceAutoSelector selector;
    const std::map<HcclCMDType, std::vector<ops_hccl::HcclAlgoType>> emptyConfig;
    auto select = [&selector, &topo, &emptyConfig](HcclDataType dtype, HcclReduceOp op, std::string& name) {
        ops_hccl::OpParam param{};
        param.DataDes.dataType = dtype;
        param.DataDes.count = 1024; // 8KB，避开大小数据阈值分支
        param.reduceType = op;
        return selector.SelectAicpuAlgo(&topo, param, emptyConfig, name);
    };

    for (auto dtype :
         {HcclDataType::HCCL_DATA_TYPE_INT64, HcclDataType::HCCL_DATA_TYPE_UINT64, HcclDataType::HCCL_DATA_TYPE_FP64}) {
        std::string name;
        ASSERT_EQ(select(dtype, HcclReduceOp::HCCL_REDUCE_SUM, name), SelectorStatus::MATCH)
            << "dtype=" << static_cast<int>(dtype);
        EXPECT_EQ(name, SEQUENCE_MESH_NHR_AICPU_REDUCE_ALG) << "dtype=" << static_cast<int>(dtype);
    }
    // PROD（含 fp32 PROD）→ 新算法
    std::string name;
    ASSERT_EQ(select(HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_PROD, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, SEQUENCE_MESH_NHR_AICPU_REDUCE_ALG);

    // 名字面量与注册表交叉校验
    const AlgAttrs* routedAttrs = AlgAttrsRegistry::Instance().Get(name);
    ASSERT_NE(routedAttrs, nullptr);
    EXPECT_EQ(routedAttrs->opType, HcclCMDType::HCCL_CMD_REDUCE);

    // 对照：fp32 SUM 保持既有 ParallelMeshNHR
    ASSERT_EQ(select(HcclDataType::HCCL_DATA_TYPE_FP32, HcclReduceOp::HCCL_REDUCE_SUM, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "AicpuReduceParallelMeshNHR");

    // 对照：pcieMix 全互连走既有 SoleMesh 分支
    ops_hccl::g_utStubLayerAllConnected = true;
    ASSERT_EQ(select(HcclDataType::HCCL_DATA_TYPE_FP64, HcclReduceOp::HCCL_REDUCE_SUM, name), SelectorStatus::MATCH);
    EXPECT_EQ(name, "AicpuReduceSoleMesh");
    ops_hccl::g_utStubLayerAllConnected = false;
}

TEST(ReduceMeshNhrAicpuReduceCostGuard, CalibratedSegmentsJoinCostmodelAndBoundaryGuardSkips)
{
    auto topo = MakeTwoLevelCostProbeTopo();
    // 夹具须位于新算法注册域内（单网络层），用例断言才有域内效力
    ASSERT_EQ(topo.topoLevelNums, 1u);
    ops_hccl::OpParam param{};

    // 新算法全模板已标定，应产出非空代价参与 costmodel
    std::unique_ptr<InsCollAlgBase> exec = CollAlgExecRegistryV2::Instance().GetAlgExec(
        HcclCMDType::HCCL_CMD_REDUCE, SEQUENCE_MESH_NHR_AICPU_REDUCE_ALG);
    ASSERT_NE(exec.get(), nullptr);
    auto cost = exec->CalcCostCoeff(nullptr, &topo, SEQUENCE_MESH_NHR_AICPU_REDUCE_ALG.c_str(), param);
    EXPECT_FALSE(cost.empty()) << "全模板已标定应产出非空代价, 实际条目数=" << cost.size();
    // 段级断言：六段各 1 条；框间 NHRPcie 段公式逐项核对（p1 入拷贝 B=3.0、
    // reduceCopies=(rankSize-1)×dataRatio；p2 三 buffer 同址无拷贝）
    ASSERT_EQ(cost.size(), 6u) << "6 段应各产出 1 条";
    EXPECT_FLOAT_EQ(cost[1].A, 2.0f);
    EXPECT_FLOAT_EQ(cost[1].B, 3.0f);
    EXPECT_FLOAT_EQ(cost[1].C, 5.0f);
    EXPECT_GT(cost[1].D, 0.0f);
    EXPECT_FLOAT_EQ(cost[1].reduceCopies, 0.1875f);
    EXPECT_FLOAT_EQ(cost[2].A, 2.0f);
    EXPECT_FLOAT_EQ(cost[2].B, 0.0f);
    EXPECT_FLOAT_EQ(cost[2].reduceCopies, 0.0f);

    // 域内正控：在产算法标定域内非空，证明夹具有效、守卫未误伤
    std::unique_ptr<InsCollAlgBase> concurExec
        = CollAlgExecRegistryV2::Instance().GetAlgExec(HcclCMDType::HCCL_CMD_REDUCE, CONCUR_NHR_ALG);
    ASSERT_NE(concurExec.get(), nullptr);
    auto concurCost = concurExec->CalcCostCoeff(nullptr, &topo, CONCUR_NHR_ALG.c_str(), param);
    EXPECT_FALSE(concurCost.empty()) << "标定域内算法不应被守卫跳过";

    // 边界：level0 组 16 卡超 ZAxisDetour 标定域 rankSize∈[2,8]，空段守卫整体返回空
    auto oversizeTopo = MakeTwoLevelCostProbeTopo(16, 32);
    auto boundaryCost = concurExec->CalcCostCoeff(nullptr, &oversizeTopo, CONCUR_NHR_ALG.c_str(), param);
    // 守卫语义若后续拆分两类空段，须同步更新本断言（人评 #196701376）
    EXPECT_TRUE(boundaryCost.empty()) << "标定域外空段应触发守卫, 实际条目数=" << boundaryCost.size();
    // 用 CONCUR 探测因 ZAxisDetour（唯一 rankSize 有界模板）闭包属其系注册；
    // 守卫为执行器级语义，对新算法同样生效

    // 边界：level0 组 1 卡退化拓扑在两级匹配处前置拒绝（d0==1）
    auto singletonTopo = MakeTwoLevelCostProbeTopo(1, 16);
    auto degenerateCost = concurExec->CalcCostCoeff(nullptr, &singletonTopo, CONCUR_NHR_ALG.c_str(), param);
    EXPECT_TRUE(degenerateCost.empty()) << "d0==1 退化拓扑应在探测阶段被前置拒绝";
}

// ==== 桩数据面记录接口（定义于 reduce_sequence_mesh_nhr_aicpu_reduce_stub.cc）====
void StubResetDataPlaneCalls();
bool StubInplaceReadsBeforeWritesPerSlice(const void* buff);
size_t StubFirstCallIdxWithSrcBuff(const void* buff);
size_t StubLastCallIdxWithSrcBuff(const void* buff);
size_t StubFirstCallIdxWithDstBuff(const void* buff);
size_t StubCountCallsWithDstBuff(const void* buff);

// Orchestrate 编排冒烟（真实执行器 + 记录桩数据面）：四段编排跑通、缓冲绑定正确、
// in-place 暂存时序安全（userBuf 读先于写）。
TEST(ReduceMeshNhrAicpuReduceOrchestrate, StagedSequenceAndInplaceSafety)
{
    // VM 地真形态：16 rank，level0 = 4 组×4 rank，level1 = 1 组×16 rank（单网络层）
    const u32 groupSize = 4;
    const u32 groupNum = 4;
    const u32 rankSize = groupSize * groupNum;
    const u32 root = 3; // 非 0 root

    // 单轮（8KB）与多轮（32MB，cclMem 4MB 强制循环）两档：多轮档逐 slice 断言写不覆盖未读输入
    for (u64 dataCount : {1024ULL, 4ULL * 1024 * 1024}) {
        // 每档取新执行器实例：通道表为实例成员、跨调用追加叠加；档内 myRank 复用的少量叠加由满额线程表覆盖
        std::unique_ptr<InsCollAlgBase> exec = CollAlgExecRegistryV2::Instance().GetAlgExec(
            HcclCMDType::HCCL_CMD_REDUCE, SEQUENCE_MESH_NHR_AICPU_REDUCE_ALG);
        ASSERT_NE(exec.get(), nullptr);
        for (u32 myRank : {root, root + 2}) { // root 视角 / 非 root 视角
            OpParam param{};                  // 值初始化
            param.root = root;
            param.reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
            param.DataDes.count = dataCount;
            param.DataDes.dataType = HcclDataType::HCCL_DATA_TYPE_FP64;

            std::vector<uint8_t> userBuf(dataCount * sizeof(double) + 64, 0);
            param.inputPtr = userBuf.data();
            param.outputPtr = userBuf.data();

            AlgResourceCtxSerializable resCtx{};
            // infos 对齐生产两级匹配：infos[0]=本 rank 的 level0 组，infos[1]=代表组（同位 rank）
            const u32 myOffset = myRank % groupSize;
            resCtx.algHierarchyInfo.infos.resize(2);
            {
                std::vector<u32> group0;
                for (u32 r = 0; r < groupSize; r++) {
                    group0.push_back((myRank / groupSize) * groupSize + r);
                }
                resCtx.algHierarchyInfo.infos[0].push_back(group0);
            }
            {
                std::vector<u32> group1;
                for (u32 g = 0; g < groupNum; g++) {
                    group1.push_back(g * groupSize + myOffset);
                }
                resCtx.algHierarchyInfo.infos[1].push_back(group1);
            }
            // cclMem：4MB 足够单轮档的多段切片
            std::vector<uint8_t> cclBuff(4 * 1024 * 1024, 0);
            resCtx.cclMem.addr = cclBuff.data();
            resCtx.cclMem.size = cclBuff.size();
            resCtx.topoInfo.userRank = myRank;
            resCtx.topoInfo.userRankSize = rankSize;
            // 线程表按满额填充避免越界，句柄仅流向桩同步原语
            resCtx.threads.resize(rankSize);
            // 每层最小通道：level0=本组对端，level1=代表组对端；内容仅参与桩地址计算
            resCtx.channels.resize(resCtx.algHierarchyInfo.infos.size());
            for (u32 level = 0; level < resCtx.algHierarchyInfo.infos.size(); level++) {
                const std::vector<u32>& peers = resCtx.algHierarchyInfo.infos[level][0];
                for (u32 peer : peers) {
                    // 含 self（环上自有步的 loopback 语义）
                    ChannelInfo ch{};
                    ch.isValid = true;
                    ch.remoteRank = peer;
                    ch.remoteCclMem.addr = cclBuff.data();
                    ch.remoteCclMem.size = cclBuff.size();
                    ch.remoteOutput.addr = cclBuff.data();
                    ch.remoteOutput.size = cclBuff.size();
                    resCtx.channels[level].push_back(ch);
                }
            }

            StubResetDataPlaneCalls();
            HcclResult ret = exec->Orchestrate(param, resCtx);
            EXPECT_EQ(ret, HCCL_SUCCESS) << "编排应可完整跑通（数据面为桩，仅验证编排与绑定）";

            // in-place 时序：userBuf 读（RSL0 暂存）先于首次写（AGL0 root 回收）
            size_t lastRead = StubLastCallIdxWithSrcBuff(userBuf.data());
            size_t firstWrite = StubFirstCallIdxWithDstBuff(userBuf.data());
            EXPECT_NE(lastRead, SIZE_MAX) << "RSL0 应有读 userBuf 的暂存搬移";
            if (myRank == root) {
                EXPECT_NE(firstWrite, SIZE_MAX) << "root 的 AGL0 应有写 userBuf 的回收搬移";
                if (dataCount == 1024) {
                    // 单轮：全局读先于全局写
                    EXPECT_LT(lastRead, firstWrite) << "输入读取必须全部先于输出写入（in-place 安全前提）";
                } else {
                    // 多轮：逐 slice 配对顺序
                    EXPECT_TRUE(StubInplaceReadsBeforeWritesPerSlice(userBuf.data())) << "多轮循环下写覆盖了未读输入";
                }
            } else {
                EXPECT_EQ(StubCountCallsWithDstBuff(userBuf.data()), 0u) << "非 root 不应写 userOut";
            }
        }
    }
}
