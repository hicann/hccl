/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_2DIE_H_
#define HCCLV2_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_2DIE_H_

#include <array>
#include <set>
#include "utils.h"
#include "ccu_alg_template_base.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {

using RankId = u32;
using RankGroup = std::vector<RankId>;

constexpr uint32_t MAX_KERNEL_NUM_2DIE = 3;
constexpr uint32_t KERNEL_FULLMESH = 0;
constexpr uint32_t KERNEL_CLOS_MAJOR = 1;
constexpr uint32_t KERNEL_CLOS_MINOR = 2;
constexpr uint32_t CLOS_RATIO_MINOR = 2;
constexpr uint32_t CLOS_RATIO_MAJOR = 6;
constexpr uint32_t CLOS_RATIO_TOTAL = 8;

struct Mesh2DieCacheCtx {
    uint32_t dieNum;
    bool is2Plus6;
    uint32_t kernelCount;
    std::vector<RankId> rankGroup[MAX_KERNEL_NUM_2DIE];
    std::set<RankId> closPeers;

    std::vector<char> Serialize() const
    {
        std::vector<char> buf;
        auto append = [&buf](const void *data, size_t len) {
            const char *p = static_cast<const char *>(data);
            buf.insert(buf.end(), p, p + len);
        };
        auto appendVec = [&buf](const std::vector<RankId> &v) {
            uint32_t sz = static_cast<uint32_t>(v.size());
            buf.insert(buf.end(), reinterpret_cast<const char *>(&sz), reinterpret_cast<const char *>(&sz) + sizeof(uint32_t));
            buf.insert(buf.end(), reinterpret_cast<const char *>(v.data()), reinterpret_cast<const char *>(v.data()) + sz * sizeof(RankId));
        };
        auto appendSet = [&buf](const std::set<RankId> &s) {
            uint32_t sz = static_cast<uint32_t>(s.size());
            buf.insert(buf.end(), reinterpret_cast<const char *>(&sz), reinterpret_cast<const char *>(&sz) + sizeof(uint32_t));
            for (auto &v : s) {
                buf.insert(buf.end(), reinterpret_cast<const char *>(&v), reinterpret_cast<const char *>(&v) + sizeof(RankId));
            }
        };
        append(&dieNum, sizeof(uint32_t));
        append(&is2Plus6, sizeof(bool));
        append(&kernelCount, sizeof(uint32_t));
        for (uint32_t i = 0; i < MAX_KERNEL_NUM_2DIE; i++) {
            appendVec(rankGroup[i]);
        }
        appendSet(closPeers);
        return buf;
    }

    HcclResult Deserialize(const char *buf, size_t len)
    {
        size_t off = 0;
        auto read = [&off, buf, len](void *dst, size_t n) -> errno_t {
            errno_t ret = memcpy_s(dst, n, buf + off, n);
            off += n;
            return ret;
        };
        auto readVec = [&off, buf, len](std::vector<RankId> &v) -> errno_t {
            uint32_t sz;
            errno_t ret = memcpy_s(&sz, sizeof(uint32_t), buf + off, sizeof(uint32_t));
            off += sizeof(uint32_t);
            if (ret != EOK) { return ret; }
            v.resize(sz);
            if (sz == 0) { return EOK; }
            ret = memcpy_s(v.data(), sz * sizeof(RankId), buf + off, sz * sizeof(RankId));
            off += sz * sizeof(RankId);
            return ret;
        };
        auto readSet = [&off, buf, len](std::set<RankId> &s) -> errno_t {
            uint32_t sz;
            errno_t ret = memcpy_s(&sz, sizeof(uint32_t), buf + off, sizeof(uint32_t));
            off += sizeof(uint32_t);
            if (ret != EOK) { return ret; }
            s.clear();
            for (uint32_t i = 0; i < sz; i++) {
                RankId v;
                ret = memcpy_s(&v, sizeof(RankId), buf + off, sizeof(RankId));
                off += sizeof(RankId);
                if (ret != EOK) { return ret; }
                s.insert(v);
            }
            return EOK;
        };
        CHK_SAFETY_FUNC_RET(read(&dieNum, sizeof(uint32_t)));
        CHK_SAFETY_FUNC_RET(read(&is2Plus6, sizeof(bool)));
        CHK_SAFETY_FUNC_RET(read(&kernelCount, sizeof(uint32_t)));
        for (uint32_t i = 0; i < MAX_KERNEL_NUM_2DIE; i++) {
            CHK_SAFETY_FUNC_RET(readVec(rankGroup[i]));
        }
        CHK_SAFETY_FUNC_RET(readSet(closPeers));
        return HcclResult::HCCL_SUCCESS;
    }
};

class CcuTempAlltoAllVMesh1D2Die : public CcuAlgTemplateBase {
public:
    CcuTempAlltoAllVMesh1D2Die() = default;
    explicit CcuTempAlltoAllVMesh1D2Die(const OpParam &param, RankId rankId,
        const std::vector<std::vector<u32>> &subCommRanks);
    ~CcuTempAlltoAllVMesh1D2Die() override;

    std::string Describe() const override
    {
        return StringFormat("Template of alltoallv ccu mesh 1D 2Die with rankSize[%u]", templateRankSize_);
    }

    HcclResult CalcRes(HcclComm comm, const OpParam &param, const TopoInfoWithNetLayerDetails *topoInfo,
        AlgResourceRequest &resourceRequest) override;

    HcclResult KernelRun(const OpParam &param, const TemplateDataParams &templateDataParams,
        TemplateResource& templateResource) override;

    HcclResult FastLaunch(const OpParam &param, const TemplateFastLaunchCtx &tempFastLaunchCtx) override;

    void SetA2ASendRecvInfo(const A2ASendRecvInfo &sendRecvInfo);

private:
    HcclResult PartitionChannels(HcclComm comm, const std::vector<HcclChannelDesc> &channelDescs,
                                std::map<u32, std::vector<HcclChannelDesc>>& rankIdToChannelDesc);
    void FillRankGroupTaskArgs(uint32_t kernelIdx, const Mesh2DieCacheCtx &cacheCtx,
        const LoopGroupConfig &config, std::vector<uint64_t> &taskArgs);
    HcclResult SaveCacheCtx(HcclComm comm, const OpParam &param);
    HcclResult LoadCacheCtx(const OpParam &param, Mesh2DieCacheCtx &cacheCtx);

    const uint32_t DIE_NUM = 2;

    bool is2Plus6_ = false;
    uint32_t kernelCount_ = 2;
    uint32_t fullmeshDieId_ = 0;

    std::vector<std::vector<HcclChannelDesc>> kernelChannels_{MAX_KERNEL_NUM_2DIE};
    std::array<RankGroup, MAX_KERNEL_NUM_2DIE> kernelRankGroup_;
    std::array<bool, MAX_KERNEL_NUM_2DIE> kernelWithMyRank_ = {true, false, false};
    std::array<uint32_t, MAX_KERNEL_NUM_2DIE> kernelType_ = {KERNEL_FULLMESH, KERNEL_CLOS_MAJOR, KERNEL_CLOS_MINOR};
    std::map<uint32_t, std::vector<HcclChannelDesc>> rankIdToChannelDesc_;
    std::set<RankId> closPeers_;

    A2ASendRecvInfo localSendRecvInfo_;
};

} // namespace ops_hccl
#endif // HCCLV2_CCU_TEMP_ALL_TO_ALL_V_MESH_1D_2DIE_H_
