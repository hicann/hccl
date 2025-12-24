/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_SRC_OPS_INC_COLL_ALG_PARAM
#define OPS_HCCL_SRC_OPS_INC_COLL_ALG_PARAM

#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include "hccl_common.h"
#include "hccl_types.h"
#include "alg_type.h"
#include "hcomm_primitives.h"
#include "hccl_res.h"
#include "hcomm_primitives.h"
#include "hccl_rank_graph.h"

// 解决与Hcomm仓合入问题，暂时定义为弱符号
#ifndef HCCL_CHANNEL_ABI
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
const uint32_t HCCL_CHANNEL_MAGIC_WORD = 0x0f0f0f0f;
const uint32_t HCCL_CHANNEL_VERSION = 1;

/**
 * @brief 兼容Abi字段结构体
 */
typedef struct {
    uint32_t version;
    uint32_t magicWord;
    uint32_t size;
    uint32_t reserved;
} HcclAbiHeader;

typedef struct {
    HcclAbiHeader header;
    uint32_t remoteRank;    ///< 远端rankId
    CommProtocol protocol;  ///< 通信协议
    uint32_t notifyNum;  ///< channel上使用的通知消息数量
    union {
        HccsAttr hccsAttr;
        RoCEAttr roceAttr;
        UbAttr ubAttr;
    };
} HcclChannelDesc;

inline void HcclChannelDescInit(HcclChannelDesc *channelDesc, uint32_t descNum)
{
    for (uint32_t idx = 0; idx < descNum; idx++) {
        if (channelDesc != nullptr) {
            // Abi字段初始化
            channelDesc->header.version     = HCCL_CHANNEL_VERSION;
            channelDesc->header.magicWord   = HCCL_CHANNEL_MAGIC_WORD;
            channelDesc->header.size        = sizeof(HcclChannelDesc);
            channelDesc->header.reserved    = 0;

            // HcclChannelDesc内容初始化
            channelDesc->remoteRank = ~0U;
            channelDesc->protocol   = COMM_PROTOCOL_RESERVED;
            channelDesc->notifyNum  = 0;
            (void)memset_s(&(channelDesc->hccsAttr), sizeof(HccsAttr), 0, sizeof(HccsAttr));
            (void)memset_s(&(channelDesc->roceAttr), sizeof(RoCEAttr), 0, sizeof(RoCEAttr));
            (void)memset_s(&(channelDesc->ubAttr), sizeof(UbAttr), 0, sizeof(UbAttr));
        }
    }
    return;
}

HcclResult HcclChannelAcquire(HcclComm comm, CommEngine engine, const HcclChannelDesc *channelDescList,
    uint32_t listNum, ChannelHandle *channelList) __attribute__((weak));

int32_t HcommAclrtNotifyRecordOnThread(ThreadHandle thread, uint64_t dstNotifyId) __attribute__((weak));
int32_t HcommAclrtNotifyWaitOnThread(ThreadHandle thread, uint64_t notifyId, uint32_t timeOut) __attribute__((weak));

int32_t HcommChannelNotifyRecordOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t remoteNotifyIdx) __attribute__((weak));
int32_t HcommChannelNotifyWaitOnThread(ThreadHandle thread, ChannelHandle channel, uint32_t localNotifyIdx, uint32_t timeout) __attribute__((weak));

int32_t HcommThreadNotifyRecordOnThread(ThreadHandle thread, ThreadHandle dstThread, uint32_t dstNotifyIdx) __attribute__((weak));
int32_t HcommThreadNotifyWaitOnThread(ThreadHandle thread, uint32_t notifyIdx, uint32_t timeout) __attribute__((weak));
HcclResult HcclThreadAcquireWithStream(HcclComm comm, CommEngine engine,
    aclrtStream stream, uint32_t notifyNum, ThreadHandle *thread) __attribute__((weak));
#ifdef __cplusplus
}
#endif  // __cplusplus
#endif

// 解决Hcomm仓合入问题
#ifndef HCCL_CTX_API
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

HcclResult HcclEngineCtxCreate(HcclComm comm, const char *ctxTag, CommEngine engine, uint64_t size, void **ctx) __attribute__((weak));
HcclResult HcclEngineCtxGet(HcclComm comm, const char *ctxTag, CommEngine engine, void **ctx, uint64_t *size) __attribute__((weak));
HcclResult HcclEngineCtxCopy(HcclComm comm, CommEngine engine, const char *ctxTag, const void *srcCtx,
    uint64_t size, uint64_t dstCtxOffset) __attribute__((weak));

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_CTX_API

namespace ops_hccl {

constexpr u32 COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH; // 算子相关的topo表达
constexpr uint32_t OP_ALG_LENGTH = 128; // 存放算法 + host/device标记
constexpr uint32_t ALG_TAG_LENGTH = TAG_LENGTH + OP_ALG_LENGTH;
constexpr uint32_t AICPU_CONTROL_NOTIFY_NUM = 2;

// 是否再拆分一个comm头文件
constexpr u32 LOCAL_NOTIFY_IDX_ZERO = 0;
constexpr u32 NOTIFY_IDX_ACK = 0;
constexpr u32 NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr u32 NOTIFY_IDX_FIN_ACK = 2;
constexpr u32 CUSTOM_TIMEOUT = 1800;

enum class TopoType {
    TOPO_TYPE_COMMON = 0,           // 普通拓扑类型 ，default单层拓扑使用
    TOPO_TYPE_8P_RING = 1,          // 特殊场景, 服务器内8 rank组成一个ring，4个逻辑环
    TOPO_TYPE_4P_MESH = 2,          // 特殊场景, 服务器内4 rank组成MESH
    TOPO_TYPE_2P_MESH = 3,          // 特殊场景, 服务器内2 rank组成MESH。仅用于测试和自验证
    TOPO_TYPE_1P_MESH = 4,          // 特殊场景, 服务器内1 rank组成MESH。仅用于测试和自验证
    TOPO_TYPE_4P_RING = 5,          // 特殊场景，服务器内4 rank组成ring
    TOPO_TYPE_NP_SINGLE_RING = 6,   // 特殊场景, 服务器内n rank组成单 ring。目前仅用于标卡
    TOPO_TYPE_8P_MESH = 7,          // 特殊场景, 服务器内8 rank通过RDMA组成MESH
    TOPO_TYPE_NP_MESH = 8,          // 特殊场景, 服务器内3~8p rank组成MESH
    TOPO_TYPE_NP_DOUBLE_RING = 9,   // 特殊场景, 910_93场景
    TOPO_TYPE_HETEROG = 10,
    TOPO_TYPE_ES_MESH = 11,
    TOPO_TYPE_RESERVED
};

// 这个应该是公共的
struct TopoInfo { // 通信域拓扑ctx
    u32 userRank; // rankId
    u32 userRankSize; // 通信域rankSize
    u32 devicePhyId; // 在服务器上的物理槽位号
    u32 serverIdx = INVALID_UINT; // Server在ranktable中的自然顺序
    u32 superPodIdx = INVALID_UINT; // SuperPod在ranktable中的自然顺序
    DevType deviceType = DevType::DEV_TYPE_COUNT; // 硬件类型
    u32 deviceNumPerModule = 0; // A2 每个module的卡数
    u32 serverNumPerSuperPod = 0; // 每个超节点的服务器个数
    u32 serverNum = 0; // 服务器数量
    u32 moduleNum = 0; // A2 A+X场景moudleNum可能与serverNum不符
    u32 superPodNum = 0; // 超节点数量
    u32 moduleIdx = INVALID_UINT; // moduleId
    bool isDiffDeviceModule = false; // A2 A+X
    bool multiModuleDiffDeviceNumMode = false;   // Server间卡数不一致
    bool multiSuperPodDiffServerNumMode = false; // 超节点间Server数不一致
    bool isHCCSSWNumEqualToTwiceSIONum = false; // A3 Server内链路属性
};

// A5用了cntNotify
struct AlgResourceRequest {
    u32 notifyNumOnMainThread = 0;
    u32 slaveThreadNum = 0;
    u32 notifyNumPerThread = 0;
    std::vector<std::vector<HcclChannelDesc>> channels;
};

constexpr u32 HCCL_LOGIC_TOPO_LEVEL_NUM = 4; // HCCL逻辑拓扑层级最多4级

struct SubCommInfo {
    u32 localRank = 0;
    u32 localRankSize = 1;
};

struct AlgHierarchyInfo {
    u32 levels = 1;
    SubCommInfo infos[HCCL_LOGIC_TOPO_LEVEL_NUM];
};

struct ChannelInfo {
    bool isValid = false;
    u32 remoteRank = INVALID_VALUE_RANKID;
    CommProtocol protocol;
    u32 notifyNum;
    ChannelHandle handle;
    HcclMem remoteInput;
    HcclMem remoteOutput;
};

// 算法ctx，key为通信域id+算法名，提前在device上
// 头部需补充版本号和长度信息
struct AlgResourceCtx {
    AlgType algType; // 环境变量设置的算法类型
    AlgHierarchyInfo algHierarchyInfo; // 算法分层信息
    HcclMem cclInputMem; // 跨Rank缓存Buffer
    HcclMem cclOutputMem; // 跨Rank缓存Buffer
    u32 notifyNumOnMainThread; // 主流上的notify数量
    u32 slaveThreadNum; // 需要的thread数量
    u32 notifyNumPerThread; // 每个thread需要的notify数量
    uint32_t notifyIds[AICPU_CONTROL_NOTIFY_NUM]; // aicpu 模式下控制notify
    TopoInfo topoInfo; // 提取的拓扑信息
    // 下面是变长数据区
    // ThreadHandle* threads; // threadNum个，主流和从流的thread句柄
    // ChannelInfo* channels; // 通信链路，数量可根据algHierarchyInfo字段进行推算
};

struct OpParam { // 不申请ctx，每个算子单独下发
    char tag[TAG_LENGTH];
    char algTag[ALG_TAG_LENGTH];
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    aclrtStream stream;
    void* inputPtr = nullptr;
    u64 inputSize = 0;
    void* outputPtr = nullptr;
    u64 outputSize = 0;
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    u32 root = INVALID_VALUE_RANKID;
    CommEngine engine = CommEngine::COMM_ENGINE_RESERVED;
    union {
        struct {
            u64 count;
            HcclDataType dataType;
            u64 strideCount;
        } DataDes = {0, HCCL_DATA_TYPE_RESERVED, 0};
    };
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    bool isZeroCopy = false;
    char algName[OP_ALG_LENGTH];
    AlgResourceCtx* resCtx = nullptr; // 资源长度变长，只能放在最后一个位置
};

struct AlgDesc {
    bool isZeroCopy = false;
    bool isAivMode = false;
    // executor所支持的各级算法，当vector为空时表示不校验，若外部传入的algType不支持，重定向为vector第一个元素
    // 由于默认算法要从列表里的第一个取，因此使用顺序确定的vector而非set
    std::vector<AlgTypeLevel0> level0SupportedAlgos;
    std::vector<AlgTypeLevel1> level1SupportedAlgos;
    std::vector<AlgTypeLevel2> level2SupportedAlgos;
};

struct Slice {
    u64 offset{0}; // Slice相对于input/output的偏移字节数，gather类操作取output，scatter类操作取input
    u64 size{0};    // Slice的数据大小，单位：字节
};
}

#ifndef HCOMM_PRIMITIVES_H_MODIFIED
typedef enum {
    HCOMM_LAUNCH_MODE_RESERVED = -1, ///< 保留的下发模式
    HCOMM_LAUNCH_MODE_EAGER = 0,     ///< 直接下发模式（实时执行）
    HCOMM_LAUNCH_MODE_BATCH          ///< 批量下发模式（延迟合并执行）
} HcommLaunchMode;
typedef enum {
    HCOMM_REDUCE_SUM = 0,    /**< sum */
    HCOMM_REDUCE_PROD = 1,   /**< prod */
    HCOMM_REDUCE_MAX = 2,    /**< max */
    HCOMM_REDUCE_MIN = 3,    /**< min */
    HCOMM_REDUCE_RESERVED = 255 /**< reserved */
} HcommReduceOp;

typedef enum {
    HCOMM_DATA_TYPE_INT8 = 0,    /**< int8 */
    HCOMM_DATA_TYPE_INT16 = 1,   /**< int16 */
    HCOMM_DATA_TYPE_INT32 = 2,   /**< int32 */
    HCOMM_DATA_TYPE_FP16 = 3,    /**< fp16 */
    HCOMM_DATA_TYPE_FP32 = 4,    /**< fp32 */
    HCOMM_DATA_TYPE_INT64 = 5,    /**< int64 */
    HCOMM_DATA_TYPE_UINT64 = 6,    /**< uint64 */
    HCOMM_DATA_TYPE_UINT8 = 7,    /**< uint8 */
    HCOMM_DATA_TYPE_UINT16 = 8,   /**< uint16 */
    HCOMM_DATA_TYPE_UINT32 = 9,   /**< uint32 */
    HCOMM_DATA_TYPE_FP64 = 10,    /**< fp64 */
    HCOMM_DATA_TYPE_BFP16 = 11,    /**< bfp16 */
    HCOMM_DATA_TYPE_INT128 = 12,   /**< int128 */
#ifndef OPEN_BUILD_PROJECT
    HCOMM_DATA_TYPE_HIF8 = 14,     /**< hif8 */
    HCOMM_DATA_TYPE_FP8E4M3 = 15,  /**< fp8e4m3 */
    HCOMM_DATA_TYPE_FP8E5M2 = 16,  /**< fp8e5m2 */
    HCOMM_DATA_TYPE_FP8E8M0 = 17,  /**< fp8e8m0 */
#endif
    HCOMM_DATA_TYPE_RESERVED = 255 /**< reserved */
} HcommDataType;
#endif
#endif
