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
#include <memory>
#include <functional>
#include <functional>
#include <memory>
#include <hccl/hccl_comm.h>
#include "hccl_common.h"
#include "hccl_types.h"
#include "alg_type.h"
#include "hccl_res_dl.h"
#include "hcomm_primitives_dl.h"
#include "hccl_rank_graph_dl.h"
#include "hccl_host_comm_dl.h"
#include "binary_stream.h"
#include "hccl_ccu_res_dl.h"
#include "ccu_types_dl.h"

namespace ops_hccl {

constexpr uint64_t UB_MAX_DATA_SIZE = 256*1024*1024; // Byte, UB协议一次传输的最大size

constexpr u32 MAX_NUM_BLOCKS = 56; // 56-72

constexpr uint32_t DATATYPE_SIZE_TABLE[HCCL_DATA_TYPE_RESERVED] = {sizeof(int8_t), sizeof(int16_t), sizeof(int32_t),
    2, sizeof(float), sizeof(int64_t), sizeof(uint64_t), sizeof(uint8_t), sizeof(uint16_t), sizeof(uint32_t),
    8, 2, 16, 2, 1, 1, 1, 1};

constexpr u32 COMM_INDENTIFIER_MAX_LENGTH = 128;
constexpr uint32_t OP_NAME_LENGTH = 32;
constexpr uint32_t TAG_LENGTH = OP_NAME_LENGTH + COMM_INDENTIFIER_MAX_LENGTH; // 算子相关的topo表达
constexpr uint32_t OP_ALG_LENGTH = 128; // 存放算法 + host/device标记
constexpr uint32_t ALG_TAG_LENGTH = TAG_LENGTH + OP_ALG_LENGTH;
constexpr uint32_t MAX_TAG_LENGTH = 255;
constexpr uint32_t AICPU_CONTROL_NOTIFY_NUM = 2;
constexpr uint32_t MAX_MEM_TAG_LENGTH = OP_ALG_LENGTH + 32;
constexpr uint32_t RES_PACK_TAG_LENGTH = 255;
constexpr uint32_t MAX_TEMP_NUM_IN_ALGO = 8; // 单个算法中最大template数量

// 是否再拆分一个comm头文件
constexpr u32 LOCAL_NOTIFY_IDX_ZERO = 0;
constexpr u32 NOTIFY_IDX_ACK = 0;
constexpr u32 NOTIFY_IDX_DATA_SIGNAL = 1;
constexpr u32 NOTIFY_IDX_FIN_ACK = 2;
constexpr u32 CUSTOM_TIMEOUT = 1836;
constexpr u32 TIME_S_TO_US = 1000000;
constexpr u32 MAX_LENGTH = 128;
constexpr u32 ALG_MAX_LENGTH = 128;

// alltoallv需要
constexpr u64 ALL_TO_ALL_V_VECTOR_NUM = 4;
constexpr u64 REDUCE_SCATTER_V_VECTOR_NUM = 2;
constexpr u64 ALL_GATHER_V_VECTOR_NUM = 2;

constexpr uint64_t GE_PARALLEL = 36;

constexpr uint64_t AICPU_ALIGN_SIZE = 4096;
// Z axis detour 需要
constexpr u32 MESH_CHANNELS_NUM = 1;

constexpr uint64_t CCU_MAX_RANK_SIZE = 16;

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

// 通信域粒度加速模式
enum class OpExecuteConfig {
    DEFAULT = 0,
    HOSTCPU_TS = 1,
    AICPU_TS = 2,
    AIV = 3,
    AIV_ONLY = 4,
    CCU_MS = 5,
    CCU_SCHED = 6,
    AICPU = 7,
    HOSTCPU = 8,
    CCU_FAIL
};

enum class OpMode {
    OFFLOAD = 0,
    OPBASE = 1
};

enum class Level0Shape {
    CLOS    = 0,
    MESH_1D = 1,
    MESH_1D_CLOS = 2,
};

enum class Level0MeshType {
    NOT_MESH = 0,
    SINGLE_DIE = 1,
    TWO_DIE_REGULAR = 2,
    TWO_DIE_NOT_REGULAR = 3,
};

struct NetLayerDetails {
    u32 netLayerNum;
    std::vector<u32> netLayers;
    std::vector<u32> netInstNumOfLayer;
    std::vector<std::vector<u32>> instSizeListOfLayer;
    std::vector<u32> localNetInsSizeOfLayer;
};
struct TopoInstDetails {
    u32 topoInstNum;
    std::vector<u32> sizeOfTopo;
    std::vector<CommTopo> typeOfTopo;
    std::vector<std::vector<u32>> ranksInTopo;
    std::map<CommTopo, std::vector<u32>> rankNumForTopoType;
};

struct TopoInfo {
    u32 userRank; // rankId
    u32 userRankSize; // 通信域rankSize
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
    ThreadHandle mainThread;    // 主流对应threadHandle
    u32 notifyNumOnMainThread = 0;  // mainThread上创建的notify数量
};

// 这个应该是公共的
struct TopoInfoWithNetLayerDetails : public TopoInfo { // 通信域拓扑ctx
    u32 topoLevelNums = 0;
    Level0Shape level0Topo;
    bool Level0Nhr{false};
    bool Level1Nhr{false};
    bool Level1Hd{false};
    bool is2DieFullMesh{false};
    bool level0PcieMix{false};
    bool level0BigClosRange{false};
    u32 topoInstDetailsOfLayerSize = 0;
    Level0MeshType level0MeshType;
    NetLayerDetails netLayerDetails;
    std::vector<TopoInstDetails> topoInstDetailsOfLayer;

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << userRank;
        binaryStream << userRankSize;
        binaryStream << serverIdx;
        binaryStream << superPodIdx;
        binaryStream << deviceType;
        binaryStream << deviceNumPerModule;
        binaryStream << serverNumPerSuperPod;
        binaryStream << serverNum;
        binaryStream << moduleNum;
        binaryStream << superPodNum;
        binaryStream << moduleIdx;
        binaryStream << isDiffDeviceModule;
        binaryStream << multiModuleDiffDeviceNumMode;
        binaryStream << multiSuperPodDiffServerNumMode;
        binaryStream << isHCCSSWNumEqualToTwiceSIONum;
        binaryStream << mainThread;
        binaryStream << notifyNumOnMainThread;
        binaryStream << topoLevelNums;
        binaryStream << level0Topo;
        binaryStream << Level0Nhr;
        binaryStream << Level1Nhr;
        binaryStream << Level1Hd;
        binaryStream << is2DieFullMesh;
        binaryStream << level0PcieMix;
        binaryStream << level0BigClosRange;
        binaryStream << topoInstDetailsOfLayerSize;
        binaryStream << level0MeshType;
        binaryStream << netLayerDetails.netLayerNum;
        binaryStream << netLayerDetails.netLayers;
        binaryStream << netLayerDetails.netInstNumOfLayer;
        binaryStream << netLayerDetails.instSizeListOfLayer;
        binaryStream << netLayerDetails.localNetInsSizeOfLayer;
        for (uint32_t idx = 0; idx < topoInstDetailsOfLayerSize; idx++) {
            binaryStream << topoInstDetailsOfLayer[idx].topoInstNum;
            binaryStream << topoInstDetailsOfLayer[idx].sizeOfTopo;
            binaryStream << topoInstDetailsOfLayer[idx].typeOfTopo;
            binaryStream << topoInstDetailsOfLayer[idx].ranksInTopo;
            binaryStream << topoInstDetailsOfLayer[idx].rankNumForTopoType;
        }
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }
 
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> userRank;
        binaryStream >> userRankSize;
        binaryStream >> serverIdx;
        binaryStream >> superPodIdx;
        binaryStream >> deviceType;
        binaryStream >> deviceNumPerModule;
        binaryStream >> serverNumPerSuperPod;
        binaryStream >> serverNum;
        binaryStream >> moduleNum;
        binaryStream >> superPodNum;
        binaryStream >> moduleIdx;
        binaryStream >> isDiffDeviceModule;
        binaryStream >> multiModuleDiffDeviceNumMode;
        binaryStream >> multiSuperPodDiffServerNumMode;
        binaryStream >> isHCCSSWNumEqualToTwiceSIONum;
        binaryStream >> mainThread;
        binaryStream >> notifyNumOnMainThread;
        binaryStream >> topoLevelNums;
        binaryStream >> level0Topo;
        binaryStream >> Level0Nhr;
        binaryStream >> Level1Nhr;
        binaryStream >> Level1Hd;
        binaryStream >> is2DieFullMesh;
        binaryStream >> level0PcieMix;
        binaryStream >> level0BigClosRange;
        binaryStream >> topoInstDetailsOfLayerSize;
        binaryStream >> level0MeshType;
        binaryStream >> netLayerDetails.netLayerNum;
        binaryStream >> netLayerDetails.netLayers;
        binaryStream >> netLayerDetails.netInstNumOfLayer;
        binaryStream >> netLayerDetails.instSizeListOfLayer;
        binaryStream >> netLayerDetails.localNetInsSizeOfLayer;
        topoInstDetailsOfLayer.resize(topoInstDetailsOfLayerSize);
        for (uint32_t idx = 0; idx < topoInstDetailsOfLayerSize; idx++) {
            binaryStream >> topoInstDetailsOfLayer[idx].topoInstNum;
            binaryStream >> topoInstDetailsOfLayer[idx].sizeOfTopo;
            binaryStream >> topoInstDetailsOfLayer[idx].typeOfTopo;
            binaryStream >> topoInstDetailsOfLayer[idx].ranksInTopo;
            binaryStream >> topoInstDetailsOfLayer[idx].rankNumForTopoType;
        }
    }
};

struct CcuKernelArgBase {
    // std::vector<ChannelHandle> channels;
    ChannelHandle channels[CCU_MAX_RANK_SIZE];
    uint32_t      channelCount;
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel资源组序号，group号不同时，资源复用
    u32 resGroup = 0;
    // kernel名 string？
    char kernelFuncName[64];
    // kernel函数
    void* kernelFunc;
    // KernelArg实例指针
    void *kernelArg;
    // kernel所需channel
    std::vector<HcclChannelDesc> channels;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template<typename T>
    void setKernelArg(std::shared_ptr<T> arg) {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void*>(arg.get());
    }
};

// 算法taskArg入参最大个数，用于快速下发缓存
#define CCU_MAX_TASK_ARG_NUM 48

struct CcuKernelSubmitInfo {
    CcuKernelHandle kernelHandle;
    uint64_t cachedArgs[CCU_MAX_TASK_ARG_NUM];
};

// ccu快速下发上下文
struct CcuFastLaunchCtx {
    char algName[OP_ALG_LENGTH];
    u32 notifyNumOnMainThread = 0;
    u32 threadNum;
    u32 ccuKernelNum[MAX_TEMP_NUM_IN_ALGO];  // 每次调用template的KernelRun下发的kernel数量
    // 紧接ThreadHandle数组
    // 紧接CcuKernelSubmitInfo数组

    ThreadHandle *GetThreadHandlePtr() const
    {
        size_t offset = offsetof(CcuFastLaunchCtx, ccuKernelNum)
                        + sizeof(u32) * MAX_TEMP_NUM_IN_ALGO;
        return reinterpret_cast<ThreadHandle*>(
                    reinterpret_cast<char*>(const_cast<CcuFastLaunchCtx*>(this)) + offset
                );
    }
    CcuKernelSubmitInfo *GetCcuKernelSubmitInfoPtr() const
    {
        size_t offset = offsetof(CcuFastLaunchCtx, ccuKernelNum)
                        + sizeof(u32) * MAX_TEMP_NUM_IN_ALGO 
                        + sizeof(ThreadHandle) * threadNum;
        return reinterpret_cast<CcuKernelSubmitInfo*>(
                    reinterpret_cast<char*>(const_cast<CcuFastLaunchCtx*>(this)) + offset
                );
    }

    static u64 GetCtxSize(u32 threadNum, u32 totalCcuKernelNum)
    {
        return sizeof(CcuFastLaunchCtx) 
               + sizeof(ThreadHandle) * threadNum 
               + sizeof(CcuKernelSubmitInfo) * totalCcuKernelNum;
    }
};

// A5用了cntNotify
struct AlgResourceRequest {
    u32 notifyNumOnMainThread = 0;
    u32 slaveThreadNum = 0;
    std::vector<u32> notifyNumPerThread;
    std::vector<std::vector<HcclChannelDesc>> channels;
    std::vector<CcuKernelInfo> ccuKernelInfos;
    std::vector<u32> ccuKernelNum;
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
    CommProtocol protocol = CommProtocol::COMM_PROTOCOL_RESERVED;
    EndpointLocType locationType = EndpointLocType::ENDPOINT_LOC_TYPE_RESERVED;
    u32 notifyNum = 0;
    u32 portGroupSize = 1; // A5用的, 端口组大小，用于数据分片比例计算
    ChannelHandle handle = 0;
    HcclMem remoteCclMem; // A5用的
    HcclMem remoteInputGraphMode;   // A5用的, 图模式下远端sendBuf地址
    HcclMem remoteOutputGraphMode;  // A5用的，图模式下远端recvBuf地址
    HcclMem remoteInput;  // A3用的，cclIn
    HcclMem remoteOutput; // A3用的, cclOut
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
    ThreadHandle opThread;  // 算子stream申请的thread，用于host、device同步
    uint32_t notifyIds[AICPU_CONTROL_NOTIFY_NUM]; // aicpu 模式下控制notify
    TopoInfo topoInfo; // 提取的拓扑信息
    void* aivCommInfoPtr = nullptr;
    // 下面是变长数据区
    // ThreadHandle* threads; // threadNum个，主流和从流的thread句柄
    // ChannelInfo* channels; // 通信链路，数量可根据algHierarchyInfo字段进行推算
};

// 如果能够序列化那么就是下面的结构体
struct AlgHierarchyInfoForAllLevel {
    std::vector<std::vector<std::vector<u32>>> infos; // 第一维表示有多少level，第二维是每个level的rankID
};
// 如果能够序列化那么就是下面的结构体
// 先序列化，把东西考到device，然后把指针存到OpParam，在device侧反序列该指针执行的内存
struct AlgResourceCtxSerializable {
    AlgType algType; // 环境变量设置的算法类型
    AlgHierarchyInfoForAllLevel algHierarchyInfo; // 算法分层信息
    HcclMem cclMem; // 跨Rank缓存Buffer
    u32 notifyNumOnMainThread; // 主流上的notify数量
    u32 slaveThreadNum; // 需要的thread数量
    u32 waitTimeout = 0; // Device侧notify wait默认超时时间
    u32 fullTimeout = 0; // Device侧队列满/资源申请超时时间
    std::vector<u32> notifyNumPerThread; // 每个thread需要的notify数量
    void* aivCommInfoPtr = nullptr;
    std::vector<ThreadHandle> threads;
    ThreadHandle unfoldThread = 0; // 展开流thread
    std::vector<std::vector<ChannelInfo>> channels;
    bool isHcommBatchTransferOnThreadSupported = false;
    void* commInfoPtr = nullptr;
    // hostdpu
    void *npu2DpuShmemPtr = nullptr;
    void *dpu2NpuShmemPtr = nullptr;
    // ccu的
    std::vector<u32> ccuKernelNum;
    std::vector<CcuKernelHandle> ccuKernels;
    u32 topoInfoSeqSize = 0;
    TopoInfoWithNetLayerDetails topoInfo; // 提取的拓扑信息

    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;

        binaryStream << algType;
        binaryStream << algHierarchyInfo.infos;
        binaryStream << cclMem;
        binaryStream << notifyNumOnMainThread;
        binaryStream << slaveThreadNum;
        binaryStream << waitTimeout;
        binaryStream << fullTimeout;
        binaryStream << notifyNumPerThread;
        binaryStream << commInfoPtr;
        binaryStream << threads;
        binaryStream << unfoldThread;
        binaryStream << channels;
        binaryStream << isHcommBatchTransferOnThreadSupported;

        binaryStream << npu2DpuShmemPtr;
        binaryStream << dpu2NpuShmemPtr;

        binaryStream << ccuKernelNum;
        binaryStream << ccuKernels;
        std::vector<char> seq = topoInfo.Serialize();
        topoInfoSeqSize = seq.size();
        binaryStream << topoInfoSeqSize;
        std::vector<char> result;
        binaryStream.Dump(result);
        result.insert(result.end(), seq.begin(), seq.end());

        return result;
    }

    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);

        binaryStream >> algType;
        binaryStream >> algHierarchyInfo.infos;
        binaryStream >> cclMem;
        binaryStream >> notifyNumOnMainThread;
        binaryStream >> slaveThreadNum;
        binaryStream >> waitTimeout;
        binaryStream >> fullTimeout;
        binaryStream >> notifyNumPerThread;
        binaryStream >> commInfoPtr;
        binaryStream >> threads;
        binaryStream >> unfoldThread;
        binaryStream >> channels;
        binaryStream >> isHcommBatchTransferOnThreadSupported;

        binaryStream >> npu2DpuShmemPtr;
        binaryStream >> dpu2NpuShmemPtr;

        binaryStream >> ccuKernelNum;
        binaryStream >> ccuKernels;
        binaryStream >> topoInfoSeqSize;
        size_t startPos = data.size() - topoInfoSeqSize;
        std::vector<char> tailData(data.begin() + startPos, data.end());
        TopoInfoWithNetLayerDetails topoTemp;
        topoTemp.DeSerialize(tailData);
        topoInfo = std::move(topoTemp);
    }
};

struct DevAicpuOpConfig {
    u32 execTimeout = 0;
    double multipleDimensionSplitRatio = 0.8;
    // 如要新增配置类字段，在此处添加
};

struct OpParam { // 不申请ctx，每个算子单独下发
    void* hcclComm;
    char tag[TAG_LENGTH] = ""; // 保存topoInfo的key值
    char algTag[ALG_TAG_LENGTH] = ""; // 保存资源的key值，和算法绑定
    char fastLaunchTag[ALG_TAG_LENGTH] = ""; // 快速下发的key值
    char fallbackTag[ALG_MAX_LENGTH] = "";
    char commName[COMM_INDENTIFIER_MAX_LENGTH] = "";
    char commModeTag[TAG_LENGTH] = ""; // 保存与执行模式相关的资源信息的key值，当前aiv使用
    aclrtStream stream;
    void* inputPtr = nullptr;
    u64 inputSize = 0;
    void* outputPtr = nullptr;
    u64 outputSize = 0;
    HcclMem hcclBuff;   // 当前仅快速下发时使用此处的地址
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    u32 root = INVALID_VALUE_RANKID;
    u32 userRank = INVALID_VALUE_RANKID;
    u32 sendRecvRemoteRank = INVALID_VALUE_RANKID;
    OpMode opMode;
    bool   enableDetour{false};
    bool   isMc2{false};
    bool   cacheValid{false};
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CommEngine engine = CommEngine::COMM_ENGINE_RESERVED;
    AlgType algType;
    char algTypeStr[ALG_MAX_LENGTH] = "";
    union {
        struct {
            u64 count;
            HcclDataType dataType;
            HcclDataType outputType;
            u64 strideCount;
        } DataDes = {0, HCCL_DATA_TYPE_RESERVED, HCCL_DATA_TYPE_RESERVED, 0};
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            u64 sendCount;
            u64 recvCount;
        } all2AllDataDes;
        struct {
            void* counts;
            void* displs;
            HcclDataType dataType;
        } vDataDes;
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            void* sendCounts;
            void* recvCounts;
            void* sdispls;
            void* rdispls; // 指向变长区指针
        } all2AllVDataDes;
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            void* sendCountMatrix;
        } all2AllVCDataDes;
        struct {
            HcclSendRecvItem* sendRecvItemsPtr;
            u32 itemNum;
        } batchSendRecvDataDes;
    };
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    bool isZeroCopy = false;
    char algName[OP_ALG_LENGTH] = "";
    HcclOpExpansionMode commOpExpansionMode = HcclOpExpansionMode::HCCL_OP_EXPANSION_MODE_INVALID;
    OpExecuteConfig opExecuteConfig;
    u32 numBlocksLimit = 0;
    bool isAivClearEnable = false;
    u64 ctxSize = 0;
    void* resCtx = nullptr;
    ThreadHandle opThread = 0;
    u32 aicpuRecordCpuIdx = 0; // aicpu record host的notifyIdx
    u32 dataCount = 0; // 算子上报dfx的数据量
    DevAicpuOpConfig opConfig; // 收编算子配置类变量
    u64 varMemSize{0};
    u8 varData[0];
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

struct HcomProInfo {
    uint8_t dataType;
    uint8_t cmdType;
    uint64_t dataCount;
    uint32_t rankSize;
    uint32_t userRank;
    uint32_t blockDim = 0;
    uint64_t beginTime;
    uint32_t root;
    uint32_t slaveThreadNum;
    uint64_t commNameLen;
    uint64_t algTypeLen;
    char tag[MAX_LENGTH];
    char commName[MAX_LENGTH];
    char algType[MAX_LENGTH];
    bool isCapture = false;
    bool isAiv = false;
    uint8_t reserved[MAX_LENGTH];
};

// 图模式相关定义
// 图模式编译阶段资源计算入参
struct OpParamGraphMode {
    char opType[64]; // 算子类型
    u64 dataCount;
    u32 rankSize;
    u64 hcclBufferSize;
    // Aiv参数
    s64 comm;
    char group[MAX_LENGTH];
    u64 count = 0;
    void* counts = nullptr;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp op = HcclReduceOp::HCCL_REDUCE_RESERVED;
    HcclCMDType opTypeAiv = HcclCMDType::HCCL_CMD_INVALID;
    u32 aivCoreLimit = 0;
    bool ifAiv = false;
};

// 图模式编译阶段申请资源
struct ResResponseGraphMode {
    u64 opMemSize = 0;  // 额外申请的scratch数量（不包括cclBuff）
    u32 streamNum = 0;  // 除用户流以外，额外申请的流（不包括算子device展开申请的流）
    u32 taskNum = 0;    // task数量，一般为前同步 + kernel + 后同步
    u32 aivCoreNum = 0;
};

// 图模式执行阶段传入的资源
struct ResPackGraphMode {
    char tag[RES_PACK_TAG_LENGTH];
    std::vector<aclrtStream> streams;
    void* scratchMemAddr;
    u64 scratchMemSize;
};

// 图模式内存注册信息
struct MemRegInfo {
    char inputBuffTag[MAX_MEM_TAG_LENGTH];    // 输入缓冲区标签
    char outputBuffTag[MAX_MEM_TAG_LENGTH];   // 输出缓冲区标签
    std::vector<HcclMemHandle> memHandles;    // 内存句柄列表
};

// AIV模式参数存储结构
struct AivParamStorage {
    u32 aivCoreLimit = 0;
    bool aivClearEnable = false;
};

// 算子参数一致性校验信息
struct OpExchangeInfo {
    uint64_t cclBufferSize{0};
    u32 root = INVALID_VALUE_RANKID;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    OpExecuteConfig opExecuteConfig = OpExecuteConfig::DEFAULT;
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    u64 count{0};
    u32 aivCoreLimit = MAX_NUM_BLOCKS;
    char group[MAX_LENGTH] = {0};
    char tag[TAG_LENGTH] = {0};
};

} 
#endif
