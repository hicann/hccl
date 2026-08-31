/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * ============================================================================================
 * HCCL-ALGO-Plugin 自定义算法开发 SDK
 *
 * 使用方法（以AllReduce算子为例）：
 *
 *   1) 编写 libhccl_plugin_allreduce_selector.so 的源码，#include 本头文件：
 *
 *        #include "hccl_algo_plugin_sdk.h"
 *
 *        REGISTER_HCCL_ALGO("MyRingAllReduce", "/path/to/libMyRingAlgImpl.so",
 *                            "HcclAlgoPluginMyRingAllReduce");
 *
 *        extern "C" bool Select(const HcclAlgoPluginParam* param, char* algName, size_t algNameLen)
 *        {
 *            // 根据param中的拓扑/数据量信息决策，命中时:
 *            snprintf(algName, algNameLen, "MyRingAllReduce");
 *            return true;
 *        }
 *
 *      本.so编译时须设置 -fvisibility=hidden，并只显式导出 Select 与
 *      HcclAlgoPluginQueryEntries 两个符号（见文末"导出符号"说明），
 *      以保证不同算子的选择动态库注册表互不可见。
 *
 *   2) 编写 lib{Name}Impl.so 的源码，实现并导出执行函数（签名见下方"标准算法执行函数签名"）：
 *
 *        extern "C" HcclResult HcclAlgoPluginMyRingAllReduce(void* sendBuf, void* recvBuf,
 *            uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
 *        {
 *            // ...自定义Ring算法实现...
 *            return HCCL_SUCCESS;
 *        }
 *
 * 开发者无需手写 HcclAlgoPluginQueryEntries()、无需手写注册表管理逻辑，均由本头文件内联提供。
 * ============================================================================================
 */

#ifndef HCCL_ALGO_PLUGIN_SDK_H
#define HCCL_ALGO_PLUGIN_SDK_H

#include <cstring>
#include <string>
#include <vector>
#include <mutex>
#include <dlfcn.h>

#include "hccl_algo_plugin_common.h"

static inline bool HcclAlgoPluginCopyString(char* dst, size_t dstLen, const char* src)
{
    if (dst == nullptr || dstLen == 0) {
        return false;
    }
    if (src == nullptr) {
        dst[0] = '\0';
        return false;
    }

    size_t i = 0;
    while (i + 1 < dstLen && src[i] != '\0') {
        dst[i] = src[i];
        ++i;
    }

    dst[i] = '\0';
    return src[i] == '\0';
}

/* ------------------------------------------------------------------------------------------- */
/* 1) 内部注册表实现（自定义算法开发者无需关心，仅通过 REGISTER_HCCL_ALGO 宏间接使用）             */
/*    本类的所有符号须在编译.so时设为 hidden 可见性（-fvisibility=hidden 或版本脚本），          */
/*    避免多个 libhccl_plugin_{op}_selector.so 同时被进程 dlopen 时符号插入导致注册表被共享。   */
/* ------------------------------------------------------------------------------------------- */
class HcclAlgoPluginRegistry {
public:
    /* 单例仅在"本.so内部"可见（前提是编译时symbol设为hidden），不同.so各自拥有独立单例实例 */
    static HcclAlgoPluginRegistry& Instance()
    {
        static HcclAlgoPluginRegistry instance;
        return instance;
    }

    void Add(const char* algName, const char* soPath, const char* fnSymbol)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& e : entries_) {
            if (algName != nullptr && std::strcmp(e.algName, algName) == 0) {
                /* 重复算法名注册：保留首次注册的条目，忽略后续重复项，并不中止加载 */
                std::fprintf(
                    stderr,
                    "[HCCL-ALGO-Plugin SDK] duplicate algName [%s] registered in this selector.so, "
                    "keep the first registration, ignore this one.\n",
                    algName);
                return;
            }
        }
        HcclAlgoPluginAlgEntry entry{};
        HcclAlgoPluginAlgEntryInit(&entry);
        HcclAlgoPluginCopyString(entry.algName, sizeof(entry.algName), algName);
        HcclAlgoPluginCopyString(entry.soPath, sizeof(entry.soPath), soPath);
        HcclAlgoPluginCopyString(entry.fnSymbol, sizeof(entry.fnSymbol), fnSymbol);
        entries_.push_back(entry);
    }

    const HcclAlgoPluginAlgEntry* Data() const { return entries_.empty() ? nullptr : entries_.data(); }

    int Count() const { return static_cast<int>(entries_.size()); }

private:
    HcclAlgoPluginRegistry() = default;
    ~HcclAlgoPluginRegistry() = default;
    HcclAlgoPluginRegistry(const HcclAlgoPluginRegistry&) = delete;
    HcclAlgoPluginRegistry& operator=(const HcclAlgoPluginRegistry&) = delete;
    std::vector<HcclAlgoPluginAlgEntry> entries_;
    std::mutex mutex_;
};

/* 以全局静态对象的形式在dlopen时（构造函数）自动完成一条算法条目的注册 */
class HcclAlgoPluginAutoRegister {
public:
    HcclAlgoPluginAutoRegister(const char* algName, const char* implSoName, const char* fnSymbol)
    {
        std::string resolvedPath = ResolveImplSoPath(implSoName);
        HcclAlgoPluginRegistry::Instance().Add(algName, resolvedPath.c_str(), fnSymbol);
    }

private:
    static std::string ResolveImplSoPath(const char* implSoName)
    {
        if (implSoName == nullptr || implSoName[0] == '\0') {
            return std::string();
        }
        if (implSoName[0] == '/') {
            return std::string(implSoName);
        }
        Dl_info info{};
        if (dladdr(reinterpret_cast<void*>(&HcclAlgoPluginAutoRegister::ResolveImplSoPath), &info) != 0
            && info.dli_fname != nullptr) {
            std::string selfPath(info.dli_fname);
            size_t pos = selfPath.find_last_of('/');
            std::string selfDir = (pos == std::string::npos) ? "." : selfPath.substr(0, pos);
            return selfDir + "/" + implSoName;
        }
        return std::string(implSoName); // dladdr失败兜底
    }
};

/*
 * 自定义算法注册宏：自定义算法开发者以全局静态对象形式声明，声明后该算法信息在算法选择动态库被
 * dlopen时由构造函数自动写入本.so私有的注册表，无需手写集中式的注册函数。
 *
 * REGISTER_HCCL_ALGO(算法名, 集合通信算法实现动态库路径, 执行函数符号名)
 *
 * 注意：变量名拼接需要经过两层宏展开（HCCL_ALGO_PLUGIN_CONCAT_ / HCCL_ALGO_PLUGIN_CONCAT）
 * __LINE__ 才能被展开成具体行号，否则同一个文件里多次使用本宏（例如一个so里注册多个算法，
 * 见examples/06_custom_algo_plugin/AllReduce/示例）会因变量名都叫`_hccl_algo_reg___LINE__`
 * 而重复定义编译失败。
 */
#define HCCL_ALGO_PLUGIN_EXPORT extern "C" __attribute__((visibility("default")))
#define HCCL_ALGO_PLUGIN_CONCAT_(a, b) a##b
#define HCCL_ALGO_PLUGIN_CONCAT(a, b) HCCL_ALGO_PLUGIN_CONCAT_(a, b)
#define REGISTER_HCCL_ALGO(algName, soPath, fnSymbol) \
    static HcclAlgoPluginAutoRegister HCCL_ALGO_PLUGIN_CONCAT(_hccl_algo_reg_, __LINE__)(algName, soPath, fnSymbol)

/* ------------------------------------------------------------------------------------------- */
/* 2) HcclAlgoPluginQueryEntries()：由SDK头文件统一内联实现并自动导出，开发者无需手写            */
/*    PluginBroker通过dlsym解析并调用，需在dlclose本.so前完成算法条目拷贝。                     */
/* ------------------------------------------------------------------------------------------- */
extern "C" __attribute__((visibility("default"), weak, used)) const HcclAlgoPluginAlgEntry*
HcclAlgoPluginQueryEntries(int* count)
{
    if (count != nullptr) {
        *count = HcclAlgoPluginRegistry::Instance().Count();
    }
    return HcclAlgoPluginRegistry::Instance().Data();
}

/* ------------------------------------------------------------------------------------------- */
/* 3) 须由算法开发者实现并导出的算法选择入口                                                    */
/*    根据param中的通信参数和拓扑信息选择合适的算法名。返回true表示命中，algName填入选中算法名  */
/* ------------------------------------------------------------------------------------------- */
extern "C" bool Select(const HcclAlgoPluginParam* param, char* algName, size_t algNameLen);

/* ------------------------------------------------------------------------------------------- */
/* 4) 各算子的标准执行函数签名。                                                 */
/*    fnSymbol（符号名）由用户自定义并通过REGISTER_HCCL_ALGO宏的第三个参数告知PluginBroker，     */
/*    但参数列表与返回类型必须与下列对应算子的标准签名严格一致，PluginBroker按标准签名dlsym调用。*/
/*    以下类型仅作为文档/类型检查用途，实现时请直接按 extern "C" 导出同签名的函数，函数名任意。  */
/* ------------------------------------------------------------------------------------------- */

/* AllReduce 标准执行函数签名 */
typedef HcclResult (*HcclAlgoPluginAllReduceFn)(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream);

/* AllGather 标准执行函数签名 */
typedef HcclResult (*HcclAlgoPluginAllGatherFn)(
    void* sendBuf, void* recvBuf, uint64_t sendCount, HcclDataType dataType, HcclComm comm, aclrtStream stream);

/* Broadcast 标准执行函数签名 */
typedef HcclResult (*HcclAlgoPluginBroadcastFn)(
    void* buf, uint64_t count, HcclDataType dataType, uint32_t root, HcclComm comm, aclrtStream stream);

/* Reduce 标准执行函数签名 */
typedef HcclResult (*HcclAlgoPluginReduceFn)(
    void* sendBuf, void* recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint32_t root, HcclComm comm,
    aclrtStream stream);

/* ReduceScatter 标准执行函数签名 */
typedef HcclResult (*HcclAlgoPluginReduceScatterFn)(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm,
    aclrtStream stream);

/* AlltoAll类算子标准执行函数签名 */
typedef HcclResult (*HcclAlgoPluginAllToAllFn)(
    const void* sendBuf, uint64_t sendCount, HcclDataType sendType, const void* recvBuf, uint64_t recvCount,
    HcclDataType recvType, HcclComm comm, aclrtStream stream);

/* Scatter 标准执行函数签名 */
typedef HcclResult (*HcclAlgoPluginScatterFn)(
    void* sendBuf, void* recvBuf, uint64_t recvCount, HcclDataType dataType, uint32_t root, HcclComm comm,
    aclrtStream stream);

/* Send/Recv 标准执行函数签名 */
typedef HcclResult (*HcclAlgoPluginSendFn)(
    void* sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank, HcclComm comm, aclrtStream stream);
typedef HcclResult (*HcclAlgoPluginRecvFn)(
    void* recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank, HcclComm comm, aclrtStream stream);

#endif /* HCCL_ALGO_PLUGIN_SDK_H */
