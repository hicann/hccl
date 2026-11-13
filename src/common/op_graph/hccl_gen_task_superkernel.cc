/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_gen_task_superkernel.cc
 * \brief
 */

#include "hccl_gen_task_superkernel.h"
#include <nlohmann/json.hpp>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <exception>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "exe_graph/runtime/task_def_builder.h"
#include "graph/ge_version_check.h"
#include "op_util.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"

using namespace ge;

// checker.h 为非对外开放接口，自封装空指针校验宏替代 GE_ASSERT_NOTNULL
#define OPS_ASSERT_NOTNULL(v)                                     \
    do {                                                          \
        if ((v) == nullptr) {                                     \
            OP_LOGE(nullptr, "Assert %s is nullptr failed.", #v); \
            return ge::GRAPH_FAILED;                              \
        }                                                         \
    } while (false)

namespace ops {
using json = nlohmann::json;

bool IsA5OrLaterShortSocVersion(const std::string& shortSocVersion)
{
    constexpr const char* ASCEND_PREFIX = "Ascend";
    constexpr size_t ASCEND_PREFIX_LEN = sizeof("Ascend") - 1U;
    if (shortSocVersion.compare(0, ASCEND_PREFIX_LEN, ASCEND_PREFIX) != 0) {
        return false;
    }

    // 取 "Ascend" 后的连续数字作为型号：Ascend950 -> 950，Ascend910B -> 910，Ascend910_93 -> 910
    int32_t model = 0;
    size_t pos = ASCEND_PREFIX_LEN;
    const size_t modelBegin = pos;
    while (pos < shortSocVersion.size() && std::isdigit(static_cast<unsigned char>(shortSocVersion[pos])) != 0) {
        model = model * 10 + (shortSocVersion[pos] - '0');
        ++pos;
    }
    if (pos == modelBegin) {
        return false;
    }
    // Ascend350 系列为 A5 同代产品，型号数字小于 950，需单独识别；
    // GE 上报串可能带后缀（如 Ascend350_355e，见 dev_type.h 判代表），数字解析自然截断后即可覆盖
    constexpr int32_t A5_GEN_BEGIN_MODEL = 950;
    constexpr int32_t A5_SAME_GEN_MODEL = 350;
    return model >= A5_GEN_BEGIN_MODEL || model == A5_SAME_GEN_MODEL;
}

ge::AscendString BuildSuperKernelArgsFormat()
{
    // HCCL SuperKernel 子算子参数布局固定为：hidden input 在前，随后 input、output
    std::vector<ge::ArgDescInfo> argDescs;
    argDescs.emplace_back(ge::ArgDescInfo::CreateHiddenInput(ge::HiddenInputSubType::kHcclSuperKernel));
    argDescs.emplace_back(ge::ArgDescType::kIrInput, 0);
    argDescs.emplace_back(ge::ArgDescType::kIrOutput, 0);
    return ge::ArgsFormatSerializer::Serialize(argDescs);
}

bool IsGeSuperKernelGenTaskSupported()
{
    // 前置契约：本守卫成立依赖 GE 头侧将 TaskDefBuilder 与 IsVersionSupported 声明为弱符号
    // （cann/ge!5881）——本仓编译探测只校验头文件在位，无法校验 weak 属性；消费侧弱引用是否
    // 真正降级为 WEAK UND 以链接产物为准（nm -D libopgraph_hccl.so 应为 w），GE 包升级时复核。
    // 老 toolkit（<9.3.0）的 GE 库无该符号，弱引用运行期解析为空指针，必须先判空再调用，
    // 避免空指针跳 0
    if (&ge::IsVersionSupported == nullptr) {
        return false;
    }
    return ge::IsVersionSupported("9.3.0"); // 与 src/hccl.cmake 及 UT 探测体中的 "9.3.0" 同口径，修改需三处同步
}

namespace {
    std::atomic<int> taskIndex(0);

    // rt_model.h 为非对外开放接口，本地常量替代其中的 RT_MODEL_TASK_KERNEL（任务类型：kernel 任务）
    constexpr uint32_t RT_MODEL_TASK_KERNEL = 0U;

    // 每次 GenTask 回调使用独立的配置，避免多节点并发编译时相互覆盖
    json MakeAscendConfig()
    {
        return json{
            {"blockDim", 0},
            {"kernelName", ""},
            {"sub_operator_params", {"hiddenInput", "input", "output"}},
            {"sub_operator_kernel_type", "KERNEL_TYPE_AIV_ONLY"},
            {"sub_operator_kernel_name", {{"AiCore", {{"func_name", ""}, {"obj_files", ""}}}}},
            {"sub_operator_early_start_set_flag", false},
            {"sub_operator_early_start_wait_flag", false}};
    }

    bool GetPathFromContext(const gert::ExeResGenerationContext* context, ge::AscendString& strRet)
    {
        std::string attrName = "hcom_bin_file_path";
        return context->GetStrAttrVal(attrName.c_str(), strRet);
    }

    bool GetFuncFromContext(const gert::ExeResGenerationContext* context, ge::AscendString& strRet)
    {
        std::string attrName = "hcom_func_name";
        return context->GetStrAttrVal(attrName.c_str(), strRet);
    }

    bool GetBlockDimFromContext(const gert::ExeResGenerationContext* context, int64_t& intRet)
    {
        std::string attrName = "hcom_block_dim";
        return context->GetIntAttrVal(attrName.c_str(), intRet);
    }

    bool GetKernelMetaFromContext(const gert::ExeResGenerationContext* context, ge::AscendString& strRet)
    {
        std::string attrName = "kernel_meta_parent_dir";
        return context->GetStrAttrVal(attrName.c_str(), strRet);
    }

    // A5 及以后代际平台返回 true，A2/A3 及更早代际返回 false。
    // PlatformInfoManager 为进程级单例，这里加锁并缓存成功结果，避免并发初始化/读取。
    // 离线编译等场景可能取不到默认平台信息，此时尝试用 OptionalInfo 中的 SoC 版本再取一次。
    bool IsA5OrLaterPlatform(const char* nodeName)
    {
        static std::mutex platformMutex;
        static std::optional<bool> isA5OrLater;

        std::lock_guard<std::mutex> lock(platformMutex);
        if (isA5OrLater.has_value()) {
            return *isA5OrLater;
        }

        fe::PlatformInfo platformInfo;
        fe::OptionalInfo optionalInfo;
        auto& platformManager = fe::PlatformInfoManager::Instance();
        if (platformManager.GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != ge::GRAPH_SUCCESS) {
            const std::string& socVersion = optionalInfo.soc_version;
            if (socVersion.empty()) {
                OP_LOGW(nodeName, "Get platform info failed, schedule_mode will not be set in ascend config.");
                return false; // 不缓存失败结果，后续节点可重试
            }
            const uint32_t initRet = platformManager.InitializePlatformInfo(socVersion);
            if (initRet != ge::GRAPH_SUCCESS) {
                OP_LOGW(
                    nodeName, "Initialize platform info failed by socVersion[%s], ret[%u].", socVersion.c_str(),
                    initRet);
                return false;
            }
            if (platformManager.GetPlatformInfo(socVersion, platformInfo, optionalInfo) != ge::GRAPH_SUCCESS) {
                OP_LOGW(nodeName, "Get platform info failed by socVersion[%s].", socVersion.c_str());
                return false;
            }
        }

        isA5OrLater = IsA5OrLaterShortSocVersion(platformInfo.str_info.short_soc_version);
        return *isA5OrLater;
    }

    ge::Status InsertHiddenInputsForAivTaskDef(
        const gert::ExeResGenerationContext* context, gert::KernelContextBuilder& contextBuilder)
    {
        OPS_ASSERT_NOTNULL(context);

        const ge::AscendString newArgsFormat = BuildSuperKernelArgsFormat();
        // Serialize 契约：失败返回空串（如老版本 GE 未实现 kHcclSuperKernel 序列化），须 fail-fast 避免假成功
        if (newArgsFormat.GetString() == nullptr || newArgsFormat.GetLength() == 0U) {
            OP_LOGE(context->GetNodeName(), "Serialize superkernel args format failed, got empty string.");
            return ge::GRAPH_FAILED;
        }
        contextBuilder.SetArgsFormat(newArgsFormat);
        OP_LOGI(context->GetNodeName(), "New args format is %s.", newArgsFormat.GetString());

        return ge::GRAPH_SUCCESS;
    }

    ge::Status HcclInsertHiddenInputForSuperKernel(
        const gert::ExeResGenerationContext* context, gert::TaskDefBuilder& taskDefBuilder, uint32_t blockDim)
    {
        const char* nodeName = context->GetNodeName();
        // 获取 aicore context（KernelDef/KernelContext 视图由 TaskDefBuilder 惰性创建，保证视图非空）
        auto kernelBuilder = taskDefBuilder.GetKernelBuilder();
        kernelBuilder.SetBlockDim(blockDim);
        auto contextBuilder = kernelBuilder.GetContextBuilder();

        OP_CHECK(
            InsertHiddenInputsForAivTaskDef(context, contextBuilder) != ge::GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "Failed to insert hidden input for mix task."), return ge::GRAPH_FAILED);

        OP_LOGD(nodeName, "Modify AICore task for mc2 node successfully.");
        return ge::GRAPH_SUCCESS;
    }

#define HCCL_DIR_PERMISSION 0755
    bool CreateDirectoriesRecursively(const std::string& path)
    {
        // 空路径直接返回成功
        if (path.empty()) {
            return true;
        }
        struct stat st;

        // 1. 检查路径是否已存在
        if (stat(path.c_str(), &st) == 0) {
            if (S_ISDIR(st.st_mode)) {
                return true; // 目录已存在
            } else {
                return false; // 路径是文件
            }
        }
        // 2. 递归创建父目录
        size_t pos = path.find_last_of('/');
        if (pos != std::string::npos) {
            std::string parent = path.substr(0, pos);
            if (!parent.empty() && !CreateDirectoriesRecursively(parent)) {
                return false; // 父目录创建失败
            }
        }
        // 3. 创建当前目录
        if (mkdir(path.c_str(), HCCL_DIR_PERMISSION) != 0) {
            if (errno != EEXIST) { // 忽略其他进程可能已创建的情况
                return false;
            }
        }
        return true;
    }

    // 仅创建文件所在目录，文件由调用方统一打开一次
    bool CreateParentDirectories(const std::string& filepath)
    {
        const size_t lastSlash = filepath.find_last_of('/');
        if (lastSlash == std::string::npos) {
            return true;
        }
        return CreateDirectoriesRecursively(filepath.substr(0, lastSlash));
    }

    // json 落盘后的失败路径清理：成功时由运行时按 json_file_path 消费该文件，
    // 任一失败 return 或异常展开时析构自动删除，避免 kernel_meta 目录遗留无主 json
    struct JsonFileGuard {
        explicit JsonFileGuard(std::string filePath) : path_(std::move(filePath)) {}
        // 栈上作用域 guard，禁止拷贝/移动：拷贝会使 released_ 不共享，导致重复删除或误删已 Release 的文件
        JsonFileGuard(const JsonFileGuard&) = delete;
        JsonFileGuard& operator=(const JsonFileGuard&) = delete;
        JsonFileGuard(JsonFileGuard&&) = delete;
        JsonFileGuard& operator=(JsonFileGuard&&) = delete;
        ~JsonFileGuard()
        {
            if (!released_) {
                (void)std::remove(path_.c_str());
            }
        }
        void Release() { released_ = true; }

        std::string path_;
        bool released_ = false;
    };
} // namespace

// 支持静态图在线编译.o
ge::Status
HcclSuperKernelGenTaskCallback(const gert::ExeResGenerationContext* context, std::vector<std::vector<uint8_t>>& tasks)
{
    const char* nodeName = context->GetNodeName();

    // 版本守卫（注册时机已探测，此处双保险）：老 toolkit（<9.3.0）上 TaskDefBuilder 各方法为
    // 空指针弱引用，任何构造/调用都会跳 0；此处优雅失败，由 GE 默认任务生成路径兜底
    if (!IsGeSuperKernelGenTaskSupported()) {
        OP_LOGW(nodeName, "GE runtime does not support superkernel gen task (requires >= 9.3.0).");
        return ge::GRAPH_FAILED;
    }

    ge::AscendString path;
    ge::AscendString func;
    int64_t blockDim = 0;
    json ascendConfig = MakeAscendConfig();

    // GetStrAttrVal 返回 true 不代表字符串非空；GE 实现对空串的 GetString() 返回静态空串指针（非 nullptr），
    // 故判空须用 GetLength：空路径会导致 json 配置错误或文件写入根目录
    if (!GetPathFromContext(context, path) || path.GetString() == nullptr || path.GetLength() == 0U) {
        OP_LOGE(nodeName, "get path failed.");
        return ge::GRAPH_FAILED;
    }
    ascendConfig.at("sub_operator_kernel_name").at("AiCore").at("obj_files") = path.GetString();

    if (!GetFuncFromContext(context, func) || func.GetString() == nullptr || func.GetLength() == 0U) {
        OP_LOGE(nodeName, "get func failed.");
        return ge::GRAPH_FAILED;
    }
    ascendConfig.at("sub_operator_kernel_name").at("AiCore").at("func_name") = func.GetString();
    ascendConfig.at("kernelName") = func.GetString();

    if (!GetBlockDimFromContext(context, blockDim) || blockDim <= 0) {
        OP_LOGE(nodeName, "get invalid blockDim[%ld].", blockDim);
        return ge::GRAPH_FAILED;
    }
    ascendConfig.at("blockDim") = blockDim;

    // schedule_mode 字段为 A5 及以后平台支持，A2/A3 及更早平台不设置该字段
    if (IsA5OrLaterPlatform(nodeName)) {
        ascendConfig["schedule_mode"] = 1;
    }

    ge::AscendString kernelMeta;
    if (!GetKernelMetaFromContext(context, kernelMeta) || kernelMeta.GetString() == nullptr
        || kernelMeta.GetLength() == 0U) {
        OP_LOGE(nodeName, "get kernelMeta failed.");
        return ge::GRAPH_FAILED;
    }
    // fetch_add 的返回值即本次唯一序号，避免 fetch_add + load 两次原子操作拿到相同值
    const int taskId = taskIndex.fetch_add(1) + 1; // 保持文件名从 1 开始
    // 将ascendConfig写入到指定路径；文件名保持与旧实现（canndev）一致的 Hcom<N>.json 约定，
    // 序号进程内唯一；跨进程隔离依赖 GE 的 kernel_meta 目录约定
    std::string newPath = kernelMeta.GetString();
    newPath = newPath + "/Hcom" + std::to_string(taskId) + ".json";
    OP_LOGI(nodeName, "newPath:%s", newPath.c_str());
    if (!CreateParentDirectories(newPath)) {
        OP_LOGE(nodeName, "Failed to create parent directories for json file.");
        return ge::GRAPH_FAILED;
    }

    // guard 须在文件创建（ofstream 构造即建文件）之前生效：写盘失败/异常 return 时析构清理，
    // 避免遗留无主或半截 json；open 失败时 remove 不存在的文件返回 -1，无害
    JsonFileGuard jsonGuard(newPath);
    try {
        std::ofstream o(newPath, std::ios::out);
        if (!o.is_open()) {
            OP_LOGE(nodeName, "Failed to open file for writing: %s", newPath.c_str());
            return ge::GRAPH_FAILED;
        }
        const std::string ascendStr = ascendConfig.dump();
        o << ascendStr;
        o.close();
        if (!o) { // failbit/badbit（含 close 时 flush 失败）
            OP_LOGE(nodeName, "Failed to write json file: %s", newPath.c_str());
            return ge::GRAPH_FAILED;
        }
    } catch (const std::exception& e) {
        OP_LOGE(nodeName, "Failed to dump json file, reason: %s", e.what());
        return ge::GRAPH_FAILED;
    }
    // 将json的路径信息写入到opdesc中（SetStrAttrVal 为 const 成员，无需 const_cast）
    // path/func/kernelMeta 均已前置判空，此处 binFilePath 保证非空；
    // 写入失败须 fail-fast：bin_file_path/json_file_path 均有运行期消费者（GE 按属性取路径并 dlopen），
    // 缺失会导致"编译成功、运行期加载失败"的任务，此处失败由 JsonFileGuard 自动清理已落盘 json
    const char* binFilePath = path.GetString();
    if (!context->SetStrAttrVal("bin_file_path", binFilePath)) {
        OP_LOGE(nodeName, "Failed to set bin_file_path attr: %s", binFilePath);
        return ge::GRAPH_FAILED;
    }
    if (!context->SetStrAttrVal("json_file_path", newPath.c_str())) {
        OP_LOGE(nodeName, "Failed to set json_file_path attr: %s", newPath.c_str());
        return ge::GRAPH_FAILED;
    }

    gert::TaskDefBuilder superkernelTaskDef;
    const int64_t streamId = context->GetStreamId();
    superkernelTaskDef.SetStreamId(static_cast<uint32_t>(streamId));
    superkernelTaskDef.SetType(RT_MODEL_TASK_KERNEL);

    if (HcclInsertHiddenInputForSuperKernel(context, superkernelTaskDef, static_cast<uint32_t>(blockDim))
        != ge::GRAPH_SUCCESS) {
        OP_LOGE(nodeName, "Insert hidden input failed.");
        return ge::GRAPH_FAILED;
    }

    const size_t pbSize = superkernelTaskDef.ByteSizeLong();
    std::vector<uint8_t> pbBuff(pbSize, 0);
    const bool ret = superkernelTaskDef.SerializeToArray(pbBuff.data(), pbSize);
    if (!ret) {
        OP_LOGE(nodeName, "superkernel_task_def generate failed.");
        return ge::GRAPH_FAILED;
    }
    tasks.emplace_back(std::move(pbBuff));
    jsonGuard.Release(); // 文件已由 json_file_path 交给运行时消费

    OP_LOGD(nodeName, "Generate task for hccl node successfully.");
    return ge::GRAPH_SUCCESS;
}

namespace {
    // 运行期条件注册等价于 IMPL_OP(...).GenerateTask(...) 的静态注册，但注册动作条件化：
    // 版本守卫不支持（老 toolkit）时不注册，保持 GE 对 Hcom 算子的默认任务生成路径兜底。
    // 返回 optional 仅为延长注册对象生命周期（注册在构造时完成），注册表条目由 register 库内部维护，
    // 不依赖本 so 的符号导出
    std::optional<gert::OpImplRegisterV2> MakeSuperKernelReg(const char* opType, bool enabled)
    {
        if (!enabled) {
            return std::nullopt;
        }
        gert::OpImplRegisterV2 reg(opType);
        reg.GenerateTask(HcclSuperKernelGenTaskCallback);
        return reg;
    }
} // namespace

// 注册时机（dlopen 本 so 的静态初始化阶段）做一次版本守卫探测：弱符号在静态初始化执行前已由
// 动态链接器解析完毕（老 toolkit 上为空指针），此后运行环境不再变化，结论全进程一致。
// 不注册仅意味着运行环境 GE 低于 9.3.0：本 so 对 GE 新符号的弱引用全部解析为空指针，
// dlopen 不再因 undefined symbol 失败；Hcom 算子由 toolkit 自带的旧 superkernel 实现接管
// （发行 toolkit 中位于 libopgraph_legacy.so），存量行为不变，实现老 toolkit 兼容。
// 判定不支持时打印一条 INFO：区分"本 so 已加载但 GE 不支持"与"so 未加载"两种部署形态，
// 便于线上诊断；静态初始化期 OP_LOG 对 nullptr opname 安全（op_util.h）
[[maybe_unused]] const bool g_geSuperKernelGenTaskSupported = [] {
    const bool supported = IsGeSuperKernelGenTaskSupported();
    if (!supported) {
        OP_LOGI(
            nullptr, "GE runtime does not support superkernel gen task (requires >= 9.3.0), "
                     "skip registration, fallback to the legacy superkernel task generation.");
    }
    return supported;
}();
[[maybe_unused]] const auto g_regHcomAllGather = MakeSuperKernelReg("HcomAllGather", g_geSuperKernelGenTaskSupported);
[[maybe_unused]] const auto g_regHcomReduceScatter
    = MakeSuperKernelReg("HcomReduceScatter", g_geSuperKernelGenTaskSupported);
[[maybe_unused]] const auto g_regHcomAllReduce = MakeSuperKernelReg("HcomAllReduce", g_geSuperKernelGenTaskSupported);
[[maybe_unused]] const auto g_regHcomAllToAll = MakeSuperKernelReg("HcomAllToAll", g_geSuperKernelGenTaskSupported);
} // namespace ops
