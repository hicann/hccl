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
 * \file test_hccl_gen_task_superkernel.cc
 * \brief HCCL SuperKernel GenTask 迁移 UT
 */

#include <cstdint>
#include <cstdio>
#include <dirent.h>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <unistd.h>
#include "gtest/gtest.h"
#include "exe_graph/runtime/task_def_builder.h"
#include "graph/arg_desc_info.h"
#include "graph/ge_version_check.h"
#include "hccl_gen_task_superkernel.h"

namespace ops {
TEST(HcclSuperKernelGenTaskTest, BuildSuperKernelArgsFormat)
{
    const ge::AscendString argsFormat = BuildSuperKernelArgsFormat();
    ASSERT_NE(argsFormat.GetString(), nullptr);
    // hidden input 插在第一个 input 前，token 与运行时旧值 HCCLSUPERKERNEL(2) 逐字符一致
    EXPECT_STREQ(argsFormat.GetString(), "{hi.hcclsk0*}{i0*}{o0*}");
}

TEST(HcclSuperKernelGenTaskTest, IsA5OrLaterShortSocVersion)
{
    // A5 及以后代际
    EXPECT_TRUE(IsA5OrLaterShortSocVersion("Ascend950"));
    EXPECT_TRUE(IsA5OrLaterShortSocVersion("Ascend960"));
    EXPECT_TRUE(IsA5OrLaterShortSocVersion("Ascend970")); // 未来代际
    EXPECT_TRUE(IsA5OrLaterShortSocVersion("Ascend350")); // A5 同代
    // 带后缀形态（见 dev_type.h 判代表）：数字解析截断后判代结论须一致
    EXPECT_TRUE(IsA5OrLaterShortSocVersion("Ascend350_355e"));   // A5 同代带后缀
    EXPECT_TRUE(IsA5OrLaterShortSocVersion("Ascend950PR_958b")); // 950 带后缀

    // A2/A3 及更早代际
    EXPECT_FALSE(IsA5OrLaterShortSocVersion("Ascend910"));
    EXPECT_FALSE(IsA5OrLaterShortSocVersion("Ascend910B"));
    EXPECT_FALSE(IsA5OrLaterShortSocVersion("Ascend910_93"));
    EXPECT_FALSE(IsA5OrLaterShortSocVersion("Ascend310P"));
    EXPECT_FALSE(IsA5OrLaterShortSocVersion("unknown"));
}

TEST(HcclSuperKernelGenTaskTest, IsGeSuperKernelGenTaskSupported)
{
    // 本用例校验"接口在位且版本满足时守卫放行"：UT 套件的整体前置条件由构建探测保证（接口头文件
    // 在位，缺失时整个 UT 在 CMake 层跳过），CANN 包头文件与运行库同源构建，故运行库提供这些符号。
    // IsVersionSupported 经下方链接期 stub 默认转发真实实现；地址判空负路径（弱引用为 nullptr）
    // 进程内无法构造（stub 使符号恒存在），由 GE 侧 ST 覆盖，见 VersionUnsupported 系列用例。
    EXPECT_TRUE(IsGeSuperKernelGenTaskSupported());
}

TEST(HcclSuperKernelGenTaskTest, TaskDefBuilderSerialize)
{
    gert::TaskDefBuilder task;
    task.SetStreamId(1U).SetType(0U);
    task.GetKernelBuilder().SetBlockDim(4U).GetContextBuilder().SetArgsFormat(BuildSuperKernelArgsFormat());

    const size_t size = task.ByteSizeLong();
    ASSERT_GT(size, 0U);
    std::vector<uint8_t> buffer(size, 0U);
    EXPECT_TRUE(task.SerializeToArray(buffer.data(), size));
}

namespace {
    // 回调测试桩：gert::ExeResGenerationContext 为 POD 非虚类，无法接口 stub，
    // 采用仓内链接期 stub 惯例（见 test/ut/common/*/stub），自行定义其 4 个外联方法并拦截到本夹具。
    struct ContextStub {
        std::map<std::string, std::string> strAttrs; // 模拟 context 中存在的字符串属性
        std::map<std::string, int64_t> intAttrs;     // 模拟 context 中存在的整型属性
        std::map<std::string, std::string> setAttrs; // 记录 SetStrAttrVal 写入
        std::string attrLeftEmpty; // 非空时：该属性模拟 GE 返回 true 但未填充 val 的异常场景
        std::string attrWriteFail; // 非空时：SetStrAttrVal 对该属性返回 false 的异常场景
        int64_t streamId = 0;

        void Reset()
        {
            strAttrs.clear();
            intAttrs.clear();
            setAttrs.clear();
            attrLeftEmpty.clear();
            attrWriteFail.clear();
            streamId = 0;
        }

        void SetAllValid()
        {
            strAttrs["hcom_bin_file_path"] = "/tmp/hcom_ut.bin";
            strAttrs["hcom_func_name"] = "hcom_ut_func";
            intAttrs["hcom_block_dim"] = 4;
            strAttrs["kernel_meta_parent_dir"] = "/tmp/hccl_sk_ut_meta";
        }
    };
    ContextStub g_ctxStub;

    // Serialize 失败注入开关：true 时模拟老版本 GE 未实现 kHcclSuperKernel 序列化（返回空串）的场景
    bool g_forceEmptySerialize = false;

    // GE 版本不满足注入开关：true 时经下方 IsVersionSupported stub 构造"版本低于 9.3.0"场景，
    // 覆盖版本守卫 false 分支与回调守卫兜底分支（弱引用判空分支进程内无法构造，由 GE 侧 ST 覆盖）
    bool g_forceGeVersionUnsupported = false;
} // namespace
} // namespace ops

namespace ge {
// 链接期 stub：以强符号覆盖 libgraph 的 IsVersionSupported（extern "C"，按可执行文件优先解析），
// 默认经 RTLD_NEXT 转发真实实现，保证"守卫放行"用例仍验证真实版本比较行为
extern "C" bool IsVersionSupported(const char* version)
{
    if (ops::g_forceGeVersionUnsupported) {
        return false;
    }
    using IsVersionSupportedFn = bool (*)(const char*);
    static const IsVersionSupportedFn realFn
        = reinterpret_cast<IsVersionSupportedFn>(dlsym(RTLD_NEXT, "IsVersionSupported"));
    if (realFn == nullptr) {
        // 运行库无该符号（老 toolkit）时无法转发，按支持处理，保持其余用例口径
        return true;
    }
    return realFn(version);
}

// 链接期 stub：拦截 ArgsFormatSerializer::Serialize，默认经 RTLD_NEXT 转发 libgraph 真实实现，
// 保证 BuildSuperKernelArgsFormat 等既有用例仍验证真实序列化行为
AscendString ArgsFormatSerializer::Serialize(const std::vector<ArgDescInfo>& args_format)
{
    if (ops::g_forceEmptySerialize) {
        return AscendString();
    }
    using SerializeFn = AscendString (*)(const std::vector<ArgDescInfo>&);
    static const SerializeFn realFn = reinterpret_cast<SerializeFn>(
        dlsym(RTLD_NEXT, "_ZN2ge20ArgsFormatSerializer9SerializeERKSt6vectorINS_11ArgDescInfoESaIS2_EE"));
    if (realFn == nullptr) {
        throw std::runtime_error("failed to resolve real ArgsFormatSerializer::Serialize");
    }
    return realFn(args_format);
}
} // namespace ge

namespace gert {
int64_t ExeResGenerationContext::GetStreamId() const { return ops::g_ctxStub.streamId; }

bool ExeResGenerationContext::GetStrAttrVal(const char* attr_name, ge::AscendString& val) const
{
    const auto it = ops::g_ctxStub.strAttrs.find(attr_name);
    if (it == ops::g_ctxStub.strAttrs.end()) {
        return false;
    }
    if (ops::g_ctxStub.attrLeftEmpty == attr_name) {
        return true; // 模拟 GE 返回 true 但 val 为空串（GetString() 返回静态空串，GetLength() 为 0）
    }
    val = ge::AscendString(it->second.c_str());
    return true;
}

bool ExeResGenerationContext::GetIntAttrVal(const char* attr_name, int64_t& val) const
{
    const auto it = ops::g_ctxStub.intAttrs.find(attr_name);
    if (it == ops::g_ctxStub.intAttrs.end()) {
        return false;
    }
    val = it->second;
    return true;
}

bool ExeResGenerationContext::SetStrAttrVal(const char* attr_name, const char* val) const
{
    ops::g_ctxStub.setAttrs[attr_name] = (val != nullptr) ? val : "";
    return ops::g_ctxStub.attrWriteFail != attr_name; // 模拟写入失败的异常场景
}
} // namespace gert

namespace ops {
TEST(HcclSuperKernelGenTaskTest, IsGeSuperKernelGenTaskUnsupported)
{
    // 注入"GE 版本低于 9.3.0"场景：版本守卫须判定不支持（老 toolkit 上不注册回调的判定依据），
    // 由上方 IsVersionSupported 链接期 stub 的注入开关构造
    g_forceGeVersionUnsupported = true;
    EXPECT_FALSE(IsGeSuperKernelGenTaskSupported());
    g_forceGeVersionUnsupported = false;
}

// 零初始化 POD：内联 GetNodeName 对空 compute_node_info 返回 nullptr，OP_LOGx 宏对 nullptr 安全
class HcclSuperKernelGenTaskCallbackTest : public testing::Test {
protected:
    void SetUp() override
    {
        g_ctxStub.Reset();
        g_forceEmptySerialize = false; // 注入开关随用例复位，避免中断的用例残留污染后续用例
        g_forceGeVersionUnsupported = false;
    }
    gert::ExeResGenerationContext ctx_{};
    std::vector<std::vector<uint8_t>> tasks_;
};

TEST_F(HcclSuperKernelGenTaskCallbackTest, GeVersionUnsupportedGuardRejects)
{
    g_ctxStub.SetAllValid();
    // 注入"GE 版本低于 9.3.0"场景：回调入口版本守卫须在 json 落盘等任何 I/O 之前优雅失败，
    // 覆盖老 toolkit 上注册被跳过后回调兜底的判定分支（正常流程下不可达，防御双保险）
    g_forceGeVersionUnsupported = true;
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
    g_forceGeVersionUnsupported = false; // 体内复位，与 SerializeArgsFormatEmpty 同惯例，防 shuffle 顺序污染
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, MissingBinFilePath)
{
    g_ctxStub.SetAllValid();
    g_ctxStub.strAttrs.erase("hcom_bin_file_path");
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, MissingFuncName)
{
    g_ctxStub.SetAllValid();
    g_ctxStub.strAttrs.erase("hcom_func_name");
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, MissingBlockDim)
{
    g_ctxStub.SetAllValid();
    g_ctxStub.intAttrs.erase("hcom_block_dim");
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, InvalidBlockDim)
{
    g_ctxStub.SetAllValid();
    g_ctxStub.intAttrs["hcom_block_dim"] = 0;
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());

    g_ctxStub.intAttrs["hcom_block_dim"] = -1;
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, MissingKernelMeta)
{
    g_ctxStub.SetAllValid();
    g_ctxStub.strAttrs.erase("kernel_meta_parent_dir");
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, UnwritableKernelMetaDir)
{
    g_ctxStub.SetAllValid();
    // /proc 下不可创建目录，CreateParentDirectories 必失败
    g_ctxStub.strAttrs["kernel_meta_parent_dir"] = "/proc/hccl_sk_ut_no_such_dir";
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, JsonOpenFailed)
{
    g_ctxStub.SetAllValid();
    // kernel_meta 直接指向 /proc：目录创建成功但 ofstream 打开必失败（procfs 不可建普通文件），
    // 覆盖 try 块内 open 失败分支（guard 已在作用域内，remove 不存在的文件无害）
    g_ctxStub.strAttrs["kernel_meta_parent_dir"] = "/proc";
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, EmptyPathAttr)
{
    g_ctxStub.SetAllValid();
    // 模拟 GE 返回 true 但值为空串：判空须拦截 fail-fast，避免空路径写入 json 配置
    g_ctxStub.attrLeftEmpty = "hcom_bin_file_path";
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, EmptyFuncAttr)
{
    g_ctxStub.SetAllValid();
    g_ctxStub.attrLeftEmpty = "hcom_func_name";
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, EmptyKernelMetaAttr)
{
    g_ctxStub.SetAllValid();
    // 空 kernelMeta 会拼出根目录路径 /Hcom<N>.json，判空须在拼路径前拦截
    g_ctxStub.attrLeftEmpty = "kernel_meta_parent_dir";
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, SerializeArgsFormatEmpty)
{
    g_ctxStub.SetAllValid();
    // kernel_meta 指向本进程独立临时目录，避免污染工作区
    const std::string metaDir = std::string("/tmp/hccl_sk_ut_") + std::to_string(getpid());
    g_ctxStub.strAttrs["kernel_meta_parent_dir"] = metaDir;

    // 注入 Serialize 空串失败，回调须 fail-fast 且不得产出 task
    g_forceEmptySerialize = true;
    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
    g_forceEmptySerialize = false;

    // 回调在失败前已写盘 json，清理
    if (g_ctxStub.setAttrs.count("json_file_path") != 0U) {
        std::remove(g_ctxStub.setAttrs["json_file_path"].c_str());
    }
    rmdir(metaDir.c_str());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, SetBinFilePathAttrFailed)
{
    g_ctxStub.SetAllValid();
    // kernel_meta 指向本进程独立临时目录，避免污染固定路径
    const std::string metaDir = std::string("/tmp/hccl_sk_ut_") + std::to_string(getpid());
    g_ctxStub.strAttrs["kernel_meta_parent_dir"] = metaDir;
    g_ctxStub.attrWriteFail = "bin_file_path";

    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
    // 失败 return 时 JsonFileGuard 应已清理落盘 json：扫描目录断言为空（bin_file_path 先于
    // json_file_path 写入失败，无法从 setAttrs 取路径，故以目录扫描口径断言清理）
    DIR* dir = opendir(metaDir.c_str());
    ASSERT_NE(dir, nullptr);
    int entryCnt = 0;
    while (struct dirent* ent = readdir(dir)) {
        const std::string name = ent->d_name;
        if (name != "." && name != "..") {
            ++entryCnt;
        }
    }
    closedir(dir);
    EXPECT_EQ(entryCnt, 0);
    rmdir(metaDir.c_str());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, SetJsonFilePathAttrFailed)
{
    g_ctxStub.SetAllValid();
    const std::string metaDir = std::string("/tmp/hccl_sk_ut_") + std::to_string(getpid());
    g_ctxStub.strAttrs["kernel_meta_parent_dir"] = metaDir;
    g_ctxStub.attrWriteFail = "json_file_path";

    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_FAILED);
    EXPECT_TRUE(tasks_.empty());
    // 失败 return 时 JsonFileGuard 应已清理落盘 json，kernel_meta 目录不留无主文件
    ASSERT_EQ(g_ctxStub.setAttrs.count("json_file_path"), 1U);
    struct stat st;
    EXPECT_NE(stat(g_ctxStub.setAttrs["json_file_path"].c_str(), &st), 0);
    rmdir(metaDir.c_str());
}

TEST_F(HcclSuperKernelGenTaskCallbackTest, Success)
{
    g_ctxStub.SetAllValid();
    // kernel_meta 指向本进程独立临时目录，避免污染工作区
    const std::string metaDir = std::string("/tmp/hccl_sk_ut_") + std::to_string(getpid());
    g_ctxStub.strAttrs["kernel_meta_parent_dir"] = metaDir;

    EXPECT_EQ(HcclSuperKernelGenTaskCallback(&ctx_, tasks_), ge::GRAPH_SUCCESS);
    ASSERT_EQ(tasks_.size(), 1U);

    // json_file_path/bin_file_path 应已写入 opdesc，文件名保持 Hcom<N>.json 旧实现约定
    ASSERT_EQ(g_ctxStub.setAttrs.count("json_file_path"), 1U);
    EXPECT_EQ(g_ctxStub.setAttrs["bin_file_path"], "/tmp/hcom_ut.bin");
    const std::string& jsonPath = g_ctxStub.setAttrs["json_file_path"];
    const std::string expectedBase = std::string(metaDir) + "/Hcom";
    EXPECT_NE(jsonPath.find(expectedBase), std::string::npos);

    // json 内容字段校验
    std::ifstream in(jsonPath);
    ASSERT_TRUE(in.is_open());
    const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    EXPECT_NE(content.find("\"blockDim\":4"), std::string::npos);
    EXPECT_NE(content.find("hcom_ut_func"), std::string::npos);
    EXPECT_NE(content.find("/tmp/hcom_ut.bin"), std::string::npos);

    std::remove(jsonPath.c_str());
    rmdir(metaDir.c_str());
}
} // namespace ops
