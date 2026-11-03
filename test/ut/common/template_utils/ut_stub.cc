/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// UT 独立编译桩：仅闭合链接，不执行通信与日志（先例 reduce_scatter_seq64/ut_stub.cc）。
// 链接为整对象粒度：测试引用 template_utils.o 的任一符号即拉入其全部未定义引用，桩集为最小必要集：
//   log 门控——template_utils.cc 的 HCCL_* 日志宏引用（先例 topo_match/log_stub.cc）；
//   HCOMM 对称窗口——FillChannelSymWinPeerAddrs 引用 GetSymWinRemoteMem（HcclRankGraph* 系为
//   weak 声明不需桩）。若本地构建出现残余未定义符号，按同样方式在本文件追加桩定义即可。
#include "log.h"
#include "hccl_sym_win_dl.h"

// ============ log 门控 ============

bool HcclCheckLogLevel(int logType, int moduleId)
{
    (void)logType;
    (void)moduleId;
    return false;
}

bool IsErrorToWarn() { return false; }

// ============ HCOMM 对称窗口（本 UT 不触达该路径） ============

HcclResult GetSymWinRemoteMem(HcclCommSymWindow winHandle, size_t offset, uint32_t peerRank, void** ptr)
{
    (void)winHandle;
    (void)offset;
    (void)peerRank;
    (void)ptr;
    return HcclResult::HCCL_E_INTERNAL;
}
