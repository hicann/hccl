# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_subdirectory(common/prepare_ut_env)
add_subdirectory(aiv_template_selector)
add_subdirectory(common/alg_parse)
add_subdirectory(common/topo_match)
add_subdirectory(common/adapter_error_manager)
add_subdirectory(reduce_scatter_selector)
add_subdirectory(common/executor)
add_subdirectory(common/channel_process_link)
add_subdirectory(common/ccu_resource_expansion)
add_subdirectory(common/template_utils)
add_subdirectory(selector_engine)
add_subdirectory(alltoall_hier)
add_subdirectory(hccl_gen_task_superkernel)
add_subdirectory(aicpu_task_cache)
# reduce_scatter_seq64: RS/AR AicpuReduce 模板 cost 建模 + MeshNHRAicpuReduce attrs 注册准入。
# 测 src/ 产品代码，与 alltoall_hier 同级无条件注册；注意 build.sh -u 的 UT 构建不传
# ENABLE_EXPERIMENTAL，放入下方 if(ENABLE_EXPERIMENTAL) 分支将不被 -u 覆盖。
add_subdirectory(reduce_scatter_seq64)
add_subdirectory(alltoall_aiv)
add_subdirectory(all_gather_aiv)
# reduce_scatter_aiv: RS 写语义 kernel（LocalTree / BigData）的 CPU 桩级覆盖，含 flag 依赖与输出平铺校验。
add_subdirectory(reduce_scatter_aiv)
# reduce_aiv: Reduce pull 语义 kernel（MultiCore / MultiRank）的 CPU 桩级覆盖，
# 含 flag 依赖与死锁、发布区不被归约写覆盖、跨卡读源已发布、root 输出平铺、SyncAll 配平校验。
add_subdirectory(reduce_aiv)
# all_reduce_aiv: AllReduce TwoShot(多核路径/少核路径) 的 CPU 桩级覆盖，校验项同 reduce_aiv。
add_subdirectory(all_reduce_aiv)
# allreduce_aiv: AllReduce TwoShot 写语义 kernel 的 CPU 桩级覆盖。
add_subdirectory(allreduce_aiv)
# recursive_executor 测试仅在 --experimental 构建（ENABLE_EXPERIMENTAL=ON）时编译
if(ENABLE_EXPERIMENTAL)
    add_subdirectory(reduce_scatter_birs)
    add_subdirectory(recursive_executor)
    add_subdirectory(all_reduce_biar)
    add_subdirectory(reduce_scatter_v_birs)
endif()
