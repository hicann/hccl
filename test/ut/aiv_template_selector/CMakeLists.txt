# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

file(GLOB UT_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*_test.cc")
add_executable(aiv_template_selector_test ${UT_FILES}
    ${PROJECT_SOURCE_DIR}/src/ops/all_reduce/algorithm/template/aiv/aiv_temp_all_reduce_mesh_1D_twoshot.cc
    ${PROJECT_SOURCE_DIR}/src/ops/all_to_all_v/algorithm/template/aiv/aiv_temp_all_to_all_mesh_1D.cc
    ${PROJECT_SOURCE_DIR}/src/ops/reduce_scatter/algorithm/template/aiv/aiv_temp_reduce_scatter_mesh_1D.cc
    ${PROJECT_SOURCE_DIR}/src/ops/all_gather/algorithm/template/aiv/aiv_temp_all_gather_mesh_1D.cc
)
target_compile_features(aiv_template_selector_test PRIVATE cxx_std_17)
target_include_directories(aiv_template_selector_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stub
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/op_common/topo_info
    ${PROJECT_SOURCE_DIR}/src/ops/all_reduce/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/all_reduce/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/all_to_all_v/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/all_to_all_v/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/broadcast/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/reduce/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/reduce_scatter/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/reduce_scatter/algorithm/template
    ${PROJECT_SOURCE_DIR}/src/ops/all_gather/algorithm/template/aiv
    ${PROJECT_SOURCE_DIR}/src/ops/all_gather/algorithm/template
)
target_link_libraries(aiv_template_selector_test PRIVATE
    $<BUILD_INTERFACE:intf_llt_pub>
    GTest::gtest_main
)
run_llt_test(TARGET aiv_template_selector_test TASK_NUM 1)
