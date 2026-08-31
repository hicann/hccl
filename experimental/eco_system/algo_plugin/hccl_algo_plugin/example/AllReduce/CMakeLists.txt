# AllReduce 自定义算法示例构建脚本
# 验证场景一：多个算法（AllReduceAlgoSmall / AllReduceAlgoLarge）对应同一个实现so
#
# 用法：
#   mkdir build && cd build
#   cmake .. -DASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
#   make -j
#   mkdir -p ${HCCL_PLUGIN_ALG_DIR}/AllReduce
#   cp libhccl_plugin_allreduce_selector.so libAllReduceCustomAlgosImpl.so ${HCCL_PLUGIN_ALG_DIR}/AllReduce/

cmake_minimum_required(VERSION 3.14)
project(allreduce_custom_algo_example CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT ASCEND_HOME)
    set(ASCEND_HOME $ENV{ASCEND_HOME})
endif()
if(NOT ASCEND_HOME)
    message(FATAL_ERROR "请通过 -DASCEND_HOME=... 指定CANN安装目录，或设置ASCEND_HOME环境变量")
endif()

# --- libhccl_plugin_allreduce_selector.so ---
add_library(hccl_plugin_allreduce_selector SHARED
    selector/allreduce_selector.cc
)
target_include_directories(hccl_plugin_allreduce_selector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../sdk
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../src/algo_plugin/inc
)
set_target_properties(hccl_plugin_allreduce_selector PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN OFF
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/selector/export.map"
)

# --- libAllReduceCustomAlgosImpl.so（同时导出两个算法的执行函数） ---
add_library(AllReduceCustomAlgosImpl SHARED
    op_host/allreduce_custom_algos.cc
)
target_include_directories(AllReduceCustomAlgosImpl PRIVATE
    ${ASCEND_HOME}/include
)
target_link_directories(AllReduceCustomAlgosImpl PRIVATE ${ASCEND_HOME}/lib64)
target_link_libraries(AllReduceCustomAlgosImpl PRIVATE hccl ascendcl)
