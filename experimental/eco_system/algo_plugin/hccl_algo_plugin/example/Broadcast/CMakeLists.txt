# Broadcast 自定义算法示例构建脚本
# 验证场景二：一个算法（BroadcastAlgoTree）独占一个实现so
# 与AllReduce示例合起来使用时，还验证场景三：不同算子各自正确路由到对应selector/so
#
# 用法：
#   mkdir build && cd build
#   cmake .. -DASCEND_HOME=/usr/local/Ascend/ascend-toolkit/latest
#   make -j
#   mkdir -p ${HCCL_PLUGIN_ALG_DIR}/Broadcast
#   cp libhccl_plugin_broadcast_selector.so libBroadcastCustomAlgoImpl.so ${HCCL_PLUGIN_ALG_DIR}/Broadcast/

cmake_minimum_required(VERSION 3.14)
project(broadcast_custom_algo_example CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT ASCEND_HOME)
    set(ASCEND_HOME $ENV{ASCEND_HOME})
endif()
if(NOT ASCEND_HOME)
    message(FATAL_ERROR "请通过 -DASCEND_HOME=... 指定CANN安装目录，或设置ASCEND_HOME环境变量")
endif()

# --- libhccl_plugin_broadcast_selector.so ---
add_library(hccl_plugin_broadcast_selector SHARED
    selector/broadcast_selector.cc
)
target_include_directories(hccl_plugin_broadcast_selector PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../sdk
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../src/algo_plugin/inc
)
set_target_properties(hccl_plugin_broadcast_selector PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN OFF
    LINK_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/selector/export.map"
)

# --- libBroadcastCustomAlgoImpl.so（独占，只导出一个算法的执行函数） ---
add_library(BroadcastCustomAlgoImpl SHARED
    op_host/broadcast_custom_algo.cc
)
target_include_directories(BroadcastCustomAlgoImpl PRIVATE
    ${ASCEND_HOME}/include
)
target_link_directories(BroadcastCustomAlgoImpl PRIVATE ${ASCEND_HOME}/lib64)
target_link_libraries(BroadcastCustomAlgoImpl PRIVATE hccl ascendcl)
