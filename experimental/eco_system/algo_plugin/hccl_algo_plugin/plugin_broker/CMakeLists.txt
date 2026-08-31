# PluginBroker（libhccl_algo_PluginBroker.so）独立构建脚本
# 与HCCL主库完全解耦编译。
#
# 用法：
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j
#   # 产出 libhccl_algo_PluginBroker.so，安装到 ${ASCEND_HOME}/opp/vendors/cust/lib64/ 下

cmake_minimum_required(VERSION 3.14)
project(hccl_algo_plugin_broker CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -fstack-protector-strong -D_FORTIFY_SOURCE=2)

add_library(hccl_algo_PluginBroker SHARED
    src/plugin_broker.cc
)

target_include_directories(hccl_algo_PluginBroker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../src/algo_plugin/inc
)

# 仅显式导出 GetHcclAlgoPlugin 符号，其余符号默认隐藏，减少符号冲突面
set_target_properties(hccl_algo_PluginBroker PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN OFF
)

target_link_libraries(hccl_algo_PluginBroker PRIVATE dl pthread)

install(TARGETS hccl_algo_PluginBroker
    LIBRARY DESTINATION lib64
)
