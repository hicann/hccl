#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

echo $(grep -E "^VERSION_ID=" /etc/os-release | cut -d'"' -f2)
sudo update-alternatives --set gcc /usr/bin/gcc-14
gcc --version
source /home/jenkins/Ascend/cann/bin/setenv.bash
set +e

DP_ASSERT_EQUAL()
{
    local actual="$1"
    local expected="$2"
    local msg="$3"
    if [ "${actual}" != "${expected}" ]; then
        echo "::error::ASSERT FAILED: ${msg} (expected=${expected}, actual=${actual})"
        exit 1
    fi
}

case "${ut_type}" in
    ut)
        bash build.sh --ut --cann_3rd_lib_path=/home/jenkins/opensource | tee test_ut.log
        ret=${PIPESTATUS[0]}
        ;;
    st)
        if [ "${TARGET_BRANCH}x" != "masterx" ]; then
            exit 0
        fi
        pip3 install Pyyaml
        wget -nv https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/cann-hccl_linux-x86_64_ubuntu24.run
        chmod u+x cann-hccl_linux-x86_64_ubuntu24.run
        sudo chmod 777 /home/jenkins/Ascend
        yes "y" | bash cann-hccl_linux-x86_64_ubuntu24.run --full --install-path=/home/jenkins/Ascend
        export ASCEND_HOME_PATH=/home/jenkins/Ascend/cann
        source /home/jenkins/Ascend/cann/bin/setenv.bash
        bash build.sh --st | tee test_st.log
        ret=${PIPESTATUS[0]}
        ;;
    *)
        echo "Skip UT test execution for ${ut_type} on non-master branch"
        exit 0
        ;;
esac

DP_ASSERT_EQUAL "$ret" "0" "Run UT TESTCASE"

echo "ut_process=ut_cov" >> "${ATOMGIT_OUTPUT}"
