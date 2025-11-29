#!/usr/bin/env csh
# Perform pre-check for hccl package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

set MIN_PIP_VERSION=19
set PYTHON_VERSION=3.7.5

set version=`pip3 --version | cut -d" " -f2 | cut -d"." -f1`
if (${version} < ${MIN_PIP_VERSION}) then
    echo "pip3 version is lower than ${MIN_PIP_VERSION}.x.x"
endif

set python_version=`python3 --version | cut -d" " -f2`
foreach num (1 2 3)
    set version_num=`echo ${python_version} | cut -d"." -f${num}`
    set min_version_num=`echo ${PYTHON_VERSION} | cut -d"." -f${num}`
    if (${version_num} != ${min_version_num}) then
        echo "python version ${python_version} is not equals ${PYTHON_VERSION}."
        break
    endif
end
