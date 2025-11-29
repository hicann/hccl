#!/usr/bin/env fish
# Perform pre-check for hccl package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

set -x MIN_PIP_VERSION 19
set -x PYTHON_VERSION 3.7.5

set -lx version (pip3 --version | cut -d" " -f2 | cut -d"." -f1)
if test {$version} -lt {$MIN_PIP_VERSION}
    echo "pip3 version is lower than $MIN_PIP_VERSION.x.x"
end

set -lx python_version (python3 --version | cut -d" " -f2)
for num in (seq 1 3)
    set -lx version_num (echo {$python_version} | cut -d"." -f{$num})
    set -lx min_version_num (echo {$PYTHON_VERSION} | cut -d"." -f{$num})
    if test {$version_num} -ne {$min_version_num}
        echo "python version $python_version is not equals $PYTHON_VERSION."
        break
    end
end
