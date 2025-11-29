#!/usr/bin/env csh
# Perform setenv for hccl package
# Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.

set CURFILE=`readlink -f ${1}`
set CURPATH=`dirname ${CURFILE}`

set install_info="$CURPATH/../ascend_install.info"
set hetero_arch=`grep -i "Hccl_Hetero_Arch_Flag=" "$install_info" | cut --only-delimited -d"=" -f2`
if ( "$2" == "multi_version" ) then
    if ( "$hetero_arch" == "y" ) then
        set INSTALL_DIR="`realpath ${CURPATH}/../../../../../latest`/hccl"
    else
        set INSTALL_DIR="`realpath ${CURPATH}/../../../latest`/hccl"
    endif
else
    set INSTALL_DIR="`realpath ${CURPATH}/..`"
endif

set lib_path="${INSTALL_DIR}/python/site-packages/"
if ( -d "${lib_path}" ) then
    set python_path=""
    if ( $?PYTHONPATH == 1 ) then
        set python_path="$PYTHONPATH"
    endif
    set num=`echo ":${python_path}:" | grep ":${lib_path}:" | wc -l`
    if ( "${num}" == 0 ) then
        if ( "-${python_path}" == "-" ) then
            setenv PYTHONPATH ${lib_path}
        else
            setenv PYTHONPATH ${lib_path}:${python_path}
        endif
    endif
endif

set library_path="${INSTALL_DIR}/lib64"
if ( -d "${library_path}" ) then
    set ld_library_path=""
    if ( $?LD_LIBRARY_PATH == 1 ) then
        set ld_library_path="$LD_LIBRARY_PATH"
    endif
    set num=`echo ":${ld_library_path}:" | grep ":${library_path}:" | wc -l`
    if ( "$num" == 0 ) then
        if ( "-${ld_library_path}" == "-" ) then
            setenv LD_LIBRARY_PATH "${library_path}:${library_path}/plugin/nnengine:${library_path}/stub"
        else
            setenv LD_LIBRARY_PATH "${library_path}:${library_path}/plugin/nnengine:${ld_library_path}:${library_path}/stub"
        endif
    endif
endif

set custom_path_file="$INSTALL_DIR/../conf/path.cfg"
set common_interface="$INSTALL_DIR/script/common_interface.csh"
set owner="`stat -c %U $CURFILE`"
if ( "`id -u`" != 0 && "`id -un`" != "$owner" && -f "$custom_path_file" && -f "$common_interface" ) then
    csh -f "$common_interface" mk_custom_path "$custom_path_file"
    foreach dir_name ("conf" "data")
        set dst_dir="`grep -w "$dir_name" "$custom_path_file" | cut --only-delimited -d"=" -f2-`"
        set dst_dir="`eval echo $dst_dir`"
        if ( -d "$INSTALL_DIR/$dir_name" && -d "$dst_dir" ) then
            chmod -R u+w $dst_dir/* >& /dev/null
            cp -rfL $INSTALL_DIR/$dir_name/* "$dst_dir"
        endif
    end
endif
