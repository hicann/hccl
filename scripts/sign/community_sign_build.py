#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 签名命令：./client --config ./client.toml add --file-type p7s --key-type x509 --key-name SignCert --detached infile
# 传入签名文件
# 解析参数
# 验证参数
# 执行签名命令
# 判断.p7s文件是否生成，没有生成则签名失败
# 返回执行结果
# Purpose:
# Copyright Huawei Technologies Co., Ltd. 2010-2023. All rights reserved.
# ----------------------------------------------------------------------------

import os
import sys
import logging
import subprocess
from subprocess import PIPE, STDOUT
from typing import List, Optional, Tuple

myfile = os.path.realpath(__file__)
mypath = os.path.dirname(myfile)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s line:%(lineno)d %(levelname)s:%(name)s:%(message)s',
                    datefmt='%H:%M:%S')

def _get_sign_filename() -> Tuple[Optional[str], Optional[str]]:
    """获取签名文件名。"""
    crlfile = "SWSCRL.crl"
    cmstag = ".p7s"
    return crlfile, cmstag

def _get_sign_crl(signtype, default_crl):
    """获取crl路径"""
    sign_crl = default_crl

    if signtype in ("atlas_cms", "cms_pss"):
        # 计算产品线会往如下路径下归档crl，并用以签名
        sign_crl = os.path.join(mypath, "../cert_path/pss/SWSCRL.crl")
        if not os.path.exists(sign_crl):
            return default_crl

    if signtype == "cms_ch_pss":
        sign_crl = os.path.join(mypath, "../cert_path/whitebox/SWSCRL.crl")

    return sign_crl

def _check_result(inputfile) -> bool:
    """签名后处理"""
    crlfile, cmstag = _get_sign_filename()
    if crlfile is None or cmstag is None:
        logging.error("get cms or crl file name error")
        return False
    for file in inputfile:
        cms = file + cmstag
        if not os.path.isfile(cms):
            logging.error("cms file:%s is not exist", cms)
            return False
    return True

def _help():
    print("==================================== 帮助信息 ==================================")
    print("通用命令，命令格式如下:")
    print("python community_sign_build.py [cmd] target ...")
    print("--------------------------------------------------------------------------------")
    print("[cmd] help|cms|")
    print("  %s: 查看帮助" % ("help".ljust(8)))
    print("  %s: 制作cms签名" % ("cms".ljust(8)))
    print("--------------------------------------------------------------------------------")
    print("  %s: 待签名的文件路径,支持多target,各target以空格分开" % ("target".ljust(8)))
    print("====================================== END =====================================")

def get_sign_cmd(file) -> str:
    """获取签名命令。"""
    sign_command = ("./home/jenkins/signclient/client --config ./home/jenkins/signclient/client.toml add "
                    "--file-type p7s --key-type x509 --key-name SignCert --detached ")
    cmd = "{} {}".format(sign_command, file)
    return cmd

def _run_sign(inputfiles):
    """执行签名。"""
    crlfile, cmstag = _get_sign_filename()
    ret=True
    for file in inputfiles:
        if not os.path.isfile(file):
            logging.warning("input file:%s is not exist", file)
            continue
        crl = os.path.join(mypath, crlfile)
        cms = file + cmstag
        for cmsfile in [crl, cms]:
            if os.path.isfile(cmsfile):
                logging.info("begin to remove old file:%s", cmsfile)
                os.remove(cmsfile)
            if os.path.isfile(cmsfile):
                logging.error("remove old file:%s fail", cmsfile)
                return False

        cmd = get_sign_cmd(file)

        logging.info("run sign cmd %s in %s", cmd, mypath)
        result = subprocess.run(cmd, cwd=mypath, shell=True, check=False, stdout=PIPE, stderr=STDOUT)
        if 0 != result.returncode:
            logging.error(result.stdout.decode())
            logging.error("file %s signed error",file)
            ret = False
            break
    return ret

# 多个文件签名场景需要拆分分别签
def main(argv):
    """主流程。"""
    if (len(argv)) < 1:
        logging.error(
            "argv number is error, it must >= 1, now (%s)", str(argv))
        sys.exit(1)

    inputfiles = argv[1:]
    # 初始化签名环境
    ret = _run_sign(inputfiles)

    if ret is not False:
        if not _check_result(inputfiles):
            logging.error("check signature result fail")
            sys.exit(1)
    else:
        logging.error("signature build fail")
        sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main(sys.argv)
