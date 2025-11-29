#!/usr/bin/env python
#-*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Purpose:
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.
#----------------------------------------------------------------------------
import argparse
import textwrap
import os
import shutil
import sys

from tools import *

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''
                                     A tool to pack image with new structure'''))
    parser.add_argument('-raw_img', required=False, dest='raw', help='INPUT: The raw image')
    parser.add_argument('-out_img', required=False, dest='out',
                        help='OUTPUT: The processed image. Filename can be same to input image')
    parser.add_argument('-platform', required=False, dest='platform',
                        choices=['ascend'],
                        help='INPUT: platform : ascend')

    # input for cms
    parser.add_argument('-cms', required=False, dest='cms', help='INPUT: The cms file')
    parser.add_argument('-ini', required=False, dest='ini', help='INPUT: The ini file')
    parser.add_argument('-crl', required=False, dest='crl', help='INPUT: The crl file')

    # 1910_version
    parser.add_argument('-version', required=False, dest='ver', help='INPUT: The version number')

    # flag cmd
    parser.add_argument('-S', help="choose whether Onchiprom", action="store_true")

    parser.add_argument('--addcms', help="choose whether add cms", action="store_true")

    parser.add_argument('-position',  required=False, choices=['before_header', 'after_header'], help='INPUT: The relative position of raw_img and head')
    parser.add_argument('-pkt_type',  required=False, choices=['normal_pkt', 'large_pkt'], nargs='?', const='large_pkt', default='large_pkt', help='INPUT: The large_pkt support larger than 4GB packet')
    parser.add_argument('-partition_size',  required=False, nargs='?', default='2048', help='INPUT: The rootfs/app.img total size(M)')

    # nvcnt
    parser.add_argument('-nvcnt', required=False, dest='nvcnt', nargs='?', const=None, help='INPUT: nvcnt for driver images')
    parser.add_argument('-tag', required=False, dest='tag', nargs='?', const=None, help='INPUT: tag for driver images')

    parser.add_argument('-certtype',  required=False, dest='certtype', choices=[0x1, 0x2, 0xFFFFFFFF], default=0xFFFFFFFF, type=int,
                        help='INPUT: 0x1:Community Certificate, 0x2:Client Certificate, 0xFFFFFFFF:HW Certificate')
    return parser.parse_args()

def check_image_headered(file_path):
    with open(file_path, 'rb') as f:
        # 读取前4个字节
        data = f.read(4)
        if len(data) < 4:
            return False
        # 将字节转换为小端序的32位整数
        word = int.from_bytes(data, byteorder='little')
        return word == 0x55aa55aa

def main():
    with open(args.raw, "rb") as f:
        hash_buf = cal_image_hash(f)
        code_len = get_filelen(f)

        tmp_file = args.out + '.tmp'
        with open(tmp_file, 'wb+') as o_f:
            platforms.write_header_huawei(args, o_f, hash_buf, code_len)
            platforms.write_image(args, o_f)
            platforms.write_cms(args, o_f, code_len)
            platforms.write_extern(args, o_f, [hash_buf, code_len])

        shutil.copyfile(tmp_file, args.out)
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

if __name__ == '__main__':
    args = get_args()

    if check_image_headered(args.raw) == True :
        print("Detected 8K header magic number, No need to add head again")
        sys.exit()  # 退出程序

    if args.platform == 'ascend' :
        import hi_platform.platform_ascend as platforms

    main()
