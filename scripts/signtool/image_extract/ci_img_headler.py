#!/usr/bin/env python
#-*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Purpose:
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.
#----------------------------------------------------------------------------
import argparse
import textwrap
import shutil
import os
import struct

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent(''' A tool to recover raw img'''))
    parser.add_argument('-img', required=True, dest='img',
                        help='INPUT: img with header, Parse N/E from it')
    parser.add_argument('-raw', required=False, dest='raw',
                        help='INPUT: recovered raw img')
    parser.add_argument('--rcvr', help="recover the raw img without header",
                        action="store_true")

    return parser.parse_args()

def __write_raw_img(raw, img, code_len):
    raw.seek(0)
    img.seek(0x2100)
    rsv_len = code_len - 0x100
    while rsv_len > 0:
        if rsv_len > 4096:
            raw.write(img.read(4096))
        else:
            raw.write(img.read(rsv_len))
        rsv_len -= 4096

def main():
    args = get_args()
    if args.rcvr:
        with open(args.img, 'rb') as img:
            img.seek(0x478)
            code_len = struct.unpack('<I', img.read(4))[0]
            tmp_file = args.raw + '.tmp'
            with open(tmp_file, 'wb+') as raw:
                __write_raw_img(raw, img, code_len)
        shutil.copyfile(tmp_file, args.raw)
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

if __name__ == '__main__':
    main()


