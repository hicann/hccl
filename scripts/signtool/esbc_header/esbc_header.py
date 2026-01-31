#!/usr/bin/env python
#-*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Purpose:
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.
#----------------------------------------------------------------------------
import argparse
import textwrap
import struct
import shutil
import os
import hashlib
import binascii
import sys


def to_bytes(n, length, endianness='big'):
    h = '%x' % n
    s = ('0' * (len(h) % 2) + h).zfill(length * 2).decode('hex')
    return s if endianness == 'big' else s[::-1]

def get_filelen(f):
    f.seek(0, 2)
    length = f.tell()
    f.seek(0)
    return length

def trans_hw_logic_version(version):
    """Transform hw_logic_version to hex string.

    >>> trans_hw_logic_version('')
    'ffffffff'
    >>> trans_hw_logic_version('0')
    'a5000000'
    >>> trans_hw_logic_version('235')
    'a500eb00'
    >>> trans_hw_logic_version('300')
    'a5002c01'
    """
    if not version:
        return 'ffffffff'
    magic_num = 0xa5
    magic_num_hex = magic_num.to_bytes(1, byteorder='little').hex()
    reserved = 0
    reserved_hex = reserved.to_bytes(1, byteorder='little').hex()
    version = int(version)
    version_hex = version.to_bytes(2, byteorder='little').hex()

    return magic_num_hex + reserved_hex + version_hex

def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''A tool to pack image with new structure'''))
    parser.add_argument('-raw_img', required=True, dest='raw', help='INPUT: The raw image')
    parser.add_argument('-out_img', required=True, dest='out', help='OUTPUT: The processed image.')
    parser.add_argument('-version', required=True, dest='ver', help='INPUT: The version number')
    parser.add_argument('-nvcnt', required=True, dest='nvcnt', choices=range(32), type=int, help='INPUT: the secure version number')
    parser.add_argument('-tag', required=True, dest='tag', help='INPUT: image name tag')
    parser.add_argument('-position',  choices=['before_header', 'after_header'], help='INPUT: The relative position of raw_img and head')
    parser.add_argument('-hw_logic_version', default='', type=trans_hw_logic_version, help='INPUT: hw logic version')
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

if __name__ == '__main__':
    args = get_args()

    # 判断镜像是否已加头
    if check_image_headered(args.raw) == True :
        print("Detected 8K header magic number, No need to add head again")
        sys.exit()  # 退出程序

    # 加ESBC 256Byte头
    s = struct.Struct('16sII64sII8s16s3I4s4sI112s')
    code_tag = bytes(args.tag,'ascii')
    nvcnt_code = args.nvcnt
    hash_alg = 0
    hash_ex = hashlib.sha256()
    code_offset = 0 if (args.position == 'before_header') else 0x100
    reserved1 = int(0).to_bytes(8, 'big')
    ver_value = bytes(args.ver, 'ascii')
    magic_num = 0x3a3aaa33
    sign_enable_field = 0x4
    hashtree_offset = 0x20000  #128K
    hw_logic_version = binascii.a2b_hex(args.hw_logic_version)
    version_magic = 0x564D   # magic VM(version magic)
    version_num = 0          # version, now is 0, reserved used in the future
    image_version = version_magic.to_bytes(2, 'little') + version_num.to_bytes(2, 'little')
    hwheader_offset = 256
    padding = "ff" * 112 # fixed padding at the end of imgdesc
    reserved2 = binascii.a2b_hex(padding)
    fs_offset = 0 if (args.position == 'before_header') else 0x100

    with open(args.raw, 'rb') as f:
        code_len = get_filelen(f)
        esbc_offset = code_len if (args.position == 'before_header') else 0
        tmp_file = args.out + '.tmp'
        f.seek(0)
        with open(tmp_file, 'wb') as o_f:
            o_f.seek(fs_offset)
            code = b''
            for byte_block in iter(lambda: f.read(4096), b""):
                o_f.write(byte_block)
                hash_ex.update(byte_block)
            code_hash = hash_ex.digest()
            pack_list = (code_tag, nvcnt_code, hash_alg, code_hash, code_offset, code_len, reserved1,
            ver_value, magic_num, sign_enable_field, hashtree_offset,
            hw_logic_version, image_version, hwheader_offset, reserved2)
            header = s.pack(*pack_list)
            o_f.seek(esbc_offset)
            o_f.write(header)
        shutil.copyfile(tmp_file, args.out)

        if os.path.exists(tmp_file):
            os.remove(tmp_file)