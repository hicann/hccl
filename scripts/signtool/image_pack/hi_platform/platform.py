#!/usr/bin/env python
#-*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Purpose:
# Copyright Huawei Technologies Co., Ltd. 2010-2025. All rights reserved.
#----------------------------------------------------------------------------
import struct
import sys
from tools import *
import binascii
from ctypes import *
import os

def __construct_header(N_buf, E_buf, hash_buf, code_len, suffix, head_type, version, nvcnt, tag, certtype, before_header=False, 
                       large_packet=False, enc=False, pss=False, bcm=False, gcm = False, gm = False):

    # if rootfs/app.img is large_packet(>4G), stub code_len 0 (invalid value)
    code_len = 0 if before_header and large_packet else code_len
    header_base = 0x1000 if head_type else 0
    zero_bytes_32 = int(0).to_bytes(32, 'big') if sys.version > '3' else to_bytes(0, 32)
    s = struct.Struct('I20sII32s32sIIIII512s512sIIIIII32sI16s16s16s88sIIII20sI32s16s8s16s16s16sI52s')
    preamble = 0x55AA55AA
    rev0 = int(0).to_bytes(20, 'big')
    head_len = 0x600
    user_len = 0x0
    user_define_data = int(0).to_bytes(32, 'big')
    code_hash = hash_buf
    sub_key_cert_offset = 0x600 + header_base
    sub_cert_len = 0x618
    uw_rootkey_alg = 0x0
    img_sign_algo = 0x8010000
    root_pubkey_len = 512
    root_pubkey_n = N_buf
    root_pubkey_e = E_buf
    img_offset = 0 if before_header else 0x2000
    img_sign_obj_len = code_len
    sign_offset = header_base + 0xE00
    sign_len = 512
    code_encrypt_flag = 0xFFFFFFFF
    code_encrypt_algo = 0x2
    derive_seed = zero_bytes_32
    km_ireation_cnt = 1000
    code_encrypt_iv = zero_bytes_32[:16]
    code_encrypt_tag = zero_bytes_32[:16]
    code_encrypt_add = zero_bytes_32[:16]
    rsv1 = int(0).to_bytes(88, 'big')
    rsv2 = int(0).to_bytes(20, 'big')

    if bcm:
        h2c_enable = 0x41544941
        h2c_cert_len = 0x800
        h2c_cert_offset = 0x1000
    else:
        h2c_enable = 0xA5A55555
        h2c_cert_offset = 0
        h2c_cert_len = 0
    root_pubkeyinfo = 0
    head_magic = 0x33CC33CC
    head_hash = zero_bytes_32 # fill in zero first, it will be calculated and filled later
    cms_flag = int(0).to_bytes(16, 'big')
    code_nvcnt = int(0).to_bytes(8, 'big')
    if tag == None:
        code_tag = int(0).to_bytes(16, 'big')
    else:
        code_tag = bytes(str(tag),'ascii')
    ver_value = bytes(version, 'ascii')
    ver_padding_val = "ff" * 0x10
    ver_padding = binascii.a2b_hex(ver_padding_val)
    padding_val = "ff" * 0x44 # fixed padding at the end of header
    padding = binascii.a2b_hex(padding_val)
    pack_list = (preamble, rev0, head_len, user_len, user_define_data, code_hash,
                 sub_key_cert_offset, sub_cert_len, uw_rootkey_alg, img_sign_algo, root_pubkey_len, root_pubkey_n,
                 root_pubkey_e, img_offset, img_sign_obj_len, sign_offset, sign_len, code_encrypt_flag, code_encrypt_algo,
                 derive_seed, km_ireation_cnt, code_encrypt_iv, code_encrypt_tag, code_encrypt_add, rsv1, h2c_enable,
                 h2c_cert_len, h2c_cert_offset, root_pubkeyinfo, rsv2, head_magic, head_hash, cms_flag, code_nvcnt,
                 code_tag, ver_value, ver_padding, certtype, padding)
    header = s.pack(*pack_list)
    return header

def __get_filelen(f):
    f.seek(0, 2)
    length = f.tell()
    f.seek(0)
    return length

def __write_header(out, header, suffix=False, head_type=0, code_len=0, before_header=False):
    header_base = 0x1000 if head_type else 0
    header_base = (header_base + code_len) if before_header else header_base
    offset = (0xC000 + header_base) if suffix else header_base
    out.seek(offset)
    out.write(header)

def __write_header_hash(out, suffix=False, head_type=0, sm=False, code_len=0, before_header=False):
    header_base = 0x1000 if head_type else 0
    offset = header_base if (suffix or not before_header) else header_base + code_len
    out.seek(offset)
    header = out.read(0x560)
    if sm == False:
        out.write(cal_bin_hash(header))
    else:
        out.write(sm3_cal(header))

def __write_raw_img(out, raw, suffix=False, before_header=False):
    offset = 0 if suffix or before_header else 0x2000
    out.seek(offset)
    with open(raw, 'rb') as raw_file:
        for byte_block in iter(lambda: raw_file.read(4096), b""):
            out.write(byte_block)

def __add_in_tail(offset, inFile, outFile, header, length):
    outFile.seek(offset, 1)
    outFile.write(header)
    outFile.write(inFile.read())
    outFile.seek(-(length + 16), 1)

def __construct_cms_header(tag, length):
    s = struct.Struct('12sI')
    if (len(tag) > 11):
        raise RuntimeError('name too long')
    value = (tag.encode(), length)
    header = s.pack(*value)
    return header

def __write_cms(out, cms, aligned_bytes):
    with open(cms, 'rb') as cf:
        length = __get_filelen(cf)
        header = __construct_cms_header('cms', length)
        __add_in_tail(aligned_bytes, cf, out, header, length)

def __write_ini(out, ini):
    with open(ini, 'rb') as ifile:
        length = __get_filelen(ifile)
        header = __construct_cms_header('ini', length)
        __add_in_tail(16 * 1024, ifile, out, header, length)

def __write_crl(out, crl):
    with open(crl, 'rb') as af:
        length = __get_filelen(af)
        header = __construct_cms_header('crl', length)
        __add_in_tail(2 * 1024, af, out, header, length)

def __write_single_header(args, out, hash_buf, code_len, head_type=0):
    if sys.version > '3':
        N_buf, E_buf = int(0).to_bytes(512, 'big'), int(0).to_bytes(512, 'big')
    else:
        N_buf, E_buf = to_bytes(0, 512), to_bytes(0, 512)

    before_header = True if (args.position == 'before_header') else False
    large_packet = True if (args.pkt_type == 'large_pkt') else False
    header = __construct_header(N_buf, E_buf, hash_buf, code_len, args.S, head_type, args.ver, args.nvcnt,
                                args.tag, args.certtype, before_header, large_packet, False, False, bcm=False,
                                gcm = False, gm = False)
    __write_header(out, header, args.S, head_type, code_len, before_header)

def write_header_huawei(args, out, hash_buf, code_len):
    __write_single_header(args, out, hash_buf, code_len)

def write_image(args, out):
    __write_raw_img(out, args.raw, args.S, False)
    __write_header_hash(out, args.S, 0, False)

def write_cms(args, out, code_len):
    if args.addcms:
        out.seek(code_len + 0x2000)
        if args.position == 'before_header':
            __write_cms(out, args.cms, 0)
        else:
            __write_cms(out, args.cms, 32 - code_len % 16)
        __write_ini(out, args.ini)
        __write_crl(out, args.crl)

def __add_magic_number_and_file_size(args, out, cms_flag, suffix=False, code_len=0, before_header=False, large_packet=False):
    if cms_flag and suffix:
        raise RuntimeError("Invalid Param: --addcms and -S can't input in the same time.")

    out.seek(0, 2)
    # if before_header img code_len >= 4G, img_len = 0 (not used)
    fileSize = 0 if before_header and (out.tell() >= 0x100000000) else out.tell()
    s = struct.Struct('QI')
    if cms_flag:
        value = (0xABCD1234AA55AA55, fileSize)
    else:
        value = (0x0, fileSize)
    stream = s.pack(*value)
    # print(binascii.hexlify(stream))
    offset = code_len + 0x580 if before_header else 0x580
    out.seek(offset, 0)
    out.write(stream)

    # Write additional nvcnt to head
    # nvcnt_offset : 0x0x590
    # [
    #     U32 nvcnt_magic : 0x5A5AA5A5
    #     U32 nvcnt
    # ] nvcnt_s
    if args.nvcnt:
        s = struct.Struct('II')
        nvcnt_magic = 0x5A5AA5A5
        pack_list = (nvcnt_magic, int(args.nvcnt))
        nvcnt_s = s.pack(*pack_list)
        nvcnt_offset = code_len + 0x590 if before_header else 0x590
        out.seek(nvcnt_offset)
        out.write(nvcnt_s)

    if before_header:
        offset = code_len + 0x4E0 if before_header else 0x4E0
        out.seek(offset, 0)
        out.write(code_len.to_bytes(8, 'little'))


def write_extern(args, out, list):
    before_header = True if (args.position == 'before_header') else False
    code_len = list[0] if before_header else 0
    if before_header:
        __write_header_hash(out, args.S, 0, args.sm, code_len, before_header)
    __add_magic_number_and_file_size(args, out, args.addcms, False, code_len, before_header)
    return

def write_hash_tree(args, out, code_len):
    hash_tree_offset = code_len + 0x20000 - 0x100   # 128K
    out.seek(hash_tree_offset)
    hash_tree_path = os.path.join(os.path.dirname(args.raw), "hashtree")
    with open(hash_tree_path, "rb") as hash_tree_file:
        hash_tree_content = hash_tree_file.read()
        out.write(hash_tree_content)

def write_header_huawei_address(args, out, code_len):
    partition_size = int(args.partition_size) * 1024 * 1024
    if args.pkt_type == 'large_pkt':
        header_huawei_address_offset = partition_size - 0xC
        out.seek(header_huawei_address_offset)
        out.write(code_len.to_bytes(8, 'little'))
    else:
        header_huawei_address_offset = partition_size - 0x8
        out.seek(header_huawei_address_offset)
        out.write(code_len.to_bytes(4, 'little'))

def write_version(args, out, code_len):
    partition_size = int(args.partition_size) * 1024 * 1024
    version_offset = partition_size - 0x4
    out.seek(version_offset)
    if args.pkt_type == 'large_pkt':
        version = int(1279739216).to_bytes(4, 'little')      # magic LGEP(large packet)
    else:
        version = int(0).to_bytes(4, 'little')
    out.write(version)