#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
#
#
#**************************************************************
# 文件名    ：add_sign_header_cann.py
# 版本号    ：初稿
# 生成日期  ：2025年11月25日
# 功能描述  ：根据bios_check_cfg.xml配置文件，根据其中配置属性，对各文件进行制作cms签名并绑定
# 使用方法  ：python add_bios_header.py $(DEVICE_RELEASE_DIR) $(DAVINCI_TOPDIR) $(PRODUCT_NAME) $(CHIP_NAME) $(signature_tag)
# 输入参数  ：DEVICE_RELEASE_DIR：待签名文件的根目录
#            DAVINCI_TOPDIR：工程根路径
#            PRODUCT_NAME：待扫描的产品名
#            CHIP_NAME：芯片名称
#            signature_tag：是否需要数字签名
#            蓝区签名步骤 1、加esbc头; 2、生成ini文件; 3、进行签名（参数控制，暂不启动）; 4、签名结果写入文件头  
# 返回值    ：0:成功，-1:失败
# 修改历史  ：
# 日期    ：2025年11月25日
# 修改内容  ：创建文件
"""
from collections import namedtuple
from typing import Dict, Iterator, List, Tuple
import os
import sys
import shutil
import subprocess
import argparse
import xml.etree.ElementTree as ET
import common_log as COMM_LOG

THIS_FILE_NAME = __file__
THIS_FILE_PATH = os.path.realpath(THIS_FILE_NAME)
MY_PATH = os.path.dirname(THIS_FILE_PATH)

PATH_SEPARATOR = "/"

class AddHeaderConfig:
    """加头工具命令行配置类"""

    def __init__(self, inputfile, output, version, fw_version, inputtype, tag,
                 rootrsa, subrsa, additional, sign_alg, encrypt_alg,
                 encrypt_type, nvcnt, rsatag, position, image_pack_version,
                 bist_flag):
        self.input = inputfile
        self.output = output
        self.version = version
        self.fw_version = fw_version
        self.rootrsa = rootrsa
        self.subrsa = subrsa
        self.additional = additional
        self.type = inputtype
        self.tag = tag
        self.sign_alg = sign_alg
        self.encrypt_alg = encrypt_alg
        self.encrypt_type = encrypt_type
        self.nvcnt = nvcnt
        self.rsatag = rsatag
        self.position = position
        self.image_pack_version = image_pack_version
        self.bist_flag = bist_flag

# 加nvcnt头配置类
AddNvcntHeaderConfig = \
    namedtuple('AddNvcntHeaderConfig', ['inputfile', 'nvcnt'])

def read_xml(in_path):
    '''
    功能：读取XML
    '''
    tree = ET.ElementTree()
    tree.parse(in_path)
    return tree

def check_config_item(node) -> bool:
    """校验节点必需属性"""
    if "input" not in node.attrib or "output" not in node.attrib:
        COMM_LOG.cilog_error(THIS_FILE_NAME, "bios_check_cfg.xml config format is invalid")
        return False

    if "type" in node.attrib:
        if "cms" in node.attrib["type"].split('/') and "tag" not in node.attrib:
            COMM_LOG.cilog_error(THIS_FILE_NAME,
                                 "when bios_check_cfg.xml has cms type, it must has 'tag' attribute")
            return False

    return True

def parse_item(node):
    tag_type = ""
    if "type" in node.attrib:
        tag_type = node.attrib["type"]
    sign_alg = "PKCSv1.5"
    if "sign_alg" in node.attrib:
        sign_alg = node.attrib["sign_alg"]
    encrypt_alg = ""
    if "encrypt_alg" in node.attrib:
        encrypt_alg = node.attrib["encrypt_alg"]
    encrypt_type = ""
    if "encrypt_type" in node.attrib:
        encrypt_type = node.attrib["encrypt_type"]
    add_para = ""
    if "additional" in node.attrib:
        add_para = node.attrib["additional"]
    add_tag = []
    if "tag" in node.attrib:
        add_tag = node.attrib["tag"]
    nvcnt = ""
    if "nvcnt" in node.attrib:
        nvcnt = node.attrib["nvcnt"]
    rsatag = ""
    if "rsatag" in node.attrib:
        rsatag = node.attrib["rsatag"]
    position = ""
    if "position" in node.attrib:
        position = node.attrib["position"]
    image_pack_version = '1.0'
    if 'image_pack' in node.attrib:
        image_pack_version = node.attrib["image_pack"]
    # parse rsa
    rootrsa = "default_rsa_rootkey"
    if "rootrsa" in node.attrib:
        rootrsa = node.attrib["rootrsa"]
    subrsa = "default_rsa_subkey"
    if "subrsa" in node.attrib:
        subrsa = node.attrib["subrsa"]
    bist_flag = ""
    if "bist_flag" in node.attrib:
        bist_flag = node.attrib["bist_flag"]

    cur_conf = AddHeaderConfig(node.attrib["input"], node.attrib["output"],
                               node.attrib["version"],
                               node.attrib.get("fw_version", ""), tag_type,
                               add_tag, rootrsa, subrsa, add_para, sign_alg,
                               encrypt_alg, encrypt_type, nvcnt, rsatag,
                               position, image_pack_version, bist_flag)
    return cur_conf

def get_item_set(config_file, image_dir, version, chip_name='') -> Tuple[int, Dict, List]:
    """
    功能：解析xml配置文件
    """
    soc_version = chip_name
    item_size_set = {}
    ini_size_set = {}
    tree = read_xml(config_file)
    origin_nodes = tree.findall("item")
    ini_nodes = tree.findall("ini")

    for node in origin_nodes:
        if 'version' not in node.attrib:
            node.attrib['version'] = version
        if not check_config_item(node):
            return -1, None, None, None

    # 排除不存在的文件
    nodes = []
    for node in origin_nodes:
        input_file = os.path.join(image_dir, node.attrib["input"])

        if os.path.exists(input_file):
            nodes.append(node)
        else:
            COMM_LOG.cilog_warning(THIS_FILE_NAME, "Image file:%s not exits!\n\t", input_file)
            continue

    for node in nodes:
        cur_conf = parse_item(node)
        item_size_set[cur_conf.input] = cur_conf

    for ini_node in ini_nodes:
        cur_conf = parse_item(ini_node)
        ini_size_set[cur_conf.input] = cur_conf

    nvcnt_configs = []
    for node in nodes:
        if "nvcnt" in node.attrib:
            inputfile = os.path.join(image_dir, node.attrib["input"])
            nvcnt = node.attrib["nvcnt"]

            config = AddNvcntHeaderConfig(inputfile, nvcnt)
            nvcnt_configs.append(config)

    return 0, item_size_set, ini_size_set, nvcnt_configs

# 生成摘要文件，每个待签名文件生成一个，生成文件相关的参数放在image_info.xml文件中
def build_inifile(item_size_set, image_dir, bios_tool_path,
                  sign_tmp_path, product_delivery_path, add_sign):
    '''
    功能：根据从bios_check_cfg.xml读取的配置，生成ini工具(ini_gen.py)的配置文件，
    然后调用ini工具读取该配置文件生成每个文件对应的ini文件
    输入：item_size_set：待制作ini镜像的清单、image_dir：镜像根路径、bios_tool_path：bios的ini制作工具路径
    返回：-1:失败，0：成功
    '''
    cms_flag = False
    if add_sign == "true":
        inicfg = os.path.join(sign_tmp_path, "image_info.xml")
        with open(inicfg, "w+", encoding='utf-8') as read_cfg:
            read_cfg.write("<image_info>\n")
            for (infile, conf_item) in list(item_size_set.items()):
                inputfile = os.path.join(image_dir, infile)
                relative_path = inputfile.replace(product_delivery_path + PATH_SEPARATOR, "")
                output_path = os.path.dirname(
                    os.path.join(sign_tmp_path, relative_path))
                output_path = os.path.realpath(output_path)
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                if "cms" in conf_item.type.split('/') and conf_item.tag != "" \
                        and os.path.isfile(inputfile):
                    cms_flag = True
                    read_cfg.write('<image path="%s" out="%s" tag="%s" ini_name="%s"/>\n'
                                   % (inputfile, output_path, conf_item.tag, os.path.basename(infile)))
            read_cfg.write("</image_info>\n")
        gen_tool = os.path.join(bios_tool_path, "ini_gen.py")
        cmd = "%s %s -in_xml %s" % (os.environ["HI_PYTHON"], gen_tool, inicfg)

    if add_sign == "true" and cms_flag:
        COMM_LOG.cilog_info(THIS_FILE_NAME, "------------------------------------")
        COMM_LOG.cilog_info(THIS_FILE_NAME, "execute:%s", cmd)
        ret = subprocess.getstatusoutput(cmd)
        if ret[0] != 0:
            COMM_LOG.cilog_error(THIS_FILE_NAME, "build inifile failed!\n\t%s", (ret[1]))
            return -1
    return 0

def build_sign(item_size_set, image_dir, sign_tool_path, sign_tmp_path, davinci_dir, product_delivery_path,
               sign_certificate, chip_name, product_name):
    '''
    功能：制作签名文件
    输入：para1：待签名的镜像清单、para2：镜像根路径、para3：签名工具路径、
    para4：签名临时路径，实际将待签名文件拷贝到该目录下进行签名、 para5: davinci工程路径
    para6: davinci镜像生成路径
    返回：-1:失败，0：成功
    '''
    sign_dict = {}
    sign_dict["cms"] = []
    for (infile, conf_item) in list(item_size_set.items()):
        input_path = os.path.join(image_dir, infile)
        if os.path.exists(input_path):
            cmd = "ls {}".format(input_path)
            ret = subprocess.getstatusoutput(cmd)
            if ret[0] != 0:
                COMM_LOG.cilog_warning(THIS_FILE_NAME, "can not find %s in %s \n\t%s", input_path, image_dir, ret[1])
                continue
        else:
            COMM_LOG.cilog_error(THIS_FILE_NAME, "infile is not exist:%s", input_path)
            return -1

        for sign in conf_item.type.split('/'):
            if sign in sign_dict:
                sign_dict[sign].append(infile)  # 需要签名的文件都写入这个字典中，前面已判断文件是否存在
    # 只制作cms签名
    cmd = ''
    for file in sign_dict["cms"]:
        # 如果没配置sign_alg属性，默认使用RSA_PSS算法签名
        file_with_path = os.path.join(image_dir, file)
        # windows平台调试跟linux平台路径分隔符不一样，实际使用中需要替换 TODO
        relative_path = file_with_path.replace(("{}" + PATH_SEPARATOR).format(product_delivery_path), "")
        # 临时目录下同层架子目录，包含文件名的完整路径
        file_sign_des = os.path.realpath(os.path.join(sign_tmp_path, relative_path))
        # 临时目录的路径，不包含文件名
        sign_path = os.path.dirname(file_sign_des)

        if not os.path.isdir(sign_path):
            os.makedirs(sign_path)
        if os.path.isfile(file_with_path):
            COMM_LOG.cilog_info(THIS_FILE_NAME, "copy %s --> %s", file_with_path, file_sign_des)
            # 待签名文件拷贝到临时路径下
            shutil.copy(file_with_path, file_sign_des)
            if not os.path.isfile(file_sign_des):
                COMM_LOG.cilog_error(THIS_FILE_NAME, "copy %s --> %s fail",
                                     file_with_path, file_sign_des)
                return -1
        else:
            COMM_LOG.cilog_error(THIS_FILE_NAME, "can not find src:%s", file_with_path)
            return -1
        # 临时目录下ini文件完整路径，实际前面生成ini文件时，已经生成到对应的目录下
        file_sign_des = "{}.ini".format(os.path.join(sign_path, os.path.basename(file)))

        # 蓝区签名平台，命令不一样
        if not cmd:
            cmd = "{} {} {}".format(os.environ["HI_PYTHON"],
                                          os.path.join(MY_PATH, "community_sign_build.py"), file_sign_des)
        else:
            cmd = '{} {}'.format(cmd, file_sign_des)

        # 调用签名平台分别对文件进行CMS签名
        COMM_LOG.cilog_info(THIS_FILE_NAME, "------------------------------------")
        COMM_LOG.cilog_info(THIS_FILE_NAME, "execute:%s", cmd)
        # 签名后会在ini文件通目录下生成p7s文件，比如a.ini=>a.ini.p7s
        ret = subprocess.getstatusoutput(cmd)
        if ret[0] != 1:
            COMM_LOG.cilog_error(THIS_FILE_NAME, "make %s sign failed!\n\t%s", sign, ret[1])
            return -1
        COMM_LOG.cilog_info(THIS_FILE_NAME, "%s", ret[1])

    return 0

# 添加esbc头，支持在不签名的情况下添加头，包含版本、tag、chip、nvcnt信息
def add_bios_esbc_header(davinci_dir, item_size_set, image_dir, chip_name):
    '''
    功能：需要做RSA签名绑定的镜像绑定esbc二级头
    输入：para1：davinci工程路径、 para2: 待签名的镜像清单、 para3: 镜像根路径
    返回：-1:失败，0：成功
    '''
    bios_esbc_header_tool_path = os.path.join(
        davinci_dir, "scripts", "signtool", "esbc_header")
    # 检查加头工具目录是否存在
    if not os.path.exists(bios_esbc_header_tool_path):
        COMM_LOG.cilog_error(THIS_FILE_NAME, "bios esbc tool dir not exits")
        return -1

    for (input_filename, conf_item) in list(item_size_set.items()):
        input_file = os.path.join(image_dir, input_filename)

        if conf_item.nvcnt:
            cmd = f'{os.environ["HI_PYTHON"]} {os.path.join(bios_esbc_header_tool_path, "esbc_header.py")}'
            # 用esbc_header.py工具脚本添加esbc头
            cmd += f" -raw_img {input_file} -out_img {input_file} -platform {chip_name}"
            cmd += f" -version {conf_item.version} -nvcnt {conf_item.nvcnt} -tag {conf_item.tag}"

            COMM_LOG.cilog_info(THIS_FILE_NAME, "------------------------------------")
            COMM_LOG.cilog_info(THIS_FILE_NAME, "execute:%s", cmd)
            ret = subprocess.getstatusoutput(cmd)
            if ret[0] != 0:
                COMM_LOG.cilog_error(THIS_FILE_NAME, "add %s esbc header failed!\n\t%s", input_file, ret[1])
                return -1
        else:
            COMM_LOG.cilog_info(THIS_FILE_NAME, "%s don't need add esbc head!\n", input_file)
    return 0

# 1 生成ini文件、2 添加esbc头、 3 执行签名、4 合入文件头
def add_bios_header(item_size_set, image_dir, bios_tool_path, sign_tool_path, davinci_dir, add_sign,
                    product_name,
                    chip_name, sign_certificate, sign_len='', rootfs_size=0):
    """
    功能：生成每个镜像的签名并绑定
    输入：para1：待签名的镜像清单
          para2：镜像根路径
          para3：bios的ini制作工具路径
          para4：签名工具路径
          para5：davinci工程路径
    返回：-1:失败，0：成功
    """
    # 分别绑定镜像文件，此处分3种流程
    # 1、add_sign不为true时,不需要进行签名,此时只需要绑定BIOS字节头
    # 2、conf_item.type为cms时：需要绑定镜像文件，ini文件、cms签名及证书、rsa签名及证书、rsa根证书

    # 当需要签名时,调用build_inifile函数生成ini文件
    sign_tmp_path = os.path.join(davinci_dir, "delivery", "sign_tmp")
    product_delivery_path = os.path.join(davinci_dir, "delivery")
    if not os.path.isdir(sign_tmp_path):
        os.makedirs(sign_tmp_path)

    ret_code = add_bios_esbc_header(davinci_dir, item_size_set, image_dir, chip_name)
    if ret_code != 0:
        return ret_code

    ret_code = build_inifile(
        item_size_set, image_dir, bios_tool_path, sign_tmp_path, product_delivery_path, add_sign)
    if ret_code != 0:
        return ret_code

    if add_sign == "true":
        ret_code = build_sign(item_size_set, image_dir, sign_tool_path, sign_tmp_path,
                              davinci_dir, product_delivery_path, sign_certificate, chip_name, product_name)
        if ret_code != 0:
            return ret_code

    for (input, conf_item) in list(item_size_set.items()):
        input_file = os.path.join(image_dir, input)
        relative_path = input_file.replace("{}" + PATH_SEPARATOR.format(product_delivery_path), "")
        # 签名文件及ini文件存放目录
        sign_file = os.path.realpath(os.path.join(sign_tmp_path, relative_path))
        sign_path = os.path.dirname(sign_file)

        # 将CRL文件
        crl_file = os.path.join(sign_tool_path, "signtool", "signature")

        cmd = "{} {}".format(os.environ["HI_PYTHON"], os.path.join(bios_tool_path, "image_pack.py"))
        add_cmd = conf_item.additional
        # 镜像绑定cms签名,用image_pack.py工具脚本绑定cms签名信息
        if add_sign != "true" or conf_item.type == '':
            cmd = cmd + " -raw_img %s -out_img %s -platform %s -version %s -nvcnt %s -tag %s" \
                  % (input_file, input_file, chip_name, conf_item.version, conf_item.nvcnt, conf_item.tag)
            if conf_item.position != "":
                cmd = cmd + " -position %s" % (conf_item.position)
        elif add_sign == "true" and conf_item.type != "":
            # 原代码支持/分割多种签名方式，实际只能一种，暂时保持不变，后续统一黄区代码时再优化
            for sign in conf_item.type.split('/'):
                cmd = cmd + " -raw_img %s -out_img %s -platform %s -version %s -nvcnt %s -tag %s %s" \
                      % (input_file, input_file, chip_name, conf_item.version, conf_item.nvcnt, conf_item.tag, add_cmd)

                if sign == "cms":
                    # 临时目录下的ini文件
                    ini_file = os.path.join(sign_path, os.path.basename(input))
                    cmd = cmd + " -cms %s.ini.p7s -ini %s.ini -crl %s\\SWSCRL.crl --addcms" \
                          % (ini_file, ini_file, crl_file)
                    if conf_item.position != "":
                        cmd = cmd + " -position %s" % (conf_item.position)

                if sign_len == '8bytes':
                    cmd = cmd + ' -pkt_type large_pkt'
        else:
            COMM_LOG.cilog_error(THIS_FILE_NAME,
                                 "bios_check_cfg.xml config format is invalid, %s is not correct!,please check!",
                                 input_file)
            return -1
        COMM_LOG.cilog_info(THIS_FILE_NAME, "------------------------------------")
        COMM_LOG.cilog_info(THIS_FILE_NAME, "execute:%s", cmd)
        ret = subprocess.getstatusoutput(cmd)
        if ret[0] != 0:
            COMM_LOG.cilog_error(THIS_FILE_NAME, "add %s header failed!\n\t%s", input_file, ret[1])
            return -1

    # 删除中间残留文件，防止打包到run包中
    if os.path.isdir(sign_tmp_path):
        shutil.rmtree(sign_tmp_path)

    COMM_LOG.cilog_info(THIS_FILE_NAME, "add header to all bios image success!")
    return 0

def check_params(params):
    """检查参数。"""
    # 检查BIOS配置文件是否存在
    if not os.path.exists(params['config_file']):
        COMM_LOG.cilog_error(THIS_FILE_NAME, "bios image header config file not exits:%s", params['config_file'])
        return -1
    # 检查BIOS工具是否存在
    if not os.path.exists(params['bios_tool_path']):
        COMM_LOG.cilog_error(THIS_FILE_NAME, "biostool dir not exits")
        return -1
    return 0

def _define_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', help='device release dir')
    parser.add_argument('davinci_dir', help='davinci source root')
    parser.add_argument('product_name', help='product name')
    parser.add_argument('chip_name', help='chip name')
    parser.add_argument('signature_tag', help='sianature tag', default='false')
    parser.add_argument('--sign_certificate', help='sign certificate', default='')
    parser.add_argument('--bios_check_cfg', help='default bios_check_cfg.xml', default='bios_check_cfg.xml')
    parser.add_argument('--encrypt_type', help='encrypt type,example:kms', default='')
    parser.add_argument('--sign_len', help='sign_len, example:8bytes', default='', nargs='?', const='')
    parser.add_argument('--pkg_type', help='pkg_type, example:asan', default='', nargs='?', const='')
    parser.add_argument('--equipment', help='equipment', default='false', nargs='?', const='')
    # 签名版本信息，编译传入
    parser.add_argument('--version', help='version', default='false')
    # 签名类型，out表示蓝区社区签名，in表示华为签名
    parser.add_argument('--sign_type', help='sign type', default='out')
    return parser


def setenv():
    """设置环境变量。"""
    if 'HI_PYTHON' not in os.environ:
        os.environ['HI_PYTHON'] = os.path.basename(sys.executable)


def main(argv=None):
    """
    主函数，检查输入参数及环境检查,并调用功能函数
    """
    parser = _define_parser()
    args = parser.parse_args()
    image_dir = args.image_dir
    davinci_dir = args.davinci_dir
    product_name = args.product_name
    chip_name = args.chip_name
    add_sign = args.signature_tag
    sign_certificate = args.sign_certificate
    bios_check_cfg = args.bios_check_cfg if args.bios_check_cfg else 'bios_check_cfg.xml'
    sign_len = args.sign_len
    version = args.version

    # 需要签名的文件清单
    config_file = os.path.join(davinci_dir, bios_check_cfg)
    COMM_LOG.cilog_info(THIS_FILE_NAME, "config_file=" + config_file)

    bios_tool_path = os.path.join(
        davinci_dir, "scripts", "signtool", "image_pack")

    sgn_tool_path = os.path.join(
        davinci_dir, "scripts","sign")

    ret_code = check_params({
        'config_file': config_file,
        'bios_tool_path': bios_tool_path,
        'image_dir': image_dir,
        'sgn_tool_path': sgn_tool_path,
        'add_sign': add_sign,
    })

    if ret_code != 0:
        return ret_code

    setenv()

    # 读取并解析BIOS的签名、绑定配置文件，并将解析的时候存储在item_size_set变量中
    ret_code, item_size_set, ini_size_set, nvcnt_configs = get_item_set(config_file, image_dir, version, chip_name)
    if ret_code != 0:
        return ret_code

    # 调用签名插件对需要签名的镜像进行签名，并绑定镜像文件
    ret_code = add_bios_header(item_size_set, image_dir, bios_tool_path,
                               sgn_tool_path, davinci_dir, add_sign, product_name, chip_name, sign_certificate,
                               sign_len)
    return ret_code

if __name__ == "__main__":
    sys.exit(main())
