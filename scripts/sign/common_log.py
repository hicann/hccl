#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#**************************************************************
#
#          版权所有 (C), 2001-2011,华为技术有限公司
#
#**************************************************************
#文件名    : ci_log.py
#版本号    : 初稿
#生成日期  : 2012年7月25日
#功能描述  : 持续集成LOG模块
#使用方法  : import ci_log as cilog
#            cilog.method
#输入参数  : 参数1:41位change-id
#          : 参数2: 组件名称
#返回值    : 0:提交都通过；1:提交未都通过
#修改历史  :
#1.日期    : 2012年7月25日
#  修改内容: 创建文件

import time
import inspect
import logging
from functools import partial

logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(pathname)s] [line:%(lineno)d] %(message)s',
                    level=logging.INFO)

THIS_FILE_NAME = __file__

def cilog_get_timestamp():
    """
    功能描述：获取本地时间，并以字符串的方式返回
    参数：NA
    返回值：指定时间格式的字符串
    异常描述：
    修改记录：1.日期    : 2012年7月25日
                修改内容: 创建函数
    """
    
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
         
def cilog_print_element(cilog_element):
    """
    功能描述：以指定格式输出
    参数：需要显示的字符串
    返回值：
    异常描述：
    修改记录：1.日期    : 2012年7月25日
                修改内容: 创建函数
    """
    print("["+cilog_element+"]", end=' ')
    return
                
def cilog_logmsg(log_level, filename, line_no, log_msg, *log_paras):
    log_timestamp = cilog_get_timestamp()
    
    cilog_print_element(log_timestamp)
    cilog_print_element(log_level)
    cilog_print_element(filename)
    cilog_print_element(str(line_no))
        
    print(log_msg % log_paras[0])
    
    return

def cilog_error(filename, log_msg, *log_paras):
    """
    功能描述：ERROR级别的log打印
    参数：1.filename: 调用的文件名称
          2.log_msg: 字符串
          3.*log_paras: 可变参数
    返回值：
    异常描述：
    修改记录：1.日期    : 2012年7月25日
                修改内容: 创建函数
    """
    line_no = inspect.currentframe().f_back.f_lineno
    cilog_logmsg("ERROR", filename, line_no, log_msg, log_paras)
    return
    
def cilog_warning(filename, log_msg, *log_paras):
    """
    功能描述：WARNING级别的log打印
    参数：1.filename: 调用的文件名称
          2.log_msg: 字符串
          3.*log_paras: 可变参数
    返回值：
    异常描述：
    修改记录：1.日期    : 2012年7月25日
                修改内容: 创建函数
    """
    line_no = inspect.currentframe().f_back.f_lineno
    cilog_logmsg("WARNING", filename, line_no, log_msg, log_paras)
    return
    
def cilog_info(filename, log_msg, *log_paras):
    """
    功能描述：INFO级别的log打印
    参数：1.filename: 调用的文件名称
          2.log_msg: 字符串
          3.*log_paras: 可变参数
    返回值：
    异常描述：
    修改记录：1.日期    : 2012年7月25日
                修改内容: 创建函数
    """
    line_no = inspect.currentframe().f_back.f_lineno
    cilog_logmsg("INFO", filename, line_no, log_msg, log_paras)
    return


def auto_fill_filename(func):
    """自动填充文件名参数。"""
    filename = inspect.stack()[1].filename
    return partial(func, filename)


if __name__ == "__main__":
    i = 0
    while i<3:
        cilog_error(THIS_FILE_NAME, "%s say %s %d times", "I", "Hello world", i)
        i+=1
