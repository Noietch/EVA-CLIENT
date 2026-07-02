# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

import os
from typing import List

# 定义日志级别常量
class LogLevel:
    """日志级别定义"""
    DEBUG = 0
    INFO = 1
    MODULE = 2
    WARNING = 3
    ERROR = 4
    SUCCESS = 5


class BeautyLogger:
    """
    Lightweight logger for Alicia-D-SDK package.
    """

    def __init__(self, log_dir: str, log_name: str = 'rofunc.log', verbose: bool = True, min_level: int = LogLevel.INFO):
        """
        Alicia-D-SDK轻量级日志器

        :param log_dir: 日志文件保存路径
        :param log_name: 日志文件名
        :param verbose: 是否在控制台打印日志
        :param min_level: 最小日志级别
        """
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(self.log_dir, self.log_name)
        self.verbose = verbose
        self.min_level = min_level

        os.makedirs(self.log_dir, exist_ok=True)
        
    def _write_log(self, content, type):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(" Alicia-D-SDK:{}] {}\n".format(type.upper(), content))

    def _should_print(self, level: int) -> bool:
        """
        检查是否应该打印日志

        :param level: 要检查的日志级别
        :return: 是否应该打印
        """
        return self.verbose and level >= self.min_level

    def set_min_level(self, level: int):
        """
        设置最小日志级别

        :param level: 最小日志级别
        """
        if level < LogLevel.DEBUG or level > LogLevel.SUCCESS:
            raise ValueError("Invalid log level. Must be between LogLevel.DEBUG and LogLevel.SUCCESS")
        self.min_level = level

    def warning(self, content, local_verbose=True):
        """
        打印警告消息

        :param content: 警告消息内容
        :param local_verbose: 是否在控制台打印
        """
        if self._should_print(LogLevel.WARNING) and local_verbose:
            beauty_print(content, type="warning")
        self._write_log(content, type="warning")

    def module(self, content, local_verbose=True):
        """
        打印模块消息

        :param content: 模块消息内容
        :param local_verbose: 是否在控制台打印
        """
        if self._should_print(LogLevel.MODULE) and local_verbose:
            beauty_print(content, type="module")
        self._write_log(content, type="module")

    def info(self, content, local_verbose=True):
        """
        打印信息消息

        :param content: 信息消息内容
        :param local_verbose: 是否在控制台打印
        """
        if self._should_print(LogLevel.INFO) and local_verbose:
            beauty_print(content, type="info")
        self._write_log(content, type="info")

    def debug(self, content, local_verbose=True):
        """
        打印调试消息
        
        :param content: 调试消息内容
        :param local_verbose: 是否在控制台打印
        """
        if self._should_print(LogLevel.DEBUG) and local_verbose:
            beauty_print(content, type="debug")
        self._write_log(content, type="debug")

    def error(self, content, local_verbose=True):
        """
        打印错误消息
        
        :param content: 错误消息内容
        :param local_verbose: 是否在控制台打印
        """
        if self._should_print(LogLevel.ERROR) and local_verbose:
            beauty_print(content, type="error")
        self._write_log(content, type="error")
        raise Exception(content)

    def success(self, content, local_verbose=True):
        """
        打印成功消息
        
        :param content: 成功消息内容
        :param local_verbose: 是否在控制台打印
        """
        if self._should_print(LogLevel.SUCCESS) and local_verbose:
            beauty_print(content, type="success")
        self._write_log(content, type="success")


def beauty_print(content, type: str = None):
    """
    使用不同颜色打印内容

    :param content: 要打印的内容
    :param type: 支持 "warning", "module", "info", "error", "debug", "success"
    """
    if type is None:
        type = "info"
    if type == "warning":
        print("\033[1;37m [Alicia-D-SDK:WARNING] {}\033[0m".format(content))  # For warning (gray)
    elif type == "module":
        print("\033[1;33m [Alicia-D-SDK:MODULE] {}\033[0m".format(content))  # For a new module (light yellow)
    elif type == "info":
        print("\033[1;35m [Alicia-D-SDK:INFO] {}\033[0m".format(content))  # For info (light purple)
    elif type == "debug":
        print("\033[1;34m [Alicia-D-SDK:DEBUG] {}\033[0m".format(content))  # For debug (light blue)
    elif type == "error":
        print("\033[1;31m [Alicia-D-SDK:ERROR] {}\033[0m".format(content))  # For error (red)
    elif type == "success":
        print("\033[1;32m [Alicia-D-SDK:SUCCESS] {}\033[0m".format(content))  # For success (green)
    else:
        raise ValueError("Invalid level")


def hex_print(logger: BeautyLogger, title: str, data: List[int]):
    """
    print the data in hex format
    :param logger: the logger
    :param title: the title of the data
    :param data: the data to print
    :return: None
    """
    hex_buf = ' '.join(f"{b:02X}" for b in data)
    logger.info(f"{title}: {hex_buf}")
