# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

from .beauty_logger import *
from datetime import datetime
from .beauty_logger import hex_print


_log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = BeautyLogger(log_dir="./logs", log_name=f"alicia_d_sdk_{_log_timestamp}.log", verbose=True, min_level=LogLevel.INFO)
