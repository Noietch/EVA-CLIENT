# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

import time


def precise_sleep(seconds: float, spin_threshold: float = 0.002, sleep_margin: float = 0.001):
    """
    Wait for `seconds` with better precision than time.sleep alone at the expense of more CPU usage.

    Parameters:
      - seconds: duration to wait
      - spin_threshold: if remaining <= spin_threshold -> spin; otherwise sleep (seconds). Default 2ms
      - sleep_margin: when sleeping leave this much time before deadline to avoid oversleep. Default 1ms

    Note:
        The default parameters are chosen to prioritize timing accuracy over CPU usage for high-frequency
        use cases like 200 Hz (5ms intervals). For lower frequencies, you may want to increase
        spin_threshold (e.g., 10ms for 30 FPS) for better CPU efficiency.
    """
    if seconds <= 0:
        return

    # Use hybrid sleep+spin approach for all platforms to ensure precise timing
    # On macOS and Windows the scheduler / sleep granularity can make short sleeps inaccurate.
    # On Linux, time.sleep can also have jitter for very short intervals (< 10ms).
    # Instead of burning CPU for the whole duration, sleep for most of the time and
    # spin for the final few milliseconds to achieve good accuracy with much lower CPU usage.
    end_time = time.perf_counter() + seconds
    while True:
        remaining = end_time - time.perf_counter()
        if remaining <= 0:
            break
        # If there's more than a couple milliseconds left, sleep most
        # of the remaining time and leave a small margin for the final spin.
        if remaining > spin_threshold:
            # Sleep but avoid sleeping past the end by leaving a small margin.
            time.sleep(max(remaining - sleep_margin, 0))
        else:
            # Final short spin to hit precise timing without long sleeps.
            pass
