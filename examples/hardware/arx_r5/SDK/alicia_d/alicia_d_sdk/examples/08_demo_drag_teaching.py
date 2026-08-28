#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""Drag Teaching Demo

This demo demonstrates:
1. Manual mode - Record key waypoints by manually dragging the robot
2. Auto mode - Continuously record trajectory while dragging the robot
3. Replay mode - Replay previously recorded motions

Usage:
python 08_demo_drag_teaching.py

The program will prompt you to select one of three modes:
1. Manual recording mode - Record key waypoints by dragging
2. Auto recording mode - Continuously record trajectory while dragging
3. Replay mode - Replay existing recorded motions

# Get help
python 08_demo_drag_teaching.py --help
"""

import os
import json
import time
import argparse
import threading
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime

from alicia_d_sdk import create_robot
from robocore.utils.beauty_logger import beauty_print

from alicia_d_sdk.execution.drag_teaching import DragTeaching
from alicia_d_sdk.execution.drag_teaching import print_available_motions, list_available_motions


def select_mode():
    """Let user select operation mode.
    
    :return: Selected mode string ('1', '2', or '3')
    """
    beauty_print("Please select operation mode:", type="module", centered=False)
    beauty_print("  1. Manual recording mode - Record key waypoints by dragging robot arm")
    beauty_print("  2. Auto recording mode - Continuously record trajectory while dragging")
    beauty_print("  3. Replay mode - Replay existing recorded motions")
    beauty_print("")

    while True:
        choice = input("Enter option (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        beauty_print("Invalid option, please enter 1, 2, or 3", type="warning")


def prompt_motion_name(mode: str) -> str:
    """Prompt user for motion name.
    
    :param mode: Recording mode ('manual' or 'auto')
    :return: Motion name string
    """
    beauty_print(f"\nPlease enter motion name for {mode} mode:", type="module", centered=False)
    beauty_print("  - This will be used as the folder name to save the motion")
    beauty_print("  - Example: my_demo, key_points, etc.")
    beauty_print("")
    
    while True:
        motion_name = input("Motion name: ").strip()
        if motion_name:
            # Validate motion name (no path separators)
            if os.sep in motion_name or '/' in motion_name or '\\' in motion_name:
                beauty_print("Motion name cannot contain path separators", type="error")
                continue
            return motion_name
        beauty_print("Motion name cannot be empty", type="warning")


def select_motion_to_replay() -> Optional[str]:
    """Let user select a motion to replay from available motions.
    
    :return: Selected motion name or None if cancelled
    """
    available_motions = list_available_motions()
    
    if not available_motions:
        beauty_print("No recorded motions found", type="warning")
        beauty_print("Please record a motion using manual or auto mode first", type="info")
        return None
    
    beauty_print(f"\nAvailable motions ({len(available_motions)}):", type="module", centered=False)
    
    for i, motion in enumerate(available_motions, 1):
        motion_dir = os.path.join("example_motions", motion)
        meta_path = os.path.join(motion_dir, "meta.json")
        
        info = f"  {i}. {motion}"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                mode = meta.get('mode', 'unknown')
                created = meta.get('created_at', 'unknown')
                count = meta.get('count', 0)
                info += f" (mode: {mode}, points: {count}, created: {created})"
            except:
                pass
        print(info)  # Use print for list items to keep formatting simple
    
    beauty_print("")
    
    while True:
        choice = input(f"Enter option (1-{len(available_motions)}) or 'q' to cancel: ").strip()
        if choice.lower() == 'q':
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available_motions):
                return available_motions[idx]
            beauty_print(f"Invalid option, please enter 1-{len(available_motions)} or 'q'", type="warning")
        except ValueError:
            beauty_print("Invalid input, please enter a number or 'q'", type="warning")


def main(args):
    """Demonstrate drag teaching functionality."""
    
    # If the user requests to list motions, display and exit
    if args.list_motions:
        print_available_motions()
        return
    
    beauty_print("Drag Teaching Demo", type="module")
    
    # Initialize robot connection
    robot = create_robot(port=args.port)
    
    try:
        # Create a temporary args object for DragTeaching
        class TempArgs:
            def __init__(self, base_args):
                self.port = base_args.port
                self.speed_deg_s = base_args.speed_deg_s
                self.sample_hz = base_args.sample_hz
                self.mode = None
                self.save_motion = None
        
        temp_args = TempArgs(args)
        
        # Check if mode and save_motion are provided via command line (backward compatibility)
        if args.mode and args.save_motion:
            # Use command line arguments
            temp_args.mode = args.mode
            temp_args.save_motion = args.save_motion
            
            # Validate parameters
            if temp_args.mode in ['manual', 'auto'] and not temp_args.save_motion:
                beauty_print(f"{temp_args.mode} mode requires motion name", type="error")
                robot.disconnect()
                return
            
            if temp_args.mode == 'replay_only' and not temp_args.save_motion:
                beauty_print("Replay mode requires motion name", type="error")
                beauty_print("Use --list-motions to view available motions", type="info")
                robot.disconnect()
                return
        else:
            # Interactive mode: select mode and get motion name
            mode_choice = select_mode()
            
            if mode_choice == '1':
                # Manual recording mode
                temp_args.mode = 'manual'
                temp_args.save_motion = prompt_motion_name('manual')
            elif mode_choice == '2':
                # Auto recording mode
                temp_args.mode = 'auto'
                temp_args.save_motion = prompt_motion_name('auto')
            else:  # mode_choice == '3'
                # Replay mode
                temp_args.mode = 'replay_only'
                motion_name = select_motion_to_replay()
                if motion_name is None:
                    beauty_print("Replay cancelled", type="warning")
                    robot.disconnect()
                    return
                temp_args.save_motion = motion_name
        
        # Create DragTeaching instance and run
        drag_teaching = DragTeaching(temp_args, robot)
        drag_teaching.run()
        
    except KeyboardInterrupt:
        beauty_print("\nProgram interrupted by user", type="warning")
    except Exception as e:
        beauty_print(f"Error: {e}", type="error")
        import traceback
        traceback.print_exc()
    finally:
        robot.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drag Teaching Demo", 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Robot configuration
    parser.add_argument('--port', type=str, default="", help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument('--speed_deg_s', type=int, default=15, help="Joint motion speed (degrees/second, default: 15, range: 10-80 deg/s)")
    parser.add_argument('--sample_hz', type=float, default=200.0, help="Sampling frequency (Hz, default: 200.0, only for auto mode)")

    # Legacy arguments (kept for backward compatibility, but not used in interactive mode)
    parser.add_argument('--mode', choices=['manual', 'auto', 'replay_only'], default=None,
                       help="Mode: manual (key waypoints) or auto (continuous) or replay_only (replay only). "
                            "If not specified, interactive mode will be used.")
    parser.add_argument('--save-motion', default=None, help="Motion name (for recording: new name; for replay: existing name). "
                                                           "If not specified, interactive mode will prompt for it.")
    parser.add_argument('--list-motions', action='store_true', help="List all available motions and exit")
    
    args = parser.parse_args()
    main(args)
