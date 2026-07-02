# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
Benchmark: Read joint angles frequency

Features:
- Benchmark API call frequency (memory read speed)
- Benchmark actual data update frequency (effective sampling rate)
"""

import time
import argparse
import alicia_d_sdk
import numpy as np

def main(args):
    """Benchmark joint reading frequency.
    
    :param args: Command line arguments
    """
    # Initialize robot instance
    robot = alicia_d_sdk.create_robot(
        port=args.port,
        gripper_type=args.gripper_type
    )
    
    try:


        print(f"\nBenchmark started (Duration: {args.duration}s)")
        print("-" * 50)
        print(f"Default update interval: {robot.servo_driver.thread_update_interval:.4f} s")
        
        # If user wants to maximize performance, we can try to lower the interval
        if args.fast:
            print("Enabling fast mode: setting update interval to 0.001s")
            robot.servo_driver.thread_update_interval = 0.001
            
        print("Collecting data...")
        
        count_api_calls = 0
        count_data_updates = 0
        last_joints = None
        
        # Pre-allocate to avoid GC impact during loop if possible, 
        # but Python creates new lists for get_joints anyway.
        
        start_time = time.perf_counter()
        end_time = start_time + args.duration
        
        while time.perf_counter() < end_time:
            # 1. API Call
            joints = robot.get_robot_state("joint")
            count_api_calls += 1
            
            # 2. Check for data update
            if joints is not None:
                if last_joints is None:
                    last_joints = joints
                    count_data_updates += 1
                else:
                    # Check if data changed. 
                    # Even tiny changes (sensor noise) indicate a new frame was parsed.
                    if joints != last_joints:
                        count_data_updates += 1
                        last_joints = joints
        
        actual_duration = time.perf_counter() - start_time
        
        print("-" * 50)
        print(f"Results ({actual_duration:.2f}s):")
        print(f"1. API Read Frequency:  {count_api_calls / actual_duration:10.2f} Hz")
        print(f"   (Speed of reading cached data from Python memory)")
        
        print(f"2. Data Update Frequency: {count_data_updates / actual_duration:9.2f} Hz")
        print(f"   (Actual rate of new data arriving from robot)")
        
        # Get Serial Stats
        if hasattr(robot.servo_driver.serial_comm, 'get_processing_stats'):
            stats = robot.servo_driver.serial_comm.get_processing_stats()
            print(f"\nSerial Stats:")
            print(f"  - Frames Processed: {stats.get('frames_processed', 'N/A')}")
            print(f"  - Frames Dropped:   {stats.get('frames_dropped', 'N/A')} (Check for CRC errors)")
            print(f"  - Buffer Size:      {stats.get('buffer_size', 'N/A')}")
        
        print("-" * 50)
        print("Note: If Data Frequency is ~40Hz, it might be the default robot reporting rate.")
        print("      If Frames Dropped is high, check baudrate or cable.")
        print("      Try running with --fast to minimize SDK sleep time.")

        if count_data_updates == 1:
            print("\nWarning: Data did not change during the test.")
            print("The robot might be perfectly still and filtering noise, or data is not updating.")
            
    except KeyboardInterrupt:
        print("\n✗ Benchmark interrupted")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        robot.disconnect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Benchmark robot joint reading frequency")
    
    # Serial port settings
    parser.add_argument('--port', type=str, default="", help="Serial port (e.g. /dev/ttyUSB0 or COM3)")
    parser.add_argument('--gripper_type', type=str, default=None, help="Gripper type (50mm or 100mm)")
    
    # Benchmark settings
    parser.add_argument('--duration', type=float, default=5.0, help="Benchmark duration in seconds (default: 5.0)")
    parser.add_argument('--fast', action='store_true', help="Set update interval to 1ms for higher speed")
    
    args = parser.parse_args()
    
    main(args)

