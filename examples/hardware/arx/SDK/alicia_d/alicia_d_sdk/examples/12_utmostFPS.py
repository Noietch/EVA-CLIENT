# Copyright (c) 2025 Synria Robotics Co., Ltd.
# Licensed under the MIT License.
#
# Author: Synria Robotics Team
# Website: https://synriarobotics.ai

"""
  # Linux/Ubuntu
  python 12_utmostFPS.py --port /dev/ttyACM0
  
  # Windows
  python 12_utmostFPS.py --port COM3
  
  # macOS
  python 12_utmostFPS.py --port /dev/cu.wchusbserial5B140413941
  
  # Custom settings
  python 12_utmostFPS.py --port /dev/ttyUSB0 --baudrate 2000000 --test-timeout 60
"""

import serial
import time
import argparse
import sys


class fpsFlay:
    # Fps
    sendNum: int = 0
    readNum: int = 0
    errNum: int = 0
    sendFPS: float = 0.0
    readFPS: float = 0.0

    # Time
    start_time: float = 0.0
    current_time: float = 0.0
    real_time: float = 0.0
    end_time: float = 0.0

    def __init__(self):
        pass

def main(args):
    Fps = fpsFlay()
    ser = None
    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
        )
        sendBuff = bytearray([0xAA, 0x06, 0x00, 0x01, 0xFE, 0x9A, 0xFF])

        sendFlay = True
        Fps.start_time = time.time()
        Fps.real_time = Fps.start_time
        while sendFlay:
            ser.write(sendBuff)
            Fps.sendNum += 1
            readBuff = ser.read(21)
            if(len(readBuff) != 21 or readBuff[0] != 0xAA or readBuff[-1] != 0xFF or readBuff[1] != 0x06):
                Fps.errNum += 1
                continue
            Fps.readNum += 1
            Fps.current_time = time.time()
            if Fps.current_time - Fps.real_time >= 1.0:
                if Fps.current_time - Fps.start_time >= args.test_timeout: 
                    sendFlay = False
                Fps.real_time = Fps.current_time
                Fps.sendFPS = Fps.sendNum / (Fps.current_time - Fps.start_time)
                Fps.readFPS = Fps.readNum / (Fps.current_time - Fps.start_time)
                print(f"[{Fps.current_time - Fps.start_time:.2f}s]Send FPS: {Fps.sendFPS:.2f}Hz, \
                      Receive FPS: {Fps.readFPS:.2f}Hz, \
                      Error count: {Fps.errNum}")

    except KeyboardInterrupt:
        print(f"Ctrl+C exit")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        Fps.end_time = time.time()
        Fps.sendFPS = Fps.sendNum / (Fps.end_time - Fps.start_time)
        Fps.readFPS = Fps.readNum / (Fps.end_time - Fps.start_time)
        print(f"All Time: {Fps.end_time - Fps.start_time:.2f}s")
        print(f"sendFPS: {Fps.sendFPS:.2f}Hz")
        print(f"readFPS: {Fps.readFPS:.2f}Hz")
        print(f"Sync rate: {(Fps.readNum/Fps.sendNum)*100:.2f}%")
        if ser is not None and ser.is_open:
            ser.close()
            print("Serial port closed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Maximum FPS test for serial communication",
        formatter_class=argparse.RawDescriptionHelpFormatter )
    
    parser.add_argument(
        '--port', 
        type=str, 
        required=True,
        help='Serial port (e.g., /dev/ttyUSB0 for Linux, COM3 for Windows, /dev/cu.* for macOS)'
    )
    parser.add_argument(
        '--baudrate', 
        type=int, 
        default=1000000,
        help='Baud rate (default: 1000000)'
    )
    parser.add_argument(
        '--timeout', 
        type=float, 
        default=0.015,
        help='Serial read timeout in seconds (default: 0.015 for 1kHz update rate)'
    )
    parser.add_argument(
        '--test-timeout', 
        type=float, 
        default=300.0,
        help='Test duration in seconds (default: 300.0)'
    )
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        print("Program terminated")
    except Exception as e:
        print(f"Error occurred: {e}")
        sys.exit(1)