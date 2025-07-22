"""
Based on:
1. https://github.com/hrc2/blossom-public/blob/24fb52742090ecf57596385fe94de2ca6b5f772b/motor_calib.py
2. https://github.com/hrc2/blossom-public/blob/24fb52742090ecf57596385fe94de2ca6b5f772b/ear_calib.py

Assumes base is id 4, towers 1-3 are ids, 5 (if it exists) is ears, 6+ are other additions. 
"""

import platform
import sys
import glob

import time

from dynamixel_sdk import *
from conversion import *

# Modified from: https://poppy-project.github.io/pypot/_modules/pypot/dynamixel.html#_get_available_ports
def get_available_ports():
    """ Tries to find the available serial ports on your system. """
    if platform.system() == 'Darwin':
        return glob.glob('/dev/tty.usb*')

    elif platform.system() == 'Linux':
        return glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyAMA*')

    elif sys.platform.lower() == 'cygwin':
        return glob.glob('/dev/com*')

    else:
        raise EnvironmentError('{} is an unsupported platform, cannot find serial ports !'.format(platform.system()))
    
    return []

def move_motor(id, pos):
    """Move a specified motor to a postion in degrees."""

    # translate into dynamixel compatible values if degrees=True
    dxl_pos = degree_to_dxl(pos, motor_type)


    # make the moves 
    if motor_type == 350:
        packet_handler.write2ByteTxRx(port_handler, id, 30, dxl_pos)
        
        # spin until completion
        while 1:
            time.sleep(0.1)

            # get present position and compare to goal position 
            current_pos, _, _ = packet_handler.read2ByteTxRx(port_handler, id, 37)
                
            if not (abs(dxl_pos - current_pos) > 10):
                break
    
    elif motor_type == 1200:
        packet_handler.write4ByteTxRx(port_handler, id, 116, dxl_pos)

        # spin until completion
        while 1:
            time.sleep(0.1)

            # get present position and compare to goal position 
            current_pos, _, _ = packet_handler.read4ByteTxRx(port_handler, id, 132)
                
            if not (abs(dxl_pos - current_pos) > 10):
                break

    return 1


def enable_torque(id):
    """Enables torque. Torque must be enabled before motors will move."""
    if motor_type == 1200:
        packet_handler.write1ByteTxRx(port_handler, id, 64, 1)
    elif motor_type == 350:
        packet_handler.write1ByteTxRx(port_handler, id, 24, 1)

def disable_torque(id):
    """Enables torque. Torque must be enabled before motors will move."""
    if motor_type == 1200:
        packet_handler.write1ByteTxRx(port_handler, id, 64, 0)
    elif motor_type == 350:
        packet_handler.write1ByteTxRx(port_handler, id, 24, 0)


# get port name 
device_name = get_available_ports()[0]

baud_rate = 1000000

# open port and set baud rate 
port_handler = PortHandler(device_name)
packet_handler = PacketHandler(2)

if not port_handler.openPort():
    print("Could not open  port")
    quit()
else:
    print(f"Successfully opened port: {device_name}")

if not port_handler.setBaudRate(baud_rate):
    print(f"Failed to set baud rate: {baud_rate}")
    quit()
else:
    print(f"Successfully set baud rate: {baud_rate}")

# TODO: this is hard coded right now. Could use more flexibility 
IDS = [1, 2, 3, 4, 5]

# ping for motor type 
motor_type, _, _ = packet_handler.ping(port_handler, 1)

for id in IDS:
    enable_torque(id=id)

    if id <= 4:
        time.sleep(1)
        move_motor(id=id, pos=100)
        input(f"Motor {id} position: 100; Attach horn then press 'Enter'. ")

        time.sleep(1)
        move_motor(id=id, pos=0)
        input(f"Motor {id} position: 0; Calibrate string length then press 'Enter'. ")

        time.sleep(1)
        move_motor(id=id, pos=100)
        print(f"Motor {id} position: 100; Calibration complete!")

    elif id == 5:
        time.sleep(1)
        move_motor(id=id, pos=100)
        input(f"Motor {id} position: 100; Attach horn then press 'Enter'. ")

        # note: cornell moved to 150 to attach string -- 
        # only works on motors the first time (once you set a movement range, 150 is out of bounds for ears )
        time.sleep(1)
        move_motor(id=id, pos=130)
        input(f"Motor {id} position: 130; tighten string around ear so that it's lined against the ear holder, press 'Enter' to continue. ")

    disable_torque(id=id)

port_handler.closePort()

print("Calibration complete.")