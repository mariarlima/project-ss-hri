# ------------------- XL-320 Control Table Definition ------------------
CT_XL320_ADDR = {
    # EEPROM Area (Addresses 0-18)
    0: ("Model Number", 2),
    2: ("Firmware Version", 1),
    3: ("ID", 1),
    4: ("Baud Rate", 1),
    5: ("Return Delay Time", 1),
    6: ("CW Angle Limit", 2),
    8: ("CCW Angle Limit", 2),
    11: ("Control Mode", 1),
    12: ("Temperature Limit", 1),
    13: ("Min Voltage Limit", 1),
    14: ("Max Voltage Limit", 1),
    15: ("Max Torque", 2),
    17: ("Status Return Level", 1),
    18: ("Shutdown", 1),

    # RAM Area (Addresses 24-51)
    24: ("Torque Enable", 1),
    25: ("LED", 1),
    27: ("D Gain", 1),
    28: ("I Gain", 1),
    29: ("P Gain", 1),
    30: ("Goal Position", 2),
    32: ("Moving Speed", 2),
    35: ("Torque Limit", 2),
    37: ("Present Position", 2),
    39: ("Present Speed", 2),
    41: ("Present Load", 2),
    45: ("Present Voltage", 1),
    46: ("Present Temperature", 1),
    47: ("Registered Instruction", 1),
    49: ("Moving", 1),
    50: ("Hardware Error Status", 1),
    51: ("Punch", 2),
}

# ------------------- XL-320 Configurations ------------------
XL320_CONFIG = {
    "ADDR_TORQUE_ENABLE" : 24,
    "ADDR_GOAL_POSITION" : 30,
    "ADDR_PRESENT_POSITION" : 37,
    "ADDR_HARDWARE_ERROR_STATUS" : 50,
    "TORQUE_ENABLE" : 1,
    "TORQUE_DISABLE" : 0,
    "ADDR_MOVING_SPEED": 32,
    "ADDR_TORQUE_LIMIT": 35,
    "ADDR_P_GAIN": 29,
    "ADDR_CW_ANGLE_LIMIT": 6,
    "ADDR_CCW_ANGLE_LIMIT": 8,
    "ADDR_MOVING": 49
}

# ------------------- XL330-M288-T Control Table Definition ------------------
CT_XL330_ADDR = {
    # EEPROM Area (Addresses 0-63)
    0: ("Model Number", 2),
    2: ("Model Information", 4),
    6: ("Firmware Version", 1),
    7: ("ID", 1),
    8: ("Baud Rate", 1),
    9: ("Return Delay Time", 1),
    10: ("Drive Mode", 1),
    11: ("Operating Mode", 1),
    12: ("Secondary ID", 1),
    13: ("Protocol Type", 1),
    20: ("Homing Offset", 4),
    24: ("Moving Threshold", 4),
    31: ("Temperature Limit", 1),
    32: ("Max Voltage Limit", 2),
    34: ("Min Voltage Limit", 2),
    36: ("PWM Limit", 2),
    38: ("Current Limit", 2),
    44: ("Velocity Limit", 4),
    48: ("Max Position Limit", 4),
    52: ("Min Position Limit", 4),
    60: ("Startup Configuration", 1),
    62: ("PWM Slope", 1),
    63: ("Shutdown", 1),

    # RAM Area (Addresses 64-227)
    64: ("Torque Enable", 1),
    65: ("LED", 1),
    68: ("Status Return Level", 1),
    69: ("Registered Instruction", 1),
    70: ("Hardware Error Status", 1),
    76: ("Velocity I Gain", 2),
    78: ("Velocity P Gain", 2),
    80: ("Position D Gain", 2),
    82: ("Position I Gain", 2),
    84: ("Position P Gain", 2),
    88: ("Feedforward 2nd Gain", 2),
    90: ("Feedforward 1st Gain", 2),
    98: ("Bus Watchdog", 1),
    100: ("Goal PWM", 2),
    102: ("Goal Current", 2),
    104: ("Goal Velocity", 4),
    108: ("Profile Acceleration", 4),
    112: ("Profile Velocity", 4),
    116: ("Goal Position", 4),
    120: ("Realtime Tick", 2),
    122: ("Moving", 1),
    123: ("Moving Status", 1),
    124: ("Present PWM", 2),
    126: ("Present Current", 2),
    128: ("Present Velocity", 4),
    132: ("Present Position", 4),
    136: ("Velocity Trajectory", 4),
    140: ("Position Trajectory", 4),
    144: ("Present Input Voltage", 2),
    146: ("Present Temperature", 1),
    147: ("Backup Ready", 1)
}

# ------------------- XL-330 Configurations ------------------
XL330_CONFIG = {
    "ADDR_DRIVE_MODE": 10,
    "ADDR_TORQUE_ENABLE" : 64,
    "ADDR_HARDWARE_ERROR_STATUS" : 70,
    "ADDR_PROFILE_ACCELERATION": 108,
    "ADDR_PROFILE_VELOCITY": 112,
    "ADDR_GOAL_POSITION" : 116,
    "ADDR_PRESENT_POSITION" : 132,
    "TORQUE_ENABLE" : 1,
    "TORQUE_DISABLE" : 0,
    "ADDR_MAX_POSITION_LIMIT": 48,
    "ADDR_MIN_POSITION_LIMIT": 52,
    "ADDR_MOVING_THRESHOLD": 24,
    "ADDR_MOVING": 122
}

DRIVE_MODE_TIME = 0x4
