# blossom-sdk

A repository for controlling XL320 and XL330 motors (in the shape of a Blossom). 

### Required files 
- robot.py : contains the Robot class, built using the dynamixel SDK
- control_table_defs.py : contains addresses for 320 and 330 control tables
- conversion.py : contains functions to convert between degrees and DXL positions 
- config.py : contains configuration dictionaries for a "robot" 
- log.conf.py : contains the log definition and initiation
- sequence.py : contains the Sequence class, used to play sequences on a "robot" 
- Sequences : a directory of .json files containing sequences
- sequence_schema.json : a json schema for validating sequence files

### Useful files
- cli-robot.py : CLI support for a robot with a velocity-based profile 

### Optional files for sanity checks
- test_320.py : testing Robot functions on a 320 setup
- test_330.py : testing Robot functions on a 330 setup
- test_seq.py : testing Sequence functions on either setup
- test_330_time.py : testing Robot functions on a 330 setup (time-based)

## A note on hardware
For XL-320 based robots, ensure you use a 7.4V power supply. 
For XL-330 based robots, ensure you use a 5.0V power supply. 
Hardware error statuses will be set if the input voltage is outside the expected range. 
Flashing LEDs on a motor indicates a hardware error has occurred.  The most common error is out of range supply voltage.
If you encounter a hardware error, use Dynamixel Wizard to reset or reload the motor.

For all robots, the baud rate used should be 1,000,000. 
You can check the baud rate of your motors using the [Dynamixel Wizard](https://emanual.robotis.com/docs/en/software/dynamixel/dynamixel_wizard2/). 
You can also manually set the baud rates in the Wizard. 
Baud rate is in the EEPROM section of the control table, meaning it persists even when the motors lose power. 

You can also use the Dynamixel Wizard to determine the name of the port your motors are connected to. 
The name of the port changes based on the U2D2 used, and the computer. 
For Raspberry Pis, the name is typically "/dev/ttyUSB0". 
For Macs, the name is typically of the format: "/dev/tty.usbserial-[XXXXXXXX]".

**Do not mix and match motor types in a robot.** 
This code does not support it. 

## Calibrating the robot
TODO

## How to make a robot configuration dictionary
Each configuration contains a "controllers" dictionary and a "motors" dictionary.

"controllers" contains:
- port - the port the robot is connected to (this changes based on the U2D2 used) 
- protocol - the DXL protocol (this should always be 2)
- baudrate - the baud rate that dictates communication speed (this should always be 1000000)
- drivemode - for 330s only, possible drive modes for the robot configuration. 0 = velocity based profile; 8 = velocity based profile + torque on by goal update; 12 = time based profile + torque on by goal update  
- blocking - a boolean indicating whether the robot should spin to wait for moves to complete 

"motors" contains dictionaries of motors, where the key is the string name of the motor.
Each motor dictionary contains:
- id - the numerical id for the motor (should be 1-5)
- type - the type of the motor. For 320s, this is 350. For 330s, this is 1200
- angle_limit - the range in degrees in which the motor should be allowed to move. For all base and tower motors, this should be [-150.0, 150]. For the ears, this should be [50.0, 130.0]. Note that 320s cannot go further than +/- 150, as they can only rotate 300 degrees, unlike 330s.

## How to use the Robot class
See test_320.py, test_330.py, or test_330_time.py for code examples: 

From the same directory as the .py files, use the following imports: 
```
from robot import *
from config import *
```

Then make your Robot object using one of the configuration dictionaries from config.
```
my_robot = Robot(config_dict = ROBOT_330_TIME)
```

Before doing anything else, you should enable the torque. 

For 320 robots, this is a requirement. 
**_If you do not enable the torque, the robot will never move. If you try to tell it to move, the move functions will enter an infinite loop. No, there is no way to prevent the infinite loop other than enabling the torque first._**

For 330 robots, this is also a requirement **if the drive mode is set to 0**. 
If the bit that enables torque enable by goal update is set, torque will be automatically enabled for you. 
```
my_robot.enable_torque()
```

You can then use any of the following functions: move_motors, move_motors_sync, reset, check_motor_status, get_diagnostic. 

When you are done with the robot, make sure to shut it down appropriately (disable torque, close ports, etc.)! Do this using:
```
my_robot.clean_shutdown()
```

## How to make a sequence
A sequence is a json file with the following features:
- "animation" - the name of the sequence 
- "frame_list" - a list of frame dictionaries
- "label" - the label of the sequence 

Each frame dictionary contains the following features: 
- "millis" - a float representing the time in milliseconds at which the robot should begin moving to the specified position (the start time for the current frame, relative to the beginning of the sequence playback) 
- "positions" - a list of dictionairies containing the name of the motor to move ("dof"), the position to move it to ("pos"), and the time to take to get to that position ("duration")

## How to use the Sequence class 
From the same directory as the .py files, use the following imports: 
```
from sequence import *
from config import *
from robot import *
```

Next, create a Robot object using one of the configuration dictionaries in config, and enable torque. 
```
my_robot = Robot(config_dict=ROBOT_330_TIME)
my_robot.enable_torque()
```

Then make a Sequence object. Sequence's `__init__` accept a file name for a specific sequence, and a configuration dictionary as input. This should be the same dictionary you used to make the robot.
```
my_sequence = Sequence(file_name="Sequences/tiny_test.json", robot_config=ROBOT_330_TIME)
```

To play the sequence, call the following function. Do not forget to pass in your Robot object as an argument. 
```
my_sequence.play_sequence(robot=my_robot)
```

Once you're done, remember to shut down the robot.
```
my_robot.clean_shutdown()
```


## Using the CLI
Start the CLI via `python cli-robot.py`.

The CLI currently supports the following commands for a robot with a velocity-based profile:
- `help`
- `status all` - gets current position of all motors 
- `status <motor ids or names, separated by spaces>` - gets current position of specified motors 
- `diagnostic all` - gets set error codes of all motors 
- `diagnostic <motor ids or names, separated by spaces>` - gets set error codes of specified motors 
- `move <motor id or name:position in degrees> ...` - blocks and moves each motor in the list to the specified position
- `sync_move <motor id or name:position in degrees> ...` - does not block and moves all motors to the specified position simultaneously
- `reset` - does not block and moves all motors to 0 (or as close as possible)
- `play_seq <file>` - play a specified sequence from a json file 
- `shutdown` - do a clean shutdown and end the program 

## Requirements 
The only requirement is the dynamixel_sdk library. 
This can be installed by running:

`pip install dynamixel-sdk`

Note that when importing the sdk in python, you use an underscore. 
When installing via pip, you use a hyphen. 

## What this code has been tested on 
Computers:
- MacBook Pro, Apple M1 Max, M3 Max, and M4 Max, Sequoia 15.3.1
- Raspberry Pi 4 B
- Raspberry Pi 5

Dynamixel Motors:
- [XL330](https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/)
- [XL320](https://emanual.robotis.com/docs/en/dxl/x/xl320/)

# So why does my XL330 Blossom not behave like my XL320 Blossom? 
All dynamixel motors are controlled by reading and writing values in a control table. 
The XL330 control table is significantly larger than the XL320 control table. 
This means that the XL330 has capabilities that the XL320s do not have. 
While it may seem like the XL330 control table should be a simple extension of the XL320 control table (where all the 320 control table registers are 
present in the 330 control table), this is not the case. Some registers are unique to each motor type. Refer to the Dynamixel Motor documentation linked above 
to better understand the capabilities and limitations of each motor type.

The most important consequence of these different control tables is that XL-320s **do not** support a time-based profile (movements are controlled by duration), while XL-330s do. 

Even in a velocity-based profile (movements are controlled by speed and acceleration), XL-330s may behave differently than XL-320s due to the additional registers in the XL-330 control table.
