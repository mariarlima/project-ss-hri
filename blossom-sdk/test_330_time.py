import log_conf

from robot import *
from config import ROBOT_330_TIME

import time

my_robot = Robot(config_dict=ROBOT_330_TIME)

print("Current positions")
my_robot.check_motor_status(["all"])

my_robot.move_motors_sync(args={1:150, 2:150, 3:150, 4:150}, duration={4:2000, 1:2000, 2:2000, 3:2000})
print("\nResult of move to 150")
my_robot.check_motor_status(["all"])
time.sleep(2)

# my_robot.move_motors_sync(args={1:0, 2:0, 3:0, 4:0}, duration={4:2000, 1:2000, 2:2000, 3:2000})
# my_robot.check_motor_status(["all"])
# time.sleep(2)

my_robot.move_motors_sync(args={1:-150, 2:-150, 3:-150, 4:-150}, duration={4:2000, 1:2000, 2:2000, 3:2000})
print("\nResult of move to -150")
my_robot.check_motor_status(["all"])
time.sleep(2)

# my_robot.move_motors_sync(args={1:0, 2:0, 3:0, 4:0}, duration={4:2000, 1:2000, 2:2000, 3:2000})
# my_robot.check_motor_status(["all"])

my_robot.move_motors_sync(args={1:150, 2:150, 3:150, 4:150}, duration={4:2000, 1:2000, 2:2000, 3:2000})
print("\nResult of move to 150")
my_robot.check_motor_status(["all"])
time.sleep(2)

my_robot.clean_shutdown()