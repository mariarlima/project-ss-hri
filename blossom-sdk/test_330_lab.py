from robot import *
from config import *
import time

from log_conf import logger

my_robot = Robot(config_dict=ROBOT_330_LAB)

logger.info("Starting positions")
my_robot.check_motor_status(["all"])

my_robot.move_motors_sync(args={1:150, 2:150, 3:150, 4:150, 5:130}, duration={1:500, 2:500, 3:500, 4:500, 5:500})
logger.info("Result of move to 150")
my_robot.check_motor_status(["all"])
logger.info("Sleep 2")
time.sleep(2)

my_robot.move_motors_sync(args={1:0, 2:0, 3:0, 4:0, 5:90}, duration={1:500, 2:500, 3:500, 4:500, 5:500})
logger.info("Result of move to 0")
my_robot.check_motor_status(["all"])
logger.info("Sleep 2")
time.sleep(2)

my_robot.move_motors_sync(args={1:-150, 2:-150, 3:-150, 4:-150, 5:50}, duration={1:500, 2:500, 3:500, 4:500, 5:500})
logger.info("Result of move to -150")
my_robot.check_motor_status(["all"])
logger.info("Sleep 2")
time.sleep(2)
my_robot.check_motor_status(["all"])

my_robot.clean_shutdown()
logger.info("Ended")