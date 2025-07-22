from robot import *
from config import *
import time

from log_conf import logger

my_robot = Robot(config_dict=ROBOT_WAVESHARE)

logger.info("Starting positions")
my_robot.check_motor_status(["all"])

start_time = time.time()
time_diff = 0
while time_diff < 300: 
    logger.info("Inhale position")
    my_robot.move_motors_sync(args={1:150, 2:150, 3:150, 4:150}, duration={4:10000, 1:10000, 2:10000, 3:10000})

    logger.info("Exhale position")
    my_robot.move_motors_sync(args={1:-150, 2:-150, 3:-150, 4:-150}, duration={4:10000, 1:10000, 2:10000, 3:10000})

    current_time = time.time()
    time_diff = current_time - start_time


start_time = time.time()
time_diff = 0
while time_diff < 300: 
    logger.info("Inhale position")
    my_robot.move_motors_sync(args={1:150, 2:150, 3:150, 4:150}, duration={4:7000, 1:7000, 2:7000, 3:7000})

    logger.info("Exhale position")
    my_robot.move_motors_sync(args={1:-150, 2:-150, 3:-150, 4:-150}, duration={4:7000, 1:7000, 2:7000, 3:7000})

    current_time = time.time()
    time_diff = current_time - start_time


start_time = time.time()
time_diff = 0
while time_diff < 120: 
    logger.info("Inhale position")
    my_robot.move_motors_sync(args={1:150, 2:150, 3:150, 4:150}, duration={4:3000, 1:3000, 2:3000, 3:3000})

    logger.info("Exhale position")
    my_robot.move_motors_sync(args={1:-150, 2:-150, 3:-150, 4:-150}, duration={4:3000, 1:3000, 2:3000, 3:3000})

    current_time = time.time()
    time_diff = current_time - start_time


start_time = time.time()
time_diff = 0
while time_diff < 120: 
    logger.info("Inhale position")
    my_robot.move_motors_sync(args={1:150, 2:150, 3:150, 4:150}, duration={4:1500, 1:1500, 2:1500, 3:1500})

    logger.info("Exhale position")
    my_robot.move_motors_sync(args={1:-150, 2:-150, 3:-150, 4:-150}, duration={4:1500, 1:1500, 2:1500, 3:1500})

    current_time = time.time()
    time_diff = current_time - start_time