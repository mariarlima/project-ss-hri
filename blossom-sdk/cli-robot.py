# only works with velocity-based profile robots 
import log_conf

from robot import *
from config import *
from sequence import *

import threading
import queue

# create robot object 
my_robot = Robot(ROBOT_330)
my_robot.enable_torque()

# make queue and threading event 
command_queue = queue.Queue()
shutdown_flag = threading.Event()

def motor_control_thread():
    while not shutdown_flag.is_set():
        # get a command from the queue 
        try:
            command = command_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        if command["type"] == "status":
            my_robot.check_motor_status(command["args"])
        
        elif command["type"] == "diagnostic":
            my_robot.get_diagnostic(command["args"])

        elif command["type"] == "move":
            my_robot.move_motors(command["args"], degrees=True)

        elif command["type"] == "sync_move":
            my_robot.move_motors_sync(command["args"], degrees=True)

        elif command["type"] == "reset":
            my_robot.reset()

        elif command["type"] == "play_seq":
            # TODO: untested 
            my_sequence = Sequence(file_name=command["args"], robot_config=ROBOT_330)
            my_sequence.play_sequence(robot=my_robot)

def cli_interface_thread():
    while not shutdown_flag.is_set():
        cmd = input("\nEnter command (type 'help' for options):\n> ")
        parts = cmd.split()

        if parts[0] == "help":
            print("Command Options:")
            print("Status: status all, status <motor ids and/or names>")
            print("Diagnostic: diagnostic all, status <motor ids and/or names>")
            print("Move (blocking): move <motor id or name>:<position in degrees or DXL increments> ...")
            print("Move (non-blocking): sync_move <motor id or name>:<position in degrees or DXL increments> ...")
            print("Reset (non-blocking): sync_move all motors to 0 degrees (top dead center) ...")
            print("Sequences: play_seq <sequence name.json>")
            print("Shutdown: shutdown")

        elif parts[0] == "status":
            args = []
            for elem in parts[1:]:
                if elem.isnumeric():
                    args.append(int(elem))
                else:
                    args.append(elem)
            
            command_queue.put({"type": "status", "args": args})
        
        elif parts[0] == "diagnostic":
            args = []
            for elem in parts[1:]:
                if elem.isnumeric():
                    args.append(int(elem))
                else:
                    args.append(elem)
            
            command_queue.put({"type": "diagnostic", "args": args})

        elif parts[0] == "move":
            args = {}
            for elem in parts[1:]:
                motor, pos = elem.split(":")
                if motor.isnumeric():
                    args[int(motor)] = int(pos)
                else:
                    args[motor] = int(pos)

            command_queue.put({"type": "move", "args": args})

        elif parts[0] == "sync_move":
            args = {}
            for elem in parts[1:]:
                motor, pos = elem.split(":")
                if motor.isnumeric():
                    args[int(motor)] = int(pos)
                else:
                    args[motor] = int(pos)

            command_queue.put({"type": "sync_move", "args": args})

        elif parts[0] == "reset":
            command_queue.put({"type": "reset", "args": None})

        elif parts[0] == "play_seq":
            file_name = parts[1]
            command_queue.put({"type": "play_seq", "args": file_name})

        elif parts[0] == "shutdown":
            shutdown_flag.set()
            my_robot.clean_shutdown()
            break

        else:
            print("Invalid command.")


motor_thread = threading.Thread(target=motor_control_thread, daemon=True)
cli_thread = threading.Thread(target=cli_interface_thread, daemon=False)

motor_thread.start()
cli_thread.start()

cli_thread.join()