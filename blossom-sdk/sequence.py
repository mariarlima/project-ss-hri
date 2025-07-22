import time 
from log_conf import logger

import json
import jsonschema
from jsonschema import validate

from robot import *

SCHEMA_PATH = "blossom-sdk/Sequences/sequence_schema.json"

class Sequence():
    def __init__(self, file_name, robot_config):
        # Load and validate the sequence file
        self.seq_dict = self.load_and_validate(file_name)

        # get the name of the animation 
        self.name = self.seq_dict['animation']

        # get a list of motors, frame times, and frame positions 
        # frame times and positions should be of the same length 
        self.motors_used, self.frame_times, self.frame_positions, self.frame_durations = self.interpret_sequence(robot_config)

        self.num_frames = len(self.frame_positions)

    def load_and_validate(self, file_path):
        """Load the JSON sequence from file, validate it against the schema, and return the JSON data."""
        try:
            # Load JSON data from file
            with open(file_path, "r") as json_file:
                json_data = json.load(json_file)

            # Load JSON Schema from file
            with open(SCHEMA_PATH, "r") as schema_file:
                schema = json.load(schema_file)

            # Validate JSON against the schema
            validate(instance=json_data, schema=schema)

            logger.info("Sequence is valid against the schema!")
            return json_data

        except jsonschema.exceptions.ValidationError as e:
            logger.critical("JSON validation error: %s", e.message)
            quit()

        except jsonschema.exceptions.SchemaError as e:
            logger.critical("Schema error: %s", e.message)
            quit()

        except FileNotFoundError as e:
            logger.critical("File not found: %s", e)
            quit()

        except json.JSONDecodeError as e:
            logger.critical("Invalid JSON format: %s", e)
            quit()

        except Exception as e:
            logger.critical("Unexpected error: %s", e)
            quit()

    def interpret_sequence(self, robot_config):
        """Gets a list of motors, frame times, frame positions, and frame 
        durations from the raw sequence dictionary for each frame in the 
        sequence. Also checks that motors in sequence match available motors 
        from robot config file. If more motors are available in the sequence 
        than in the config file, drops the additional motors from the 
        sequence data structure."""

        # get available motors from config file
        available_motors = set(robot_config["motors"].keys())

        # get motors used from first frame
        motors_used = [position["dof"] for position in self.seq_dict["frame_list"][0]["positions"]]

        # determine valid indices where motors_used is in available_motors
        valid_indices = [i for i, motor in enumerate(motors_used) if motor in available_motors]

        # filter motors_used to only include valid indices
        motors_used = [motors_used[i] for i in valid_indices]

        # get the frame times in milliseconds, and lists of positions and durations for each frame
        frame_times = []
        frame_positions = []
        frame_durations = []
        for frame in self.seq_dict["frame_list"]:
            frame_times.append(frame['millis'])
            positions = [frame["positions"][i]["pos"] for i in valid_indices]
            durations = [frame["positions"][i]["duration"] for i in valid_indices]
            frame_positions.append(positions)
            frame_durations.append(durations)

        # check that frame times and positions are the same size 
        assert len(frame_times) == len(frame_positions)
        assert len(frame_times) == len(frame_durations)

        return (motors_used, frame_times, frame_positions, frame_durations)

    def play_sequence(self, robot=None):
        # start time
        start_time = time.monotonic_ns()

         # iterate through the list of frames in the sequence
        for i in range(self.num_frames):
            # get args for motor movement
            args = {motor: pos for motor, pos in zip(self.motors_used, self.frame_positions[i])}
            # get durations for motor movement 
            durations = {motor: dur for motor, dur in zip(self.motors_used, self.frame_durations[i])}

            delta_time_ms = (time.monotonic_ns() - start_time) / 1000000.0
            t_delay_ms = self.frame_times[i] - delta_time_ms

            if t_delay_ms > 0:
                t_delay_s = t_delay_ms / 1000.0
                logger.info("Frame %d begins in %2.4f seconds", i, t_delay_s)
                time.sleep(t_delay_s)
            else:
                logger.warning("Frame %d is late by %2.4f ms; skipping sleep", i, abs(t_delay_ms))

           # move robot
            logger.info("Frame %d starting", i)
            robot.move_motors_sync(args, duration=durations, degrees=True)
            logger.info("Frame: %d ended", i)
            
        return 1

    def to_string(self):
        logger.info("Name: %s", self.name)
        logger.info("Sequences: %s", self.seq_dict)
        logger.info("Motors: %s", self.motors_used)
        logger.info("Frame Times: %s", self.frame_times)
        logger.info("Frame Positions: %s", self.frame_positions)
        logger.info("Frame Durations: %s", self.frame_durations)