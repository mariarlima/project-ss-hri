import random
import sys
from configuration import sequence_metadata
import os

sys.path.insert(0, './Blossom-Controller')
from blossom_interface import BlossomInterface

PATH_JSON_FILES = '/home/pedrodias/Documents/git-repos/project-ss-hri/Blossom-Controller/blossom-public/blossompy/src/sequences/woody/cognitive'

class BlossomWrapper:
    def __init__(self, server_ip=None, server_port=None):
        """
        Initialize the Blossom robot interface.
        """
        self.bli = BlossomInterface(server_ip, server_port)
        self.sequence_metadata = [
            {
            'filename': pos_json,
            'path': os.path.join(PATH_JSON_FILES, pos_json)
            }
            for pos_json in os.listdir(PATH_JSON_FILES) if pos_json.endswith('.json')
        ]

        self.seq_durations = sequence_metadata["sequence_length_boundary_list"]["prompt"]
        self.seq_names = sequence_metadata["sequence_list"]["prompt_timed"]

    def do_sequence(self, seq, delay_time=0):
        """
        Execute a specific predefined sequence.
        """
        self.bli.do_sequence(seq, delay_time)

    def do_random_sequence_from_list(self, seq_list, delay_time=0):
        """
        Select and play random sequence from list. Print name to track.
        """
        seq = random.choice(seq_list)
        print(f"[Blossom] Playing random sequence: {seq}")
        self.bli.do_sequence(seq, delay_time)

    def reset(self):
        """
        Reset Blossom motors to neutral position.
        """
        self.bli.reset()

    def do_prompt_sequence_matching(self, delay_time=0, audio_length=0):
        """
        Select and play a sequence with closest duration to the audio duration.
        Sync robot motion with TTS output.
        """
        print(f"[Blossom] Matching sequence for audio_length={audio_length:.2f}s with delay_time={delay_time:.2f}s")

        # Find closest matching sequence length (in configuration.py) to audio_length
        closest_seq = None
        min_diff = float('inf')

        # Check the duration of the audio length
        for idx, seq_duration in enumerate(self.seq_durations):
            diff = abs(seq_duration - audio_length)
            if diff < min_diff:
                min_diff = diff
                closest_seq = self.seq_names[idx] if isinstance(self.seq_names[idx], str) else self.seq_names[idx][0]

        # Get the duration from sequence_metadata using the correct structure
        if closest_seq:
            # Find the index of the sublist in seq_names where closest_seq is present
            matched_idx = None
            for idx, seq_group in enumerate(self.seq_names):
                if isinstance(seq_group, list) and closest_seq in seq_group:
                    matched_idx = idx
                    break
                elif seq_group == closest_seq:
                    matched_idx = idx
                    break
            if matched_idx is not None:
                duration = sequence_metadata["sequence_length_boundary_list"]["prompt"][matched_idx]
                print(f"[Blossom] Playing matched sequence: {closest_seq} (duration={duration:.2f}s)")
                self.bli.do_sequence(closest_seq, delay_time)
            else:
                print(f"[Blossom] No matching sequence found for '{closest_seq}'.")
        else:
            print(f"[Blossom] No matching sequence found for '{closest_seq}'.")

if __name__ == '__main__':
    a = BlossomWrapper()
