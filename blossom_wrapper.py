import random
import sys
from blossom_interface import BlossomInterface
from configuration import sequence_metadata

sys.path.insert(0, './Blossom-Controller')

class BlossomWrapper:
    def __init__(self, server_ip=None, server_port=None):
        """
        Initialize the Blossom robot interface.
        """
        self.bli = BlossomInterface(server_ip, server_port)

    def do_sequence(self, seq, delay_time=0):
        """
        Execute a specific predefined sequence.
        """
        # TODO: complete
        self.bli.do_sequence(seq, delay_time)

    def do_random_sequence_from_list(self, seq_list, delay_time=0):
        """
        Select and play random sequence from list. Print name to track.
        """
        # TODO: complete

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

        # TODO: Complete. Find closest matching sequence length (in configuration.py) to audio_length
