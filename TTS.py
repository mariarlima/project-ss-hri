from unrealspeech import UnrealSpeechAPI, play
from configuration import TTS_config
from utils import get_audio_length, read_mp3_as_bytes_url


class TTS:
    def __init__(self, api_key, signal_queue, api_provider="unrealspeech"):
        print("Initializing TTS API")
        # TODO: Initialize UnrealSpeech with API key 
        # TODO: Store signal_queue for coordination with main interaction loop (main.py)
        # TODO: Use TTS_config["unrealspeech"] from configuration.py to set parameters


    def play_text_audio(self, text):
        """
        Convert given text to speech using UnrealSpeech and play the audio.
        Returns audio duration (in seconds) for later use in signal_queue
        """
        print("Calling TTS API")

        # TODO: Convert TTS 
        # TODO: Retrieve audio bytes using read_mp3_as_bytes_url() 
        # TODO: Play the audio
        # TODO: Get audio duration using get_audio_length() and put into the signal queue
        
    