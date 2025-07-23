from unrealspeech import UnrealSpeechAPI, play
from configuration import TTS_config
from utils import get_audio_length, read_mp3_as_bytes_url

class TTS:
    
    def __init__(self, api_key, signal_queue, api_provider="unrealspeech"):
        print("Initializing TTS API")
        # Initialize UnrealSpeech with API key
        self.tts_api = UnrealSpeechAPI(api_key)

        # Store signal_queue for coordination with main interaction loop (main.py)
        self.signal_queue = signal_queue

        # Use TTS_config["unrealspeech"] from configuration.py to set parameters
        self.us_voice_id = TTS_config["unrealspeech"]["voice_id"]
        self.us_bitrate = TTS_config["unrealspeech"]["bit_rate"]
        self.us_speed = TTS_config["unrealspeech"]["speed"]
        self.us_pitch = TTS_config["unrealspeech"]["pitch"]
        self.us_timestamp_type = "sentence"

    def play_text_audio(self, text):
        """
        Convert given text to speech using UnrealSpeech and play the audio.
        Returns audio duration (in seconds) for later use in signal_queue
        """
        print("Calling TTS API")

        # Convert TTS 
        self.audio_data = self.tts_api.speech(text=text,
                                                 voice_id=self.us_voice_id,
                                                 bitrate=self.us_bitrate, 
                                                 timestamp_type=self.us_timestamp_type, 
                                                 speed=self.us_speed, 
                                                 pitch=self.us_pitch)

        # Retrieve audio bytes using read_mp3_as_bytes_url()
        audio_url = self.audio_data["OutputUri"]
        audio_bytes = read_mp3_as_bytes_url(audio_url)

        # Play the audio
        play(audio_bytes)

        # Get audio duration using get_audio_length() and put into the signal queue
        self.signal_queue.put(get_audio_length(audio_bytes))