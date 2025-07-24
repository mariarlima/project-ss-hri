from unrealspeech import UnrealSpeechAPI, play
from configuration import TTS_config
from utils import get_audio_length, read_mp3_as_bytes_url


class TTS:
    def __init__(self, api_key, signal_queue, api_provider="unrealspeech"):
        print("Initializing TTS API")
        # TODO: Initialize UnrealSpeech with API key 
        # TODO: Store signal_queue for coordination with main interaction loop (main.py)
        # TODO: Use TTS_config["unrealspeech"] from configuration.py to set parameters

        #  Working example
        self.api_provider = api_provider
        self.signal_queue = signal_queue

        if api_provider == "unrealspeech":
            self.voice_id = TTS_config["unrealspeech"]["voice_id"]
            self.bit_rate = TTS_config["unrealspeech"]["bit_rate"]
            self.speed = TTS_config["unrealspeech"]["speed"]
            self.pitch = TTS_config["unrealspeech"]["pitch"]
            self.speech_api = UnrealSpeechAPI(api_key)


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

        #  Working example
        audio_bytes = None

        if self.api_provider == "unrealspeech":
            tts_audio_data = self.speech_api.speech(
                text=text,
                voice_id=self.voice_id,
                bitrate=self.bit_rate,
                speed=self.speed,
                pitch=self.pitch
            )

            print("Playing TTS")
            audio_bytes = read_mp3_as_bytes_url(tts_audio_data["OutputUri"])
            play(audio_bytes)

            duration = get_audio_length(audio_bytes)
            self.signal_queue.put(duration)

        return get_audio_length(audio_bytes)
        
    