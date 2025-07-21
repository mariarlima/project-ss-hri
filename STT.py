import torch
import whisper
import speech_recognition as sr
from configuration import whisper_model_id


class STT:
    def __init__(self):
        print("Initializing STT")
        # TODO: Set device (CUDA if available, else CPU)
        # TODO: Load Whisper model using whisper_model_id from configuration.py
        # TODO: Set up speech recognizer and microphone with pause_threshold from configuration.py


    def initialize_microphone(self):
        """
        Try to find and select a microphone from list. Fall back to default.
        Returns the chosen mic object
        """
        # TODO: Complete


    def get_voice_as_text(self, pause_threshold, phrase_time_limit=10, language="en"):
        """
        Listen to user speech and transcribe it to text using Whisper.
        Returns the result of the transcription attempt
        """
        # TODO: Record audio using timeout (max time r.listen will wait until a speech is picked up) and phrase_time_limit (max duration of the audio clip) settings
        # TODO: Handle exceptions (e.g., no speech detected)
        # TODO: Save audio to a temporaty .wav file for transcription
        # TODO: Transcribe the audio using Whisper model and handle exceptions (e.g., API errors, unintelligible speech)

        response = {
            "success": True,
            "error": None,
            "transcription": None
        }

        
