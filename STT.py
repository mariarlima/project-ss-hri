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

        #  Working example
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch device: {self.device}")

        self.whisper_model = whisper.load_model(whisper_model_id).to(self.device)
        print(f"Loaded Whisper model: {whisper_model_id}")

        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 3

        self.mic = self.initialize_microphone()


    def initialize_microphone(self):
        """
        Try to find and select a microphone from list. Fall back to default.
        Returns the chosen mic object
        """
        # TODO: Complete

        # Working example
        try:
            mic_list = sr.Microphone.list_microphone_names()
            print(f"Available microphones: {mic_list}")

            preferred_mics = [
                'USBAudio1.0',
                'External Headphones',
                'MacBook Pro Microphone'
            ]

            for name in preferred_mics:
                if name in mic_list:
                    index = mic_list.index(name)
                    print(f"Using preferred microphone: {name}")
                    return sr.Microphone(
                        device_index=index,
                        sample_rate=16000 if 'MacBook' in name else None
                    )

            print("Using default microphone.")
            return sr.Microphone()

        except Exception as e:
            print(f"Error initializing microphone: {e}")
            return sr.Microphone()


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

        # Working example
        try:
            with self.mic as source:
                self.recognizer.pause_threshold = pause_threshold
                print("Listening...")

                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=10,
                        phrase_time_limit=phrase_time_limit
                    )
                except sr.WaitTimeoutError:
                    response["success"] = False
                    response["error"] = "Timeout: No speech detected."
                    print(response["error"])
                    return response

                with open("playback.wav", "wb") as f:
                    f.write(audio.get_wav_data())

        except Exception as e:
            response["success"] = False
            response["error"] = f"Microphone/audio error: {str(e)}"
            print(response["error"])
            return response

        try:
            print("Transcribing with Whisper")
            transcription = self.whisper_model.transcribe("playback.wav", language=language)
            response["transcription"] = transcription
            print(f"You said: {transcription['text']}")
        except Exception as e:
            response["success"] = False
            response["error"] = f"Transcription error: {str(e)}"
            print(response["error"])

        return response

        
