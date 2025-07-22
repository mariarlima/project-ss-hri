import torch
import whisper
import speech_recognition as sr
import pyaudio
from configuration import whisper_model_id
import tempfile
import os


class STT:
    def __init__(self):
        print("Initializing STT")
        # Set device (CUDA if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load Whisper model using whisper_model_id from configuration.py
        self.model = whisper.load_model(whisper_model_id, device=self.device)

        # Set up speech recognizer
        self.recognizer = sr.Recognizer()

        self.initialize_microphone()


    def initialize_microphone(self):
        """
        Try to find and select a microphone from list. Fall back to default.
        Returns the chosen mic object
        """
        # Complete
        devices = pyaudio.PyAudio()
        print("Number of devices (all APIs, input + output): " + str(devices.get_device_count()))
        for i in range (devices.get_device_count()):
            device_info = devices.get_device_info_by_index(i)
            if device_info['maxInputChannels'] != 0 and device_info['hostApi'] == 0:
                print('Device ' + str(i) + ': ' + device_info['name'])

        while True:
            try:
                mic_index = int(input("Please select a microphone by index from the list above: "))
                if 0 <= mic_index < devices.get_device_count():
                    device_info = devices.get_device_info_by_index(mic_index)
                    if device_info['maxInputChannels'] != 0:
                        self.microphone = sr.Microphone(device_index=mic_index)
                        print(f"Selected microphone: {device_info['name']}")
                        break
                    else:
                        print("Selected device is not an input device. Please choose again.")
                else:
                    print("Invalid index. Please choose a valid device index.")
            except ValueError:
                print("Please enter a valid integer index.")              

    def get_voice_as_text(self, pause_threshold, phrase_time_limit=10, language="en"):
        """
        Listen to user speech and transcribe it to text using Whisper.
        Returns the transcribed text string or None if failed
        """
        # Initialize response dictionary FIRST
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }
        
        try:
            # Record audio using timeout and phrase_time_limit settings
            # Set pause threshold for speech recognition
            self.recognizer.pause_threshold = pause_threshold
            self.recognizer.dynamic_energy_threshold = True

            # Use the selected microphone to capture audio
            with self.microphone as source:
                print("Listening to the user")
                audio = self.recognizer.listen(source, 
                                               phrase_time_limit=phrase_time_limit, 
                                               timeout=pause_threshold)
            
            # Save audio to a temporary WAV file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_file.write(audio.get_wav_data())
                temp_audio_path = temp_audio_file.name

            # Transcribe using Whisper
            try:
                result = self.model.transcribe(temp_audio_path, language=language)
                response["transcription"] = result.get("text", "").strip()
                
                # Clean up temp file
                os.remove(temp_audio_path)
                
                # Return just the transcription text (like your main.py expects)
                return response["transcription"] if response["transcription"] else None
                
            except Exception as e:
                response["success"] = False
                response["error"] = f"Whisper transcription failed: {str(e)}"
                
                # Clean up temp file even if transcription fails
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                
                print(f"[STT Error]: {response['error']}")
                return None
                
        except Exception as e:
            response["success"] = False
            response["error"] = f"Audio capture failed: {str(e)}"
            print(f"[STT Error]: {response['error']}")
            return None

if __name__ == "__main__":
    stt = STT()
    # Example: transcribe a WAV file from your root directory
    wav_path = "M1F1-Alaw-AFsp.wav"  # Change this to your actual WAV file path
    try:
        result = stt.model.transcribe(wav_path)
        print("Transcription:", result.get("text", ""))
    except Exception as e:
        print("Error during transcription:", e)
        
