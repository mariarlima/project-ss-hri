import torch
import whisper
import speech_recognition as sr
import pyaudio
import numpy as np
from configuration import whisper_model_id
import tempfile
import os
import threading
import time
from typing import Callable, Optional

class STT:
    def __init__(self, realtime_mode=False):
        """
        Initialize STT with optional realtime mode support
        
        Parameters:
        - realtime_mode: If True, optimizes for real-time streaming
        """
        print("Initializing STT")
        
        # Set device (CUDA if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Whisper model - use smaller model for realtime if needed
        model_to_use = "base" if realtime_mode else whisper_model_id
        self.model = whisper.load_model(model_to_use, device=self.device)
        
        # Set up speech recognizer (for traditional mode)
        self.recognizer = sr.Recognizer()
        
        # Realtime mode settings
        self.realtime_mode = realtime_mode
        self.audio_buffer = []
        self.is_recording = False
        self.transcription_callback = None
        
        # Audio stream settings for realtime
        self.audio_stream = None
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_size = 1024
        self.channels = 1
        
        # Initialize microphone
        self.initialize_microphone()

    def initialize_microphone(self):
        """
        Try to find and select a microphone from list. Fall back to default.
        Returns the chosen mic object
        """
        devices = pyaudio.PyAudio()
        print("Number of devices (all APIs, input + output): " + str(devices.get_device_count()))
        
        input_devices = []
        for i in range(devices.get_device_count()):
            device_info = devices.get_device_info_by_index(i)
            if device_info['maxInputChannels'] != 0 and device_info['hostApi'] == 0:
                input_devices.append((i, device_info['name']))
                print(f'Device {i}: {device_info["name"]}')

        if not input_devices:
            print("No suitable input devices found!")
            devices.terminate()
            return
        
        while True:
            try:
                mic_index = int(input("Please select a microphone by index from the list above: "))
                if 0 <= mic_index < devices.get_device_count():
                    device_info = devices.get_device_info_by_index(mic_index)
                    if device_info['maxInputChannels'] != 0:
                        # For traditional mode
                        self.microphone = sr.Microphone(device_index=mic_index)
                        
                        # Store device index for realtime mode
                        self.mic_device_index = mic_index
                        
                        print(f"Selected microphone: {device_info['name']}")
                        break
                    else:
                        print("Selected device is not an input device. Please choose again.")
                else:
                    print("Invalid index. Please choose a valid device index.")
            except ValueError:
                print("Please enter a valid integer index.")
        
        devices.terminate()

    def get_voice_as_text(self, pause_threshold, phrase_time_limit=10, language="en"):
        """
        Traditional STT method - Listen to user speech and transcribe using Whisper.
        Returns the transcribed text string or None if failed
        """
        if self.realtime_mode:
            print("[STT Warning]: get_voice_as_text called in realtime mode")
            return None
            
        # Initialize response dictionary FIRST
        response = {
            "success": True,
            "error": None,
            "transcription": None
        }
        
        try:
            # Record audio using timeout and phrase_time_limit settings
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

    def start_realtime_transcription(self, callback: Callable[[str], None]):
        """
        Start real-time audio transcription
        
        Parameters:
        - callback: Function to call with transcription results
        """
        if not self.realtime_mode:
            print("[STT Error]: Not in realtime mode")
            return False
            
        self.transcription_callback = callback
        
        try:
            # Initialize PyAudio for realtime recording
            audio = pyaudio.PyAudio()
            
            # Open audio stream
            self.audio_stream = audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.mic_device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start the stream
            self.audio_stream.start_stream()
            self.is_recording = True
            
            print("[STT]: Real-time transcription started")
            return True
            
        except Exception as e:
            print(f"[STT Error]: Failed to start realtime transcription: {str(e)}")
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Audio callback for real-time processing
        """
        if self.is_recording:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            
            # Process buffer when it reaches sufficient size (e.g., 3 seconds of audio)
            buffer_size_seconds = 3
            required_samples = self.sample_rate * buffer_size_seconds
            
            if len(self.audio_buffer) >= required_samples:
                # Process the audio buffer in a separate thread to avoid blocking
                buffer_copy = np.array(self.audio_buffer[:required_samples])
                self.audio_buffer = self.audio_buffer[required_samples // 2:]  # Keep some overlap
                
                # Start transcription in separate thread
                threading.Thread(
                    target=self._transcribe_buffer, 
                    args=(buffer_copy,), 
                    daemon=True
                ).start()
        
        return (in_data, pyaudio.paContinue)

    def _transcribe_buffer(self, audio_data):
        """
        Transcribe audio buffer in a separate thread
        """
        try:
            # Use Whisper to transcribe
            result = self.model.transcribe(
                audio_data, 
                language="en",
                task="transcribe",
                no_speech_threshold=0.3,  # More permissive for real-time
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4
            )
            
            text = result.get("text", "").strip()
            
            if text and len(text) > 5:  # Only send meaningful transcriptions
                if self.transcription_callback:
                    self.transcription_callback(text)
                    
        except Exception as e:
            print(f"[STT Error]: Transcription failed: {str(e)}")

    def stop_realtime_transcription(self):
        """
        Stop real-time transcription
        """
        if self.is_recording:
            self.is_recording = False
            
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
                self.audio_stream = None
            
            self.audio_buffer.clear()
            print("[STT]: Real-time transcription stopped")

    def set_realtime_mode(self, enabled: bool):
        """
        Enable or disable realtime mode
        """
        if enabled and not self.realtime_mode:
            # Switch to realtime mode - reload with smaller model if needed
            if whisper_model_id not in ["base", "small", "tiny"]:
                print("[STT]: Switching to base model for better realtime performance")
                self.model = whisper.load_model("base", device=self.device)
        
        self.realtime_mode = enabled
        print(f"[STT]: Realtime mode {'enabled' if enabled else 'disabled'}")

    def get_available_languages(self):
        """
        Get list of supported languages for Whisper
        """
        return list(whisper.tokenizer.LANGUAGES.keys())

    def transcribe_file(self, file_path, language="en"):
        """
        Transcribe an audio file directly
        
        Parameters:
        - file_path: Path to audio file
        - language: Language code for transcription
        
        Returns:
        - Transcription result dictionary
        """
        try:
            result = self.model.transcribe(file_path, language=language)
            return {
                "success": True,
                "text": result.get("text", ""),
                "segments": result.get("segments", []),
                "language": result.get("language", language)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "segments": [],
                "language": language
            }

    def cleanup(self):
        """
        Clean up resources
        """
        if self.is_recording:
            self.stop_realtime_transcription()
        
        print("[STT]: Cleanup completed")

# Enhanced STT class specifically designed for Realtime API integration
class RealtimeSTT(STT):
    """
    STT class optimized for use with OpenAI Realtime API
    Provides lower latency and better integration
    """
    
    def __init__(self):
        super().__init__(realtime_mode=True)
        
        # Use the fastest, most efficient model for realtime
        self.model = whisper.load_model("tiny", device=self.device)
        
        # Optimized settings for realtime
        self.buffer_duration = 1.0  # Process every 1 second
        self.overlap_duration = 0.2  # 200ms overlap
        
        print("[RealtimeSTT]: Initialized with optimized settings")

    def start_streaming(self, audio_callback: Callable[[bytes], None], 
                       text_callback: Optional[Callable[[str], None]] = None):
        """
        Start streaming audio with optional text transcription
        
        Parameters:
        - audio_callback: Called with raw audio bytes for forwarding
        - text_callback: Optional callback for transcribed text
        """
        self.audio_forward_callback = audio_callback
        self.text_callback = text_callback
        
        return self.start_realtime_transcription(self._handle_transcription)
    
    def _handle_transcription(self, text):
        """
        Handle transcription results
        """
        if self.text_callback:
            self.text_callback(text)
        
        print(f"[RealtimeSTT]: {text}")

if __name__ == "__main__":
    # Test both traditional and realtime modes
    print("Testing STT in traditional mode...")
    stt_traditional = STT(realtime_mode=False)
    
    # Test file transcription
    wav_path = "M1F1-Alaw-AFsp.wav"  # Change this to your actual WAV file path
    if os.path.exists(wav_path):
        try:
            result = stt_traditional.transcribe_file(wav_path)
            print("File transcription result:", result)
        except Exception as e:
            print("Error during file transcription:", e)
    
    # Test realtime mode
    print("\nTesting STT in realtime mode...")
    stt_realtime = RealtimeSTT()
    
    def transcription_handler(text):
        print(f"Realtime transcription: {text}")
    
    # Uncomment to test realtime transcription
    # print("Starting realtime transcription for 10 seconds...")
    # stt_realtime.start_realtime_transcription(transcription_handler)
    # time.sleep(10)
    # stt_realtime.stop_realtime_transcription()
    
    print("STT testing completed")