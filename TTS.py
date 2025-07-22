from unrealspeech import UnrealSpeechAPI, play
from configuration import TTS_config
from utils import get_audio_length, read_mp3_as_bytes_url
import openai
import io
import pyaudio
import threading
import time
from typing import Callable, Optional
import queue

class TTS:
    def __init__(self, api_key, signal_queue=None, api_provider="openai", realtime_mode=False):
        """
        Initialize TTS with support for both traditional and realtime modes
        
        Parameters:
        - api_key: API key for the TTS service
        - signal_queue: Queue for coordinating with main loop (traditional mode)
        - api_provider: "openai" or "unrealspeech"
        - realtime_mode: If True, optimizes for real-time streaming
        """
        print("Initializing TTS API")
        
        # Clear audio queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        print("[TTS]: Stopped realtime audio playback")

    def add_audio_chunk(self, audio_chunk: bytes):
        """
        Add audio chunk to playback queue (for external audio sources)
        """
        if self.realtime_mode and self.is_playing:
            self.audio_queue.put(audio_chunk)
        else:
            print("[TTS Warning]: add_audio_chunk requires realtime mode and active playback")

    def set_voice(self, voice_id: str):
        """
        Change the voice used for TTS
        
        Parameters:
        - voice_id: Voice identifier
        """
        if self.api_provider == "openai":
            # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
            valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            if voice_id in valid_voices:
                self.openai_voice_id = voice_id
                print(f"[TTS]: Voice changed to {voice_id}")
            else:
                print(f"[TTS Warning]: Invalid OpenAI voice '{voice_id}'. Valid voices: {valid_voices}")
        elif self.api_provider == "unrealspeech":
            # UnrealSpeech voice IDs
            self.us_voice_id = voice_id
            print(f"[TTS]: UnrealSpeech voice changed to {voice_id}")

    def set_speed(self, speed: float):
        """
        Set speech speed (0.25 to 4.0 for OpenAI, custom range for UnrealSpeech)
        """
        if self.api_provider == "unrealspeech":
            self.us_speed = max(0.1, min(2.0, speed))  # Clamp to reasonable range
            print(f"[TTS]: Speed set to {self.us_speed}")
        else:
            print("[TTS Warning]: Speed adjustment not supported for OpenAI TTS in this implementation")

    def get_supported_voices(self):
        """
        Get list of supported voices for the current provider
        """
        if self.api_provider == "openai":
            return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        elif self.api_provider == "unrealspeech":
            # Return commonly used UnrealSpeech voices
            return ["Scarlett", "Dan", "Liv", "Will", "Amy"]
        return []

    def test_audio_output(self):
        """
        Test audio output with a simple message
        """
        test_text = "Hello, this is a test of the text-to-speech system."
        
        if self.realtime_mode:
            print("[TTS]: Testing realtime audio output...")
            self.stream_text_audio(test_text)
            time.sleep(3)  # Give time for audio to play
            self.stop_realtime_playback()
        else:
            print("[TTS]: Testing traditional audio output...")
            self.play_text_audio(test_text)

    def cleanup(self):
        """
        Clean up TTS resources
        """
        if self.realtime_mode:
            self.stop_realtime_playback()
            if hasattr(self, 'audio'):
                self.audio.terminate()
        
        print("[TTS]: Cleanup completed")

    def get_audio_info(self):
        """
        Get current audio configuration info
        """
        info = {
            "provider": self.api_provider,
            "realtime_mode": self.realtime_mode,
            "sample_rate": getattr(self, 'sample_rate', None),
            "channels": getattr(self, 'channels', None),
            "is_playing": getattr(self, 'is_playing', False)
        }
        
        if self.api_provider == "openai":
            info["model"] = self.openai_model_id
            info["voice"] = self.openai_voice_id
        elif self.api_provider == "unrealspeech":
            info["voice"] = self.us_voice_id
            info["speed"] = self.us_speed
            info["pitch"] = self.us_pitch
            info["bitrate"] = self.us_bitrate
        
        return info

# Enhanced TTS class specifically designed for Realtime API integration
class RealtimeTTS(TTS):
    """
    TTS class optimized for use with OpenAI Realtime API
    Handles audio streaming with minimal latency
    """
    
    def __init__(self, api_key, api_provider="openai"):
        super().__init__(
            api_key=api_key, 
            signal_queue=None, 
            api_provider=api_provider, 
            realtime_mode=True
        )
        
        # Optimized settings for realtime
        self.chunk_size = 512  # Smaller chunks for lower latency
        self.buffer_size = 2048  # Small buffer to minimize delay
        
        print("[RealtimeTTS]: Initialized with optimized low-latency settings")

    def handle_realtime_audio_chunk(self, audio_chunk: bytes):
        """
        Handle audio chunks received from Realtime API
        
        Parameters:
        - audio_chunk: Raw audio data from Realtime API
        """
        if not self.is_playing:
            self._start_audio_playback()
        
        self.add_audio_chunk(audio_chunk)

    def start_output_stream(self):
        """
        Start the audio output stream for Realtime API integration
        """
        self._start_audio_playback()
        return self.is_playing

    def stop_output_stream(self):
        """
        Stop the audio output stream
        """
        self.stop_realtime_playback()

# Utility class for managing multiple TTS instances
class TTSManager:
    """
    Manages multiple TTS instances and provides unified interface
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.instances = {}
        self.current_instance = None
        self.default_provider = "openai"
        
    def create_instance(self, name: str, api_provider: str = None, realtime_mode: bool = False, signal_queue=None):
        """
        Create a new TTS instance
        """
        provider = api_provider or self.default_provider
        
        try:
            instance = TTS(
                api_key=self.api_key,
                signal_queue=signal_queue,
                api_provider=provider,
                realtime_mode=realtime_mode
            )
            
            self.instances[name] = instance
            
            if not self.current_instance:
                self.current_instance = name
                
            print(f"[TTSManager]: Created TTS instance '{name}' with {provider}")
            return True
            
        except Exception as e:
            print(f"[TTSManager Error]: Failed to create instance '{name}': {str(e)}")
            return False

    def switch_instance(self, name: str):
        """
        Switch to a different TTS instance
        """
        if name in self.instances:
            self.current_instance = name
            print(f"[TTSManager]: Switched to instance '{name}'")
            return True
        else:
            print(f"[TTSManager Error]: Instance '{name}' not found")
            return False

    def get_current_instance(self):
        """
        Get the current active TTS instance
        """
        if self.current_instance and self.current_instance in self.instances:
            return self.instances[self.current_instance]
        return None

    def speak(self, text: str, instance_name: str = None):
        """
        Speak text using specified instance or current instance
        """
        instance = None
        
        if instance_name:
            instance = self.instances.get(instance_name)
        else:
            instance = self.get_current_instance()
        
        if not instance:
            print("[TTSManager Error]: No valid TTS instance available")
            return False
        
        try:
            if instance.realtime_mode:
                return instance.stream_text_audio(text)
            else:
                instance.play_text_audio(text)
                return True
        except Exception as e:
            print(f"[TTSManager Error]: Failed to speak: {str(e)}")
            return False

    def cleanup_all(self):
        """
        Clean up all TTS instances
        """
        for name, instance in self.instances.items():
            try:
                instance.cleanup()
                print(f"[TTSManager]: Cleaned up instance '{name}'")
            except Exception as e:
                print(f"[TTSManager Error]: Failed to cleanup '{name}': {str(e)}")
        
        self.instances.clear()
        self.current_instance = None

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    print("Testing TTS implementations...")
    
    # Test traditional TTS
    print("\n=== Testing Traditional TTS ===")
    tts_traditional = TTS(api_key=api_key, api_provider="openai", realtime_mode=False)
    tts_traditional.test_audio_output()
    
    # Test realtime TTS
    print("\n=== Testing Realtime TTS ===")
    tts_realtime = RealtimeTTS(api_key=api_key)
    tts_realtime.test_audio_output()
    
    # Test TTS Manager
    print("\n=== Testing TTS Manager ===")
    manager = TTSManager(api_key=api_key)
    manager.create_instance("traditional", "openai", False)
    manager.create_instance("realtime", "openai", True)
    
    manager.switch_instance("traditional")
    manager.speak("This is traditional mode.")
    
    time.sleep(2)
    
    manager.switch_instance("realtime")
    manager.speak("This is realtime mode.")
    
    time.sleep(3)
    
    # Cleanup
    tts_traditional.cleanup()
    tts_realtime.cleanup()
    manager.cleanup_all()
    
    print("\nTTS testing completed!") Store configuration
        self.api_key = api_key
        self.signal_queue = signal_queue
        self.api_provider = api_provider
        self.realtime_mode = realtime_mode
        
        # Initialize based on provider
        if api_provider == "unrealspeech":
            self.tts_api = UnrealSpeechAPI(api_key)
            self.us_voice_id = TTS_config["unrealspeech"]["voice_id"]
            self.us_bitrate = TTS_config["unrealspeech"]["bit_rate"]
            self.us_speed = TTS_config["unrealspeech"]["speed"]
            self.us_pitch = TTS_config["unrealspeech"]["pitch"]
            self.us_timestamp_type = "sentence"
        elif api_provider == "openai":
            self.openai_model_id = TTS_config["openai"]["model_id"]
            self.openai_voice_id = TTS_config["openai"]["voice_id"]
            openai.api_key = api_key
        else:
            raise ValueError(f"Unknown TTS provider: {api_provider}")
        
        # Realtime streaming setup
        if realtime_mode:
            self.setup_realtime_audio()
        
        print(f"[TTS]: Initialized with {api_provider} provider, realtime_mode={realtime_mode}")

    def setup_realtime_audio(self):
        """
        Setup audio streaming for realtime mode
        """
        # Audio configuration for realtime playback
        self.sample_rate = 24000 if self.api_provider == "openai" else 22050
        self.channels = 1
        self.chunk_size = 1024
        
        # Audio streaming components
        self.audio_queue = queue.Queue()
        self.audio_stream = None
        self.audio_thread = None
        self.is_playing = False
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        print("[TTS]: Realtime audio setup completed")

    def play_text_audio(self, text):
        """
        Traditional method - Convert text to speech and play audio
        """
        if self.realtime_mode:
            print("[TTS Warning]: play_text_audio called in realtime mode")
            return
            
        print("Calling TTS API")
        
        try:
            # Generate audio based on provider
            if self.api_provider == "unrealspeech":
                audio_data = self.tts_api.speech(
                    text=text,
                    voice_id=self.us_voice_id,
                    bitrate=self.us_bitrate,
                    timestamp_type=self.us_timestamp_type,
                    speed=self.us_speed,
                    pitch=self.us_pitch
                )
                audio_url = audio_data["OutputUri"]
                audio_bytes = read_mp3_as_bytes_url(audio_url)
                
            elif self.api_provider == "openai":
                response = openai.audio.speech.create(
                    model=self.openai_model_id,
                    voice=self.openai_voice_id,
                    input=text
                )
                audio_bytes = response.content
            
            # Play the audio
            play(audio_bytes)
            
            # Send audio duration to signal queue if provided
            if self.signal_queue:
                duration = get_audio_length(audio_bytes)
                self.signal_queue.put(duration)
                
        except Exception as e:
            print(f"[TTS Error]: Failed to generate/play audio: {str(e)}")
            if self.signal_queue:
                self.signal_queue.put(2.0)  # Fallback duration

    def stream_text_audio(self, text, audio_callback: Optional[Callable[[bytes], None]] = None):
        """
        Realtime method - Stream text-to-speech audio
        
        Parameters:
        - text: Text to convert to speech
        - audio_callback: Optional callback to receive audio chunks
        """
        if not self.realtime_mode:
            print("[TTS Error]: stream_text_audio requires realtime_mode=True")
            return False
        
        try:
            if self.api_provider == "openai":
                return self._stream_openai_audio(text, audio_callback)
            elif self.api_provider == "unrealspeech":
                return self._stream_unrealspeech_audio(text, audio_callback)
        except Exception as e:
            print(f"[TTS Error]: Failed to stream audio: {str(e)}")
            return False

    def _stream_openai_audio(self, text, audio_callback):
        """
        Stream audio using OpenAI TTS
        """
        try:
            # OpenAI doesn't have native streaming yet, so we'll use regular generation
            # and stream the playback
            response = openai.audio.speech.create(
                model=self.openai_model_id,
                voice=self.openai_voice_id,
                input=text,
                response_format="pcm"  # Raw PCM for better streaming
            )
            
            audio_data = response.content
            
            # Stream audio in chunks
            if audio_callback:
                # Stream in chunks
                chunk_size = self.chunk_size * 2  # 2 bytes per sample for 16-bit
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    audio_callback(chunk)
                    time.sleep(0.01)  # Small delay for streaming effect
            else:
                # Add to queue for internal streaming
                self.audio_queue.put(audio_data)
                self._start_audio_playback()
            
            return True
            
        except Exception as e:
            print(f"[TTS Error]: OpenAI streaming failed: {str(e)}")
            return False

    def _stream_unrealspeech_audio(self, text, audio_callback):
        """
        Stream audio using UnrealSpeech
        """
        try:
            # Generate audio
            audio_data = self.tts_api.speech(
                text=text,
                voice_id=self.us_voice_id,
                bitrate=self.us_bitrate,
                timestamp_type=self.us_timestamp_type,
                speed=self.us_speed,
                pitch=self.us_pitch
            )
            
            # Get audio bytes
            audio_url = audio_data["OutputUri"]
            audio_bytes = read_mp3_as_bytes_url(audio_url)
            
            # Stream the audio
            if audio_callback:
                # Stream in chunks
                chunk_size = self.chunk_size * 2
                for i in range(0, len(audio_bytes), chunk_size):
                    chunk = audio_bytes[i:i + chunk_size]
                    audio_callback(chunk)
                    time.sleep(0.01)
            else:
                # Add to queue for internal streaming
                self.audio_queue.put(audio_bytes)
                self._start_audio_playback()
            
            return True
            
        except Exception as e:
            print(f"[TTS Error]: UnrealSpeech streaming failed: {str(e)}")
            return False

    def _start_audio_playback(self):
        """
        Start audio playback thread for realtime streaming
        """
        if self.is_playing:
            return
        
        self.is_playing = True
        
        # Initialize audio stream
        try:
            self.audio_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.audio_stream.start_stream()
            print("[TTS]: Started realtime audio playback")
            
        except Exception as e:
            print(f"[TTS Error]: Failed to start audio playback: {str(e)}")
            self.is_playing = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        Audio playback callback
        """
        try:
            # Try to get audio data from queue
            audio_data = self.audio_queue.get_nowait()
            
            # Ensure proper size
            bytes_needed = frame_count * 2  # 2 bytes per sample for 16-bit
            
            if len(audio_data) >= bytes_needed:
                output = audio_data[:bytes_needed]
                # Put remaining data back
                remaining = audio_data[bytes_needed:]
                if remaining:
                    self.audio_queue.put(remaining)
            else:
                # Pad with zeros if not enough data
                output = audio_data + b'\x00' * (bytes_needed - len(audio_data))
            
            return (output, pyaudio.paContinue)
            
        except queue.Empty:
            # No audio data available - return silence
            silence = b'\x00' * (frame_count * 2)
            return (silence, pyaudio.paContinue)

    def stop_realtime_playback(self):
        """
        Stop realtime audio playback
        """
        if not self.is_playing:
            return
        
        self.is_playing = False
        
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        #