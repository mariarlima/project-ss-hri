#!/usr/bin/env python3
"""
Simple OpenAI Realtime API Conversation System
A clean, single-file implementation for elderly-friendly voice conversations.

Requirements:
pip install websockets>=10.0 pyaudio python-dotenv

Note: If you get connection errors, try:
pip install --upgrade websockets
"""

import os
import asyncio
import json
import base64
import pyaudio
import threading
import time
import signal
import sys
from queue import Queue, Empty
from dotenv import load_dotenv

# Check websockets version and import appropriately
try:
    import websockets
    # Test if we have the modern version
    websockets_version = getattr(websockets, '__version__', '0.0')
    print(f"[System]: Using websockets version {websockets_version}")
except ImportError:
    print("❌ Error: websockets not installed. Run: pip install websockets")
    sys.exit(1)

# Load environment variables
load_dotenv()

class RealtimeConversation:
    """Simple Realtime API conversation system"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        
        # WebSocket connection
        self.websocket = None
        self.connected = False
        self.running = False
        
        # Audio configuration - FIXED for proper audio quality
        self.sample_rate = 24000  # OpenAI Realtime API outputs at 24kHz
        self.input_sample_rate = 16000  # Input to API must be 16kHz
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16
        
        # Audio components
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        self.audio_queue = Queue()
        self.audio_buffer = b""  # Buffer for proper audio assembly
        
        # System prompt for elderly-friendly responses
        self.system_prompt = """You are Blossom, a friendly AI companion for elderly users. 
        Keep responses short (under 200 characters), use simple language, be warm and encouraging. 
        Avoid complex topics or technical terms. Focus on being a supportive, patient companion."""
        
        # Response evaluation for elderly users
        self.forbidden_words = ['death', 'dying', 'kill', 'suicide', 'depressed']
        self.max_response_length = 200
        
        print("[System]: Realtime Conversation System initialized")

    def setup_audio(self):
        """Initialize audio input and output"""
        try:
            # List available microphones
            print("\n=== Available Microphones ===")
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"  {i}: {info['name']}")
            
            # Get user's microphone choice
            while True:
                try:
                    choice = input("\nSelect microphone (number) or press Enter for default: ").strip()
                    mic_index = int(choice) if choice else None
                    
                    # Test the microphone
                    test_stream = self.audio.open(
                        format=self.audio_format,
                        channels=self.channels,
                        rate=self.sample_rate,
                        input=True,
                        input_device_index=mic_index,
                        frames_per_buffer=self.chunk_size
                    )
                    test_stream.close()
                    
                    self.mic_index = mic_index
                    selected_name = self.audio.get_device_info_by_index(mic_index)['name'] if mic_index else "Default"
                    print(f"[Audio]: Selected microphone: {selected_name}")
                    break
                    
                except (ValueError, Exception) as e:
                    print(f"Invalid choice or device error: {e}. Please try again.")
            
            return True
            
        except Exception as e:
            print(f"[Audio Error]: Failed to setup audio: {e}")
            return False

    def start_audio_streams(self):
        """Start audio input and output streams with correct sample rates"""
        try:
            # Input stream for recording (16kHz for API)
            self.input_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.input_sample_rate,  # 16kHz for input
                input=True,
                input_device_index=self.mic_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_input_callback
            )
            
            # Output stream for playback (24kHz for output)
            self.output_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,  # 24kHz for output
                output=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_output_callback
            )
            
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
            print(f"[Audio]: Audio streams started (Input: {self.input_sample_rate}Hz, Output: {self.sample_rate}Hz)")
            return True
            
        except Exception as e:
            print(f"[Audio Error]: Failed to start streams: {e}")
            return False

    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """Handle audio input - send to Realtime API"""
        if self.connected and self.running:
            # Send audio to Realtime API (non-blocking)
            asyncio.run_coroutine_threadsafe(
                self._send_audio(in_data), 
                self.event_loop
            )
        return (in_data, pyaudio.paContinue)

    def _audio_output_callback(self, in_data, frame_count, time_info, status):
        """Handle audio output - play received audio with proper buffering"""
        try:
            # Collect audio chunks until we have enough data
            while len(self.audio_buffer) < frame_count * 2:  # 2 bytes per sample
                try:
                    chunk = self.audio_queue.get_nowait()
                    self.audio_buffer += chunk
                except Empty:
                    break
            
            bytes_needed = frame_count * 2
            
            if len(self.audio_buffer) >= bytes_needed:
                # Extract the needed audio data
                output = self.audio_buffer[:bytes_needed]
                self.audio_buffer = self.audio_buffer[bytes_needed:]
                return (output, pyaudio.paContinue)
            elif len(self.audio_buffer) > 0:
                # Use what we have and pad with silence
                output = self.audio_buffer + b'\x00' * (bytes_needed - len(self.audio_buffer))
                self.audio_buffer = b""
                return (output, pyaudio.paContinue)
            else:
                # No audio available - return silence
                silence = b'\x00' * bytes_needed
                return (silence, pyaudio.paContinue)
                
        except Exception as e:
            print(f"[Audio Output Error]: {e}")
            silence = b'\x00' * (frame_count * 2)
            return (silence, pyaudio.paContinue)

    async def connect(self):
        """Connect to OpenAI Realtime API"""
        try:
            url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
            
            print("[System]: Connecting to OpenAI Realtime API...")
            
            # Try multiple connection methods for compatibility
            connection_methods = [
                # Method 1: Modern websockets with extra_headers
                lambda: websockets.connect(url, extra_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }),
                # Method 2: Alternative header method (for older versions)
                lambda: websockets.connect(url, additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }),
                # Method 3: Basic connection (will need manual auth)
                lambda: websockets.connect(url)
            ]
            
            connection_successful = False
            for i, method in enumerate(connection_methods):
                try:
                    print(f"[System]: Trying connection method {i+1}...")
                    self.websocket = await method()
                    connection_successful = True
                    print(f"[System]: Connection method {i+1} successful!")
                    break
                except Exception as e:
                    print(f"[System]: Connection method {i+1} failed: {e}")
                    continue
            
            if not connection_successful:
                print("[Connection Error]: All connection methods failed")
                print("\nTroubleshooting steps:")
                print("1. Update websockets: pip install --upgrade websockets")
                print("2. Check your OpenAI API key is correct")
                print("3. Ensure you have Realtime API access")
                print("4. Check your internet connection")
                return False
            
            self.connected = True
            
            # Configure session with correct audio settings
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": self.system_prompt,
                    "voice": "alloy",  # Use alloy for most natural voice
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 1000  # Increased for better turn detection
                    }
                }
            }
            
            await self.websocket.send(json.dumps(session_config))
            print("[System]: ✅ Connected to Realtime API!")
            return True
            
        except Exception as e:
            print(f"[Connection Error]: {e}")
            print("\nTroubleshooting steps:")
            print("1. Update websockets: pip install --upgrade websockets")
            print("2. Check your OpenAI API key")
            print("3. Ensure you have Realtime API access")
            self.connected = False
            return False

    async def _send_audio(self, audio_data):
        """Send audio data to Realtime API"""
        if not self.connected or not self.websocket:
            return
            
        try:
            # Encode audio as base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }
            
            await self.websocket.send(json.dumps(message))
            
        except Exception as e:
            print(f"[Audio Send Error]: {e}")

    async def listen_for_responses(self):
        """Listen for responses from Realtime API"""
        try:
            async for message in self.websocket:
                event = json.loads(message)
                await self._handle_event(event)
                
        except websockets.exceptions.ConnectionClosed:
            print("[System]: Connection closed by server")
            self.connected = False
        except Exception as e:
            print(f"[Listen Error]: {e}")
            self.connected = False

    async def _handle_event(self, event):
        """Handle events from Realtime API"""
        event_type = event.get("type")
        
        if event_type == "session.created":
            print("[System]: Session created successfully")
            
        elif event_type == "input_audio_buffer.speech_started":
            print("🎤 Listening...")
            
        elif event_type == "input_audio_buffer.speech_stopped":
            print("🤖 Processing...")
            
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            print(f"[You said]: {transcript}")
            
        elif event_type == "response.audio.delta":
            # Received audio from AI - handle properly
            audio_data = event.get("delta", "")
            if audio_data:
                try:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(audio_data)
                    
                    # Add to queue for playback
                    self.audio_queue.put(audio_bytes)
                    
                except Exception as e:
                    print(f"[Audio Decode Error]: {e}")
                    
        elif event_type == "response.text.delta":
            # Received text response (for display)
            text = event.get("delta", "")
            print(text, end="", flush=True)
            
        elif event_type == "response.done":
            print("\n[Response complete]")
            
        elif event_type == "error":
            error_msg = event.get("error", {}).get("message", "Unknown error")
            print(f"[API Error]: {error_msg}")

    def evaluate_response(self, text):
        """Simple response evaluation for elderly users"""
        if not text:
            return False, "Empty response"
            
        if len(text) > self.max_response_length:
            return False, "Response too long"
            
        text_lower = text.lower()
        for word in self.forbidden_words:
            if word in text_lower:
                return False, f"Contains inappropriate word: {word}"
                
        return True, "Response appropriate"

    async def start_conversation(self):
        """Start the conversation loop"""
        self.running = True
        
        # Send initial greeting
        greeting = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello! I'm Blossom, your friendly companion. I need you to describe me the image that you are seeing."}]
            }
        }
        
        await self.websocket.send(json.dumps(greeting))
        
        # Request response
        response_request = {
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]}
        }

        # It is necessary to evaluate the response
        eval_response = self.evaluate_response("text")
        if not eval_response[0]:
            # If response is not appropriate, send a default message instead
            response_request["response"]["modalities"] = ["text"]
            response_request["response"]["text"] = "Keep going, I am hearing."
            
        await self.websocket.send(json.dumps(response_request))
        
        print("\n" + "="*50)
        print("🌸 Conversation Started!")
        print("💬 Start talking - I'll listen and respond")
        print("⏹️  Press Ctrl+C to end the conversation")
        print("="*50)
        
        # Listen for responses
        await self.listen_for_responses()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.connected = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        self.audio.terminate()
        print("\n[System]: Cleanup completed")

    def run(self):
        """Main run method"""
        try:
            # Setup audio
            if not self.setup_audio():
                return False
                
            if not self.start_audio_streams():
                return False
            
            # Create event loop for async operations
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            
            # Run the conversation
            async def main_loop():
                if await self.connect():
                    await self.start_conversation()
                else:
                    print("[System]: Failed to connect to Realtime API")
                    return False
            
            self.event_loop.run_until_complete(main_loop())
            
        except KeyboardInterrupt:
            print("\n[System]: Conversation ended by user")
        except Exception as e:
            print(f"[System Error]: {e}")
        finally:
            self.cleanup()

def main():
    """Main entry point"""
    print("Blossom - Simple Realtime Conversation System")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Create and run conversation system
    conversation = RealtimeConversation(api_key)
    
    # Setup graceful shutdown
    def signal_handler(signum, frame):
        print("\n[System]: Shutting down gracefully...")
        conversation.cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the conversation
    conversation.run()

if __name__ == "__main__":
    main()