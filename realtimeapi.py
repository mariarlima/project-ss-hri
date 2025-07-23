#!/usr/bin/env python3
"""
Enhanced OpenAI Realtime API Conversation System
Integrated with Blossom wrapper, image analysis, and comprehensive logging.

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
import queue
import websockets

from queue import Queue, Empty
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Optional, Any
from blossom_wrapper import BlossomWrapper
BLOSSOM_AVAILABLE = False # Enable Blossom

# Load environment variables from .env file in the root file
load_dotenv()

# Class to set the message history fo the LLM to be ablo to know previous messages
class MessageHistory:
    """Manages conversation message history"""
    
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.session_start_time = datetime.now()
        self.session_id = f"session_{int(time.time())}"
    
    def add_message(self, role: str, content: str, message_type: str = "text", metadata: Optional[Dict] = None):
        """Add a message to history"""
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,  # "user", "assistant", "system"
            "content": content,
            "type": message_type,  # "text", "audio", "image"
            "metadata": metadata or {}
        }
        self.messages.append(message)
        return message
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation"""
        return {
            "session_id": self.session_id,
            "start_time": self.session_start_time.isoformat(),
            "message_count": len(self.messages),
            "duration_minutes": (datetime.now() - self.session_start_time).total_seconds() / 60,
            "messages": self.messages
        }
    
    def save_to_file(self, filename: Optional[str] = None):
        """Save conversation history to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{self.session_id}_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.get_conversation_summary(), f, indent=2, ensure_ascii=False)
            print(f"[History]: Conversation saved to {filename}")
            return filename
        except Exception as e:
            print(f"[History Error]: Failed to save conversation: {e}")
            return None

class InteractiveChatRT:
    """Enhanced Realtime API conversation system with Blossom integration"""
    
    def __init__(self, api_key: str, image_url: str = None):
        self.api_key = api_key
        self.image_url = image_url
        
        # WebSocket connection
        self.websocket = None
        self.connected = False
        self.running = False
        
        # Audio configuration
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
        self.audio_buffer = b""
        self.audio_start_time = 0
        
        # Turn-taking control
        # Thi is used since it is hard do code the inerruptions part, and with this we can guarantee it listens to the user.
        self.ai_is_speaking = False
        self.listening_enabled = True
        self.audio_lock = threading.Lock()
        self.audio_playback_active = False
        self.last_audio_time = 0
        self.response_complete_time = 0
        self.total_audio_received = 0
        self.audio_playback_finished = False
        
        # Blossom integration
        self.blossom = None
        self.signal_queue = queue.Queue()  # For TTS-Blossom coordination
        if BLOSSOM_AVAILABLE:
            try:
                self.blossom = BlossomWrapper()
                print("[System]: Blossom wrapper initialized successfully")
            except Exception as e:
                print(f"[Blossom Warning]: Failed to initialize Blossom: {e}")
                self.blossom = None
        
        # Message history
        self.history = MessageHistory()
        
        # Enhanced system prompt for image description
        self.system_prompt = self._build_system_prompt()
        
        # Response evaluation
        self.forbidden_words = ['death', 'dying', 'kill', 'suicide', 'depressed']
        self.max_response_length = 300  # Increased for image descriptions
        
        # Performance tracking
        self.performance_metrics = {}
        
        print("[System]: Enhanced Realtime Conversation System initialized")

    def _build_system_prompt(self) -> str:
        """Build the system prompt based on whether image is provided"""
        base_prompt = """You are Blossom, a friendly AI companion designed to help elderly users engage in meaningful conversations about visual content. 

Your primary role is to guide users through describing images in detail, helping them observe and articulate what they see while providing gentle encouragement and support.

Guidelines:
- Keep responses concise but warm (under 300 characters for speech)
- Use simple, clear language appropriate for elderly users
- Be patient and encouraging
- Ask follow-up questions to help users notice more details
- Provide gentle hints when users seem stuck
- Celebrate their observations and descriptions
- If users repeat themselves, acknowledge it kindly and guide them to new aspects"""

        if self.image_url:
            image_prompt = f"""

IMPORTANT: An image has been provided for the user to describe. Your task is to:
1. Initially greet the user warmly and explain that you'd like them to describe the image they're seeing
2. Guide them through describing different aspects: people, objects, colors, settings, emotions, actions
3. Ask specific follow-up questions like "What do you notice about the person's expression?" or "What's happening in the background?"
4. Help them explore details they might miss
5. Keep the conversation focused on the image description exercise

Image URL: {self.image_url}
"""
            return base_prompt + image_prompt
        
        return base_prompt

    def setup_audio(self):
        """Initialize audio input and output"""
        try:
            # List available microphones
            print("\n List of Available Microphones")
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

                    # In case that mic_index is equal to None it just goes Default
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
        """Start audio input and output streams"""
        try:
            # Input stream for recording (16kHz for API)
            self.input_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.input_sample_rate,
                input=True,
                input_device_index=self.mic_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_input_callback
            )
            
            # Output stream for playback (24kHz for output)
            self.output_stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
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
        """Handle audio input - send to Realtime API only when not speaking"""
        with self.audio_lock:
            if self.connected and self.running and self.listening_enabled and not self.ai_is_speaking:
                # Send audio to Realtime API (non-blocking)
                asyncio.run_coroutine_threadsafe(
                    self._send_audio(in_data), 
                    self.event_loop
                )
        return (in_data, pyaudio.paContinue)

    def _audio_output_callback(self, in_data, frame_count, time_info, status):
        """Handle audio output with simple buffering"""
        try:
            # Collect audio chunks
            while len(self.audio_buffer) < frame_count * 2:
                try:
                    chunk = self.audio_queue.get_nowait()
                    self.audio_buffer += chunk
                except Empty:
                    break
            
            bytes_needed = frame_count * 2
            
            if len(self.audio_buffer) >= bytes_needed:
                output = self.audio_buffer[:bytes_needed]
                self.audio_buffer = self.audio_buffer[bytes_needed:]
                return (output, pyaudio.paContinue)

            elif len(self.audio_buffer) > 0:
                output = self.audio_buffer + b'\x00' * (bytes_needed - len(self.audio_buffer))
                self.audio_buffer = b""
                
                # 🔍 Check if this was the last chunk and mic not yet reactivated
                if not self.audio_buffer and self.ai_is_speaking and not self.audio_playback_finished:
                    self.audio_playback_finished = True
                    print("[Debug]: Audio buffer empty – reactivating microphone")
                    threading.Thread(target=self._reactivate_microphone).start()
                
                return (output, pyaudio.paContinue)

            else:
                silence = b'\x00' * bytes_needed

                # 🔍 Reactivate mic if it wasn't already
                if self.ai_is_speaking and not self.audio_playback_finished:
                    self.audio_playback_finished = True
                    print("[Debug]: Silent fallback – reactivating microphone")
                    threading.Timer(0.5, self._reactivate_microphone).start()

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
                lambda: websockets.connect(url, extra_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }),
                lambda: websockets.connect(url, additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }),
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
                return False
            
            self.connected = True
            
            # Configure session with enhanced settings
            session_config = {
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": self.system_prompt,
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 1200  # Slightly longer for elderly users
                    }
                }
            }
            
            await self.websocket.send(json.dumps(session_config))
            print("[System]: ✅ Connected to Realtime API!")
            
            # Log connection to history
            self.history.add_message(
                "system", 
                "Connected to Realtime API", 
                "system",
                {"image_url": self.image_url, "session_config": session_config}
            )
            
            return True
            
        except Exception as e:
            print(f"[Connection Error]: {e}")
            self.connected = False
            return False

    def set_listening_state(self, enabled: bool):
        """Control whether the system listens for user input"""
        with self.audio_lock:
            self.listening_enabled = enabled
            status = "enabled" if enabled else "disabled"
            print(f"[Audio Control]: Listening {status}")

    def set_ai_speaking_state(self, speaking: bool):
        """Control AI speaking state to manage turn-taking"""
        with self.audio_lock:
            self.ai_is_speaking = speaking
            if speaking:
                print("AI speaking - microphone muted")
            else:
                print("AI finished - microphone active")

    async def _send_audio(self, audio_data):
        """Send audio data to Realtime API with turn-taking control"""
        if not self.connected or not self.websocket:
            return
            
        # Double-check we should be sending audio
        with self.audio_lock:
            if self.ai_is_speaking or not self.listening_enabled:
                return
            
        try:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }
            
            await self.websocket.send(json.dumps(message))
            
        except Exception as e:
            print(f"[Audio Send Error]: {e}")

    def _trigger_blossom_animation(self, audio_length: float, delay: float = 0.0):
        """Trigger Blossom animation in a separate thread"""
        if self.blossom:
            try:
                robot_thread = threading.Thread(
                    target=self.blossom.do_prompt_sequence_matching,
                    kwargs={"audio_length": audio_length, "delay_time": delay}
                )
                robot_thread.daemon = True
                robot_thread.start()
                return robot_thread
            except Exception as e:
                print(f"[Blossom Error]: Failed to trigger animation: {e}")
                return None
        return None

    async def listen_for_responses(self):
        """Listen for responses from Realtime API"""
        try:
            current_response_text = ""
            response_start_time = None
            
            async for message in self.websocket:
                event = json.loads(message)
                await self._handle_event(event, current_response_text, response_start_time)
                
        except websockets.exceptions.ConnectionClosed:
            print("[System]: Connection closed by server")
            self.connected = False
        except Exception as e:
            print(f"[Listen Error]: {e}")
            self.connected = False

    async def _handle_event(self, event, current_response_text="", response_start_time=None):
        """Handle events from Realtime API with enhanced logging and Blossom integration"""
        event_type = event.get("type")
        
        if event_type == "session.created":
            print("[System]: Session created successfully")
            
        elif event_type == "input_audio_buffer.speech_started":
            print("Listening...")
            self.current_speech_start = time.time()
            
        elif event_type == "input_audio_buffer.speech_stopped":
            print("Processing...")
            if hasattr(self, 'current_speech_start'):
                speech_duration = time.time() - self.current_speech_start
                self.performance_metrics['last_speech_duration'] = speech_duration
            
        elif event_type == "response.audio_transcript.started":
            # AI started speaking - disable listening
            #self.set_ai_speaking_state(True)
            response_start_time = time.time()
            
        elif event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            print(f"[You said]: {transcript}")
            
            # Add user message to history
            self.history.add_message(
                "user", 
                transcript, 
                "audio",
                {
                    "speech_duration": self.performance_metrics.get('last_speech_duration', 0),
                    "transcription_confidence": event.get("confidence", None)
                }
            )
            
        elif event_type == "response.audio.delta":
            # Handle AI audio response
            audio_data = event.get("delta", "")
            if audio_data:
                try:
                    audio_bytes = base64.b64decode(audio_data)
                    audio_duration = len(audio_bytes) / (self.sample_rate * 2)  # 2 bytes per sample
                    self.total_audio_received += audio_duration
                    
                    self.audio_queue.put(audio_bytes)
                    
                    # Mark that we're receiving audio - this will disable listening
                    if not self.ai_is_speaking:
                        self.audio_start_time = time.time()
                        self.set_ai_speaking_state(True)
                        print(f"[Debug]: Started receiving audio")
                        
                except Exception as e:
                    print(f"[Audio Decode Error]: {e}")
                    
        elif event_type == "response.text.delta":
            # Accumulate text response
            text_delta = event.get("delta", "")
            current_response_text += text_delta
            print(text_delta, end="", flush=True)
            
        elif event_type == "response.audio_transcript.delta":
            # Handle audio transcript (what the AI is saying)
            transcript_delta = event.get("delta", "")
            current_response_text += transcript_delta
            
        elif event_type == "response.done":
            print(f"\n[Response complete] - Total audio received: {self.total_audio_received:.2f}s")
            self.response_complete_time = time.time()

            # Guarda o texto no histórico
            if current_response_text.strip():
                self.history.add_message(
                    "assistant", 
                    current_response_text.strip(), 
                    "audio",
                    {
                        "response_time": time.time() - (response_start_time or time.time()),
                        "response_length": len(current_response_text),
                        "audio_duration": self.total_audio_received
                    }
                )

            # Trigger Blossom animation
            try:
                estimated_audio_length = max(self.signal_queue.get(timeout=1.0), self.total_audio_received)
                if estimated_audio_length > 0:
                    self._trigger_blossom_animation(estimated_audio_length, delay=0.0)
            except queue.Empty:
                fallback_duration = max(self.total_audio_received, 2.0)
                self._trigger_blossom_animation(fallback_duration, delay=0.0)

            # Only reactivate microphone after safety delay
            final_delay = max(self.total_audio_received, 0.3) + 0.8  # adiciona margem extra
            print(f"[Debug]: Delaying mic reactivation by {final_delay:.2f}s")
            threading.Timer(0.5, self._reactivate_microphone).start()  # 500ms delay

            # Reset
            self.audio_playback_finished = False  # Reset flag for next response

            current_response_text = ""
            response_start_time = None
            self.total_audio_received = 0
            
        elif event_type == "response.cancelled":
            # AI response was cancelled - re-enable listening immediately since no audio
            print("\n[Response cancelled]")
            self.set_ai_speaking_state(False)
            
        elif event_type == "error":
            error_msg = event.get("error", {}).get("message", "Unknown error")
            print(f"[API Error]: {error_msg}")
            
            # Re-enable listening on error immediately
            self.set_ai_speaking_state(False)
            
            # Log error to history
            self.history.add_message(
                "system", 
                f"API Error: {error_msg}", 
                "error",
                {"error_details": event.get("error", {})}
            )

    def _reactivate_microphone(self):
        """Reactivate microphone after audio finishes"""
        self.set_ai_speaking_state(False)
        print("[Debug]: Microphone reactivated after audio completion")

    def evaluate_response(self, text):
        """Enhanced response evaluation"""
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
        """Start the enhanced conversation with image context"""
        self.running = True
        
        # Build initial greeting with image context
        if self.image_url:
            initial_message = """Hello! I'm Blossom, your friendly companion. I have an image that I'd love for you to describe to me in detail. 

Take your time to look at the image and tell me everything you can see - the people, objects, colors, setting, and anything else that catches your attention. Don't worry about getting everything perfect; I'm here to help guide you through it!

What do you notice first when you look at the image?"""
        else:
            initial_message = """Hello! I'm Blossom, your friendly companion. I'm here to have a nice conversation with you. What would you like to talk about today?"""
        
        # Send initial greeting
        greeting = {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": initial_message}]
            }
        }
        
        # Add image context if provided
        if self.image_url:
            # Add system message about the image
            image_context = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "text", "text": f"Image URL for reference: {self.image_url}. Guide the user to describe this image in detail."}]
                }
            }
            await self.websocket.send(json.dumps(image_context))  # <-- Send image context, not response_request

        await self.websocket.send(json.dumps(greeting))

        # Request response
        response_request = {
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]}
        }

        # Mute microphone BEFORE requesting response to prevent feedback
        self.set_ai_speaking_state(True)
        await self.websocket.send(json.dumps(response_request))
        
        # Log session start
        self.history.add_message(
            "assistant", 
            initial_message, 
            "audio",
            {
                "session_type": "image_description" if self.image_url else "general_conversation",
                "image_url": self.image_url
            }
        )
        
        print("\n" + "="*60)
        print("Blossom Chat has Started!")
        if self.image_url:
            print("Image Description Session Active")
            print(f"Image: {self.image_url}")
        if self.blossom:
            print("Blossom animations enabled")
        print("Start talking - I'll listen and respond")
        print("⏹Press Ctrl+C to end the conversation")
        print("="*60)
        
        # Listen for responses
        await self.listen_for_responses()

    def cleanup(self):
        """Clean up resources and save conversation history"""
        print("\n[System]: Cleaning up...")
        self.running = False
        self.connected = False
        
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            
        self.audio.terminate()
        
        # Save conversation history
        filename = self.history.save_to_file()
        if filename:
            print(f"[System]: Conversation history saved to {filename}")
        
        # Print session summary
        summary = self.history.get_conversation_summary()
        print(f"[System]: Session completed")
        print(f"[Stats]: Duration: {summary['duration_minutes']:.1f} minutes")
        print(f"[Stats]: Messages: {summary['message_count']}")
        
        print("[System]: Cleanup completed")

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
            # Log error to history if available
            if hasattr(self, 'history'):
                self.history.add_message(
                    "system", 
                    f"System Error: {str(e)}", 
                    "error",
                    {"error_type": "system_error", "timestamp": time.time()}
                )
        finally:
            self.cleanup()

def main():
    """Enhanced main entry point"""
    print("Blossom Enhanced Realtime Conversation System")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    # Get image URL from user or environment
    image_url = "/home/pedrodias/Documents/git-repos/project-ss-hri/Cookie-Theft-Picture-4_W640.jpg"
    if image_url and os.path.exists(image_url):
        print(f"[System]: Using image URL: {image_url}")
    
    # Create and run enhanced conversation system
    conversation = InteractiveChatRT(api_key, image_url)
    
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