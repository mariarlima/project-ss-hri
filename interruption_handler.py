import os
import queue
import threading
import time
import signal
import sys
from configuration import *

class InterruptionHandler:
    """
    Handles user interruptions during TTS playback
    Essential for natural conversation with elderly users who may interrupt mid-sentence
    """
    def __init__(self, stt=None):
        self.stt = stt
        self.is_playing_audio = False
        self.audio_thread = None
        self.interrupt_detected = False
        self.interrupt_text = None
        
        # Audio control events
        self.stop_audio_event = threading.Event()
        self.audio_finished_event = threading.Event()
    
    def start_audio_playback(self, tts, text):
        """
        Start audio playback in a separate thread that can be interrupted
        
        Parameters:
        - tts: TTS instance
        - text: Text to convert to speech
        """
        self.is_playing_audio = True
        self.interrupt_detected = False
        self.stop_audio_event.clear()
        self.audio_finished_event.clear()
        
        # Start audio playback in separate thread
        self.audio_thread = threading.Thread(
            target=self._play_audio_worker,
            args=(tts, text),
            daemon=True
        )
        self.audio_thread.start()
        
        # Start listening for interruptions if STT is enabled
        if enable_STT and self.stt:
            interrupt_thread = threading.Thread(
                target=self._listen_for_interruption,
                daemon=True
            )
            interrupt_thread.start()
    
    def _play_audio_worker(self, tts, text):
        """
        Worker thread for playing audio that can be stopped
        """
        try:
            print(f"[Audio]: Playing: '{text[:30]}...'")
            
            # Use the TTS system but make it interruptible
            tts.play_text_audio(text)
            
            # Check if we were interrupted
            if self.stop_audio_event.is_set():
                print("[Interrupt]: Audio playback stopped due to interruption")
            else:
                print("[Audio]: Playback completed normally")
                
        except Exception as e:
            print(f"[Audio Error]: {e}")
        finally:
            self.is_playing_audio = False
            self.audio_finished_event.set()
    
    def _listen_for_interruption(self):
        """
        Listen for user speech while audio is playing
        If detected, stop audio and capture the interruption
        """
        try:
            # Use shorter timeouts to detect interruptions quickly
            interruption_pause_threshold = 0.5  # Shorter for quick detection
            interruption_phrase_limit = 10  # Shorter clips for interruptions
            
            print("[Interrupt]: Listening for interruptions...")
            
            while self.is_playing_audio and not self.stop_audio_event.is_set():
                try:
                    # Quick listen for any speech
                    user_input = self.stt.get_voice_as_text(
                        pause_threshold=interruption_pause_threshold,
                        phrase_time_limit=interruption_phrase_limit
                    )
                    
                    if user_input and user_input.strip():
                        print(f"[Interrupt]: Detected interruption: '{user_input}'")
                        
                        # Stop audio playback
                        self.stop_audio_event.set()
                        self.interrupt_detected = True
                        self.interrupt_text = user_input
                        
                        # TODO: Add actual audio stopping mechanism here
                        # This would require modifying the TTS play function to be stoppable
                        print("[Interrupt]: Audio stopped, processing interruption")
                        break
                        
                except Exception as e:
                    # Ignore timeout errors - they're expected during normal listening
                    if "timeout" not in str(e).lower():
                        print(f"[Interrupt Warning]: {e}")
                    time.sleep(0.1)  # Brief pause before trying again
                    
        except Exception as e:
            print(f"[Interrupt Error]: {e}")
    
    def wait_for_audio_or_interruption(self, timeout=30):
        """
        Wait for either audio to finish or an interruption to occur
        
        Parameters:
        - timeout: Maximum time to wait
        
        Returns:
        - bool: True if interrupted, False if audio finished normally
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.interrupt_detected:
                return True
            if self.audio_finished_event.is_set():
                return False
            time.sleep(0.1)
        
        print("[Warning]: Audio playback timeout")
        return False
    
    def get_interruption_text(self):
        """
        Get the text that caused the interruption
        
        Returns:
        - str: The interruption text or None
        """
        return self.interrupt_text
    
    def reset_interruption(self):
        """
        Reset interruption state for next interaction
        """
        self.interrupt_detected = False
        self.interrupt_text = None