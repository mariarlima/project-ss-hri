import json
import os
import time
import logging
import hashlib
from datetime import datetime
import threading
from collections import defaultdict

class ConversationLogger:
    """
    Comprehensive logging system for conversational AI interactions
    
    Logs:
    - Full conversation transcripts
    - Audio files and metadata
    - System performance metrics
    - User interaction patterns
    - Error logs and debugging info
    - Contextual variables (emotions, interruptions, etc.)
    """
    
    def __init__(self, session_id=None, log_dir="conversation_logs"):
        # Generate unique session ID if not provided
        self.session_id = session_id or self.generate_session_id()
        
        # Set up logging directories
        self.log_dir = log_dir
        self.audio_dir = os.path.join(log_dir, "audio", self.session_id)
        self.transcript_dir = os.path.join(log_dir, "transcripts")
        self.metrics_dir = os.path.join(log_dir, "metrics")
        
        self.setup_directories()
        
        # Initialize session data
        self.session_data = {
            "session_id": self.session_id,
            "start_time": time.time(),
            "start_datetime": datetime.now().isoformat(),
            "conversation": [],
            "audio_logs": [],
            "system_metrics": {
                "total_interactions": 0,
                "interruptions": 0,
                "tts_cache_hits": 0,
                "tts_cache_misses": 0,
                "llm_evaluation_failures": 0,
                "errors": []
            },
            "contextual_data": {
                "user_emotions": [],
                "response_times": [],
                "audio_durations": [],
                "interruption_patterns": []
            }
        }
        
        # Thread-safe logging
        self.log_lock = threading.Lock()
        
        # Set up Python logging for system errors
        self.setup_system_logging()
        
        print(f"[Logger]: Session started - ID: {self.session_id}")
    
    def generate_session_id(self):
        """Generate unique session ID based on timestamp and random hash"""
        timestamp = str(int(time.time()))
        random_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"session_{timestamp}_{random_hash}"
    
    def setup_directories(self):
        """Create necessary directories for logging"""
        directories = [self.log_dir, self.audio_dir, self.transcript_dir, self.metrics_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def setup_system_logging(self):
        """Set up Python logging for system-level errors and debug info"""
        log_filename = os.path.join(self.log_dir, f"system_{self.session_id}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()  # Also print to console
            ]
        )
        
        self.system_logger = logging.getLogger(f"ConversationSystem_{self.session_id}")
    
    def log_interaction(self, user_input, assistant_response, metadata=None):
        """
        Log a complete interaction between user and assistant
        
        Parameters:
        - user_input: What the user said/typed
        - assistant_response: Assistant's response
        - metadata: Additional context (emotions, timings, etc.)
        """
        with self.log_lock:
            interaction = {
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "interaction_id": len(self.session_data["conversation"]) + 1,
                "user_input": user_input,
                "assistant_response": assistant_response,
                "metadata": metadata or {}
            }
            
            self.session_data["conversation"].append(interaction)
            self.session_data["system_metrics"]["total_interactions"] += 1
            
            # Log to system logger as well
            self.system_logger.info(
                f"Interaction {interaction['interaction_id']}: "
                f"User='{user_input[:50]}...' "
                f"Assistant='{assistant_response[:50]}...'"
            )
    
    def log_audio(self, audio_type, audio_data, text, duration=None, is_cached=False):
        """
        Log audio-related information
        
        Parameters:
        - audio_type: "input" (STT) or "output" (TTS)
        - audio_data: Binary audio data or file path
        - text: Transcribed/generated text
        - duration: Audio duration in seconds
        - is_cached: Whether TTS used cached audio
        """
        with self.log_lock:
            audio_log = {
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "type": audio_type,
                "text": text,
                "duration": duration,
                "is_cached": is_cached,
                "file_path": None
            }
            
            # Save audio file if data provided
            if audio_data and isinstance(audio_data, bytes):
                audio_filename = f"{audio_type}_{int(time.time())}_{len(self.session_data['audio_logs'])}.wav"
                audio_path = os.path.join(self.audio_dir, audio_filename)
                
                try:
                    with open(audio_path, 'wb') as f:
                        f.write(audio_data)
                    audio_log["file_path"] = audio_path
                except Exception as e:
                    self.system_logger.error(f"Failed to save audio file: {e}")
            
            self.session_data["audio_logs"].append(audio_log)
            
            # Update metrics
            if audio_type == "output":  # TTS
                if is_cached:
                    self.session_data["system_metrics"]["tts_cache_hits"] += 1
                else:
                    self.session_data["system_metrics"]["tts_cache_misses"] += 1
                    
                if duration:
                    self.session_data["contextual_data"]["audio_durations"].append(duration)
    
    def log_interruption(self, interruption_text, original_response, timestamp_of_interruption):
        """
        Log when user interrupts the assistant
        
        Parameters:
        - interruption_text: What the user said to interrupt
        - original_response: The response that was interrupted
        - timestamp_of_interruption: When the interruption occurred
        """
        with self.log_lock:
            interruption = {
                "timestamp": timestamp_of_interruption,
                "datetime": datetime.fromtimestamp(timestamp_of_interruption).isoformat(),
                "interruption_text": interruption_text,
                "original_response": original_response,
                "interaction_id": len(self.session_data["conversation"])
            }
            
            self.session_data["contextual_data"]["interruption_patterns"].append(interruption)
            self.session_data["system_metrics"]["interruptions"] += 1
            
            self.system_logger.info(f"Interruption detected: '{interruption_text}'")
    
    def log_llm_evaluation(self, original_response, evaluation_result, fallback_used=False):
        """
        Log LLM response evaluation results
        
        Parameters:
        - original_response: The original LLM response
        - evaluation_result: Result from auto-evaluation
        - fallback_used: Whether a fallback response was used
        """
        with self.log_lock:
            evaluation = {
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "original_response": original_response,
                "evaluation_passed": evaluation_result[0],
                "evaluation_reason": evaluation_result[1],
                "fallback_used": fallback_used
            }
            
            # Store in session data (you might want to add an evaluations list)
            if "evaluations" not in self.session_data:
                self.session_data["evaluations"] = []
            self.session_data["evaluations"].append(evaluation)
            
            if not evaluation_result[0]:
                self.session_data["system_metrics"]["llm_evaluation_failures"] += 1
                self.system_logger.warning(
                    f"LLM evaluation failed: {evaluation_result[1]} - "
                    f"Original: '{original_response[:50]}...'"
                )
    
    def log_performance_metric(self, metric_name, value, context=None):
        """
        Log performance metrics (response times, error rates, etc.)
        
        Parameters:
        - metric_name: Name of the metric
        - value: Metric value
        - context: Additional context about the metric
        """
        with self.log_lock:
            metric = {
                "timestamp": time.time(),
                "metric_name": metric_name,
                "value": value,
                "context": context or {}
            }
            
            if "performance_metrics" not in self.session_data:
                self.session_data["performance_metrics"] = []
            self.session_data["performance_metrics"].append(metric)
            
            # Also track response times specifically
            if metric_name == "response_time":
                self.session_data["contextual_data"]["response_times"].append(value)
    
    def log_error(self, error_type, error_message, context=None):
        """
        Log system errors
        
        Parameters:
        - error_type: Type of error (STT, TTS, LLM, etc.)
        - error_message: Error description
        - context: Additional context about the error
        """
        with self.log_lock:
            error = {
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {}
            }
            
            self.session_data["system_metrics"]["errors"].append(error)
            self.system_logger.error(f"{error_type}: {error_message}")
    
    def log_user_emotion(self, emotion, confidence=None, source="text_analysis"):
        """
        Log detected user emotions (if you have emotion detection)
        
        Parameters:
        - emotion: Detected emotion
        - confidence: Confidence score
        - source: How the emotion was detected
        """
        with self.log_lock:
            emotion_log = {
                "timestamp": time.time(),
                "emotion": emotion,
                "confidence": confidence,
                "source": source
            }
            
            self.session_data["contextual_data"]["user_emotions"].append(emotion_log)
    
    def save_session_log(self):
        """
        Save complete session log to file
        Should be called at the end of each session
        """
        with self.log_lock:
            # Add session end information
            self.session_data["end_time"] = time.time()
            self.session_data["end_datetime"] = datetime.now().isoformat()
            self.session_data["session_duration"] = (
                self.session_data["end_time"] - self.session_data["start_time"]
            )
            
            # Calculate summary statistics
            self.session_data["summary"] = self.generate_session_summary()
            
            # Save to JSON file
            session_filename = os.path.join(
                self.transcript_dir, 
                f"session_{self.session_id}.json"
            )
            
            try:
                with open(session_filename, 'w', encoding='utf-8') as f:
                    json.dump(self.session_data, f, indent=2, ensure_ascii=False)
                
                print(f"[Logger]: Session log saved to {session_filename}")
                self.system_logger.info(f"Session completed and saved: {session_filename}")
                
            except Exception as e:
                self.system_logger.error(f"Failed to save session log: {e}")
    
    def generate_session_summary(self):
        """Generate summary statistics for the session"""
        conversation = self.session_data["conversation"]
        metrics = self.session_data["system_metrics"]
        contextual = self.session_data["contextual_data"]
        
        summary = {
            "total_interactions": len(conversation),
            "session_duration_minutes": (
                self.session_data.get("session_duration", 0) / 60
            ),
            "interruption_rate": (
                metrics["interruptions"] / max(1, metrics["total_interactions"])
            ),
            "cache_hit_rate": (
                metrics["tts_cache_hits"] / 
                max(1, metrics["tts_cache_hits"] + metrics["tts_cache_misses"])
            ),
            "evaluation_failure_rate": (
                metrics["llm_evaluation_failures"] / max(1, metrics["total_interactions"])
            ),
            "average_response_time": (
                sum(contextual["response_times"]) / max(1, len(contextual["response_times"]))
                if contextual["response_times"] else 0
            ),
            "average_audio_duration": (
                sum(contextual["audio_durations"]) / max(1, len(contextual["audio_durations"]))
                if contextual["audio_durations"] else 0
            ),
            "error_count": len(metrics["errors"]),
            "most_common_emotions": self.get_emotion_summary(contextual["user_emotions"])
        }
        
        return summary
    
    def get_emotion_summary(self, emotions):
        """Get summary of detected emotions"""
        if not emotions:
            return {}
            
        emotion_counts = defaultdict(int)
        for emotion_log in emotions:
            emotion_counts[emotion_log["emotion"]] += 1
            
        return dict(emotion_counts)
    
    def get_session_id(self):
        """Get the current session ID"""
        return self.session_id

# Example usage and integration helper
def create_logger_for_session():
    """
    Helper function to create and return a logger instance
    Call this at the start of your main.py
    """
    return ConversationLogger()

# Context manager for automatic session logging
class LoggedConversationSession:
    """
    Context manager that automatically handles session logging
    
    Usage:
    with LoggedConversationSession() as logger:
        # Your conversation code here
        logger.log_interaction(user_input, response)
    """
    
    def __init__(self):
        self.logger = None
    
    def __enter__(self):
        self.logger = ConversationLogger()
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger:
            if exc_type:
                self.logger.log_error(
                    "SESSION_ERROR",
                    f"{exc_type.__name__}: {exc_val}",
                    {"traceback": str(exc_tb)}
                )
            self.logger.save_session_log()