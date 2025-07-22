import os
import queue
import threading
import time
import signal
import sys
from dotenv import load_dotenv

# Import your existing modules
from STT import STT
from LLM import LLM  # Use your enhanced LLM.py with auto-evaluation
from TTS import TTS
from blossom_wrapper import BlossomWrapper
from conversation_logger import ConversationLogger, LoggedConversationSession

from configuration import (
    enable_TTS,
    enable_STT,
    enable_blossom,
    prompt,
    session_time_limit,
    phrase_time_limit,
    pause_threshold,
    mic_time_offset
)

# Load environment variables
load_dotenv()

def main():
    """
    Enhanced main function with comprehensive logging integration
    Keeps all original functionality while adding detailed logging
    """
    
    # Use the context manager for automatic logging
    with LoggedConversationSession() as logger:
        
        try:
            # Log system initialization start
            logger.log_performance_metric("system_init_start", time.time())
            
            # Initialize: the signal queue for TTS audio
            signal_queue = queue.Queue()
            
            # Get API keys with error logging
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                error_msg = "OPENAI_API_KEY environment variable not set. Please check your .env file."
                logger.log_error("CONFIG", error_msg)
                raise EnvironmentError(error_msg)
            
            # Initialize LLM with logging
            llm_init_start = time.time()
            llm = LLM(api_key=openai_api_key, llm_prompt=prompt)
            llm_init_time = time.time() - llm_init_start
            logger.log_performance_metric("llm_init_time", llm_init_time)
            print(f"[Logger]: LLM initialized in {llm_init_time:.2f}s")
            
            # Initialize STT if enabled with logging
            stt = None
            if enable_STT:
                stt_init_start = time.time()
                stt = STT()
                stt_init_time = time.time() - stt_init_start
                logger.log_performance_metric("stt_init_time", stt_init_time)
                print(f"[Logger]: STT initialized in {stt_init_time:.2f}s")
            
            # Initialize TTS if enabled with logging
            tts = None
            if enable_TTS:
                tts_init_start = time.time()
                tts = TTS(api_key=os.getenv("UNREAL_SPEECH_KEY"), signal_queue=signal_queue)
                tts_init_time = time.time() - tts_init_start
                logger.log_performance_metric("tts_init_time", tts_init_time)
                print(f"[Logger]: TTS initialized in {tts_init_time:.2f}s")
            
            # Initialize the BlossomWrapper
            if enable_blossom:
                blossom = BlossomWrapper()

            # Set up graceful shutdown with logging
            def graceful_shutdown(signum, frame):
                print("\n[System]: Shutting down...")
                logger.log_performance_metric("system_shutdown_requested", time.time())
                sys.exit(0)
            
            signal.signal(signal.SIGINT, graceful_shutdown)
            
            # Log session start
            session_start_time = time.perf_counter()
            logger.log_performance_metric("session_start", session_start_time)
            print(f"[System]: Starting conversation session - ID: {logger.get_session_id()}")
            
            # Prompt the model to begin the session with logging
            initial_request_start = time.time()
            initial_response = llm.request_response(text="Start")
            initial_request_time = time.time() - initial_request_start
            
            # Log the initial interaction
            logger.log_interaction(
                user_input="Start",
                assistant_response=initial_response,
                metadata={
                    "type": "session_initialization",
                    "response_time": initial_request_time
                }
            )
            logger.log_performance_metric("initial_llm_response_time", initial_request_time)
            
            # Play intro response if TTS enabled with logging
            if enable_TTS:
                intro_tts_start = time.time()
                tts.play_text_audio(initial_response)
                intro_tts_time = time.time() - intro_tts_start
                
                # Log audio generation
                logger.log_audio(
                    audio_type="output",
                    audio_data=None,  # We don't have raw audio data here
                    text="Starting Exercise",
                    duration=intro_tts_time,
                    is_cached=False
                )
                logger.log_performance_metric("intro_tts_time", intro_tts_time)
            
            # Main interaction loop with comprehensive logging
            end_interaction = False
            interaction_count = 0
            
            print("\n[System]: Conversation started. Begin speaking or typing!")
            
            while not end_interaction:
                try:
                    interaction_count += 1
                    interaction_start_time = time.time()
                    
                    print(f"\n--- Interaction {interaction_count} ---")
                    
                    # Get user input via voice or keyboard with timing
                    user_input = None
                    input_start_time = time.time()
                    
                    if enable_STT:
                        print("[Input]: Listening for your response...")
                        
                        # Log STT attempt
                        logger.log_performance_metric("stt_attempt_start", input_start_time)
                        
                        user_input = stt.get_voice_as_text(
                            pause_threshold=pause_threshold, 
                            phrase_time_limit=phrase_time_limit
                        )
                        
                        input_duration = time.time() - input_start_time
                        logger.log_performance_metric("stt_processing_time", input_duration)
                        
                        # Log STT result
                        if user_input:
                            logger.log_audio(
                                audio_type="input",
                                audio_data=None,  # Raw audio not available in current STT implementation
                                text=user_input,
                                duration=input_duration
                            )
                        else:
                            logger.log_error("STT", "No speech detected or transcription failed")
                    else:
                        user_input = input("\n[Input]: Please type your response: ")
                        input_duration = time.time() - input_start_time
                        logger.log_performance_metric("keyboard_input_time", input_duration)
                    
                    # Validate user input
                    if not user_input:
                        print("[System]: No user input detected.")
                        logger.log_error("INPUT_VALIDATION", "Empty user input received")
                        continue
                    
                    if not user_input.strip():
                        print("[System]: Empty input detected.")
                        logger.log_error("INPUT_VALIDATION", "Whitespace-only input received")
                        continue
                    
                    print(f"[User]: {user_input}")
                    
                    # Request LLM response with detailed timing
                    llm_request_start = time.time()
                    response = llm.request_response(text=user_input)
                    llm_response_time = time.time() - llm_request_start
                    
                    logger.log_performance_metric("llm_response_time", llm_response_time)
                    print(f"[LLM Response]: {response}")
                    print(f"[Timing]: LLM responded in {llm_response_time:.2f}s")
                
                    # Wait for the TTS module to report the audio length
                    try:
                        audio_length = signal_queue.get(timeout=10)
                    except queue.Empty:
                        audio_length = 2  # fallback default

                    # Start Blossom and TTS threads (Blossom has optional delay)
                    robot_th = threading.Thread(
                        target=blossom.do_prompt_sequence_matching,
                        kwargs={
                            "audio_length": audio_length,
                            "delay_time": 2  # delay so robot waits a bit after TTS starts
                        }
                    )
                    robot_th.start()

                    # Start TTS in parallel (if enabled)
                    if enable_TTS:
                        tts_th = threading.Thread(
                            target=tts.play_text_audio,
                            args=(response,)
                        )
                        tts_th.start()

                        # Wait for both to complete
                        tts_th.join()
                        robot_th.join()
                    else:
                        robot_th.join()

                    # Play LLM response with TTS if enabled
                    # tts_duration = 0
                    # if enable_TTS:
                    #     tts_start_time = time.time()
                        
                    #     # Play the audio
                    #     tts.play_text_audio(response)
                        
                    #     # Wait for the audio duration from the TTS module via signal_queue
                    #     try:
                    #         duration = signal_queue.get(timeout=10)  # Wait for duration from TTS
                    #         tts_total_time = time.time() - tts_start_time
                    #         tts_duration = duration
                            
                    #         # Log TTS performance
                    #         logger.log_audio(
                    #             audio_type="output",
                    #             audio_data=None,
                    #             text=response,
                    #             duration=duration,
                    #             is_cached=False  # Original TTS doesn't have caching
                    #         )
                    #         logger.log_performance_metric("tts_generation_time", tts_total_time)
                    #         logger.log_performance_metric("tts_audio_duration", duration)
                            
                    #         print(f"[Timing]: TTS completed in {tts_total_time:.2f}s, audio duration: {duration:.2f}s")
                            
                    #         # Sleep for the audio duration to maintain conversation flow
                    #         time.sleep(duration)
                            
                    #     except queue.Empty:
                    #         warning_msg = "No duration received from TTS within timeout"
                    #         print(f"[Warning]: {warning_msg}")
                    #         logger.log_error("TTS_TIMEOUT", warning_msg)
                    
                    # Calculate total interaction time
                    total_interaction_time = time.time() - interaction_start_time
                    logger.log_performance_metric("total_interaction_time", total_interaction_time)
                    
                    # Log the complete interaction with comprehensive metadata
                    interaction_metadata = {
                        "interaction_number": interaction_count,
                        "input_method": "voice" if enable_STT else "keyboard",
                        "input_duration": input_duration,
                        "llm_response_time": llm_response_time,
                        "tts_enabled": enable_TTS,
                        "tts_duration": tts_duration,
                        "total_time": total_interaction_time,
                        "user_input_length": len(user_input),
                        "response_length": len(response)
                    }
                    
                    logger.log_interaction(user_input, response, interaction_metadata)
                    
                    print(f"[Timing]: Total interaction time: {total_interaction_time:.2f}s")
                    
                    # Check time limit
                    elapsed = time.perf_counter() - session_start_time
                    if elapsed >= session_time_limit:
                        print(f"\n[System]: Session time limit reached ({elapsed:.1f}s)")
                        logger.log_performance_metric("session_ended_by_timeout", elapsed)
                        end_interaction = True
                    
                    # Check for explicit end commands
                    end_keywords = ['goodbye', 'bye', 'end', 'stop', 'quit', 'exit']
                    if user_input and any(word in user_input.lower() for word in end_keywords):
                        print("[System]: End command detected")
                        logger.log_performance_metric("session_ended_by_user", elapsed)
                        logger.log_interaction(
                            user_input="[END_COMMAND_DETECTED]",
                            assistant_response="[SESSION_TERMINATED]",
                            metadata={"trigger_word": user_input, "session_duration": elapsed}
                        )
                        end_interaction = True
                    
                except KeyboardInterrupt:
                    print("\n[System]: Keyboard interrupt received")
                    logger.log_performance_metric("session_ended_by_interrupt", time.perf_counter() - session_start_time)
                    graceful_shutdown(None, None)
                    
                except Exception as e:
                    error_msg = f"Error in main interaction loop: {str(e)}"
                    print(f"[System Error]: {error_msg}")
                    
                    # Log the error with context - safely handle undefined variables
                    error_context = {
                        "interaction_number": interaction_count,
                        "timestamp": time.time()
                    }
                    
                    # Safely add context variables if they exist
                    if 'user_input' in locals() and user_input is not None:
                        error_context["user_input"] = str(user_input)
                    else:
                        error_context["user_input"] = "Not captured"
                    
                    if 'response' in locals():
                        error_context["response"] = str(response)
                    else:
                        error_context["response"] = "Not generated"
                    
                    logger.log_error("MAIN_LOOP_ERROR", error_msg, error_context)
                    
                    print("[System]: Continuing conversation despite error...")
                    continue
            
            # Session ended - calculate final statistics
            final_session_time = time.perf_counter() - session_start_time
            
            print(f"\n[System]: Conversation ended successfully!")
            print(f"[Statistics]: Total session time: {final_session_time:.1f}s")
            print(f"[Statistics]: Total interactions: {interaction_count}")
            print(f"[Statistics]: Average interaction time: {final_session_time/max(1, interaction_count):.1f}s")
            
            # Log final session statistics
            logger.log_performance_metric("final_session_duration", final_session_time)
            logger.log_performance_metric("final_interaction_count", interaction_count)
            logger.log_performance_metric("average_interaction_time", final_session_time/max(1, interaction_count))
            
            print(f"[Logger]: Session log will be saved automatically")
            
        except Exception as e:
            fatal_error_msg = f"Fatal system error: {str(e)}"
            print(f"[System Fatal Error]: {fatal_error_msg}")
            
            # Log fatal error if logger is available
            if 'logger' in locals():
                logger.log_error(
                    "SYSTEM_FATAL",
                    fatal_error_msg,
                    {
                        "timestamp": time.time(),
                        "session_duration": time.perf_counter() - session_start_time if 'session_start_time' in locals() else 0
                    }
                )
            
            raise  # Re-raise the exception after logging

if __name__ == "__main__":
    main()