import os
import queue
import threading
import time
from dotenv import load_dotenv

from STT import STT
from LLM import LLM
from TTS import TTS

from configuration import (
    enable_TTS,
    enable_STT,
    enable_blossom,
    prompt,
    session_time_limit,
    phrase_time_limit,
    pause_threshold,
    mic_time_offset,
    image_path
)

# TODO: Add API keys in a `.env` file in the project root directory.
load_dotenv()

# Initialize: the signal queue for TTS audio; LLM with key and structured prompt; STT and TTS classes
signal_queue = queue.Queue()

llm = LLM(os.getenv("OPENAI_API_KEY"), llm_prompt=prompt)

if enable_STT:
    stt = STT()

if enable_TTS:
    tts = TTS(os.getenv("UNREAL_SPEECH_KEY"), signal_queue)

# ADDED FOR BLOSSOM CONTROLLER
import sys
from pathlib import Path
controller_path = (Path(Path.cwd()) / "Blossom-Controller").resolve()
sys.path.insert(0, str(controller_path)) 
from blossom_wrapper import BlossomWrapper
if enable_blossom:
    bl = BlossomWrapper()
bl_thread = None
bl_thread_target = None
bl_thread_kwargs = None
tts_thread = None
start_time = time.time()
user_input_text = ""

# Prompt the model to begin the session with text="Start"
llm_response_text = llm.request_response(text="Start the interaction, greet the user, and tell him to start describing the image.", 
                                        image_path=image_path)
print(llm_response_text)

# Play intro response if TTS enabled
if enable_TTS:
        tts_thread = threading.Thread(target=tts.play_text_audio, args=(llm_response_text,))
        tts_thread.start()
        intro_audio_length = signal_queue.get()  # Consume signal here, keep queue empty.

# handle Blossom activation
if enable_blossom:
        if not enable_TTS:
                intro_audio_length = 0
        bl_thread = threading.Thread(target=bl.do_prompt_sequence_matching, args=(),
                                        kwargs={"audio_length": intro_audio_length})
        bl_thread.start()

if enable_STT:
    intro_audio_length = 0.5  # Set a safe default at the start
    time.sleep(mic_time_offset)
else:
    time.sleep(0)

if enable_blossom:
        bl.reset()  # Cutoff Blossom's movement after audio ends


# Main interaction loop
start_time = time.perf_counter()
end_interaction = False
role = "user"

while True:
    stt_response = None

    if enable_STT:
        print("Listening for user input...")
        # listen to user
        stt_response = stt.get_voice_as_text(
            phrase_time_limit=phrase_time_limit,
            pause_threshold=pause_threshold)
        print("--->After get_voice_as_text")

    # trigger random behaviour Blossom (prompt)
    if enable_blossom:
        bl_thread_target = bl.do_prompt_sequence_matching
        bl_thread_kwargs = {"audio_length": 0}

    if enable_STT:
        # Fix: stt_response is a string, not a dict
        if stt_response:  # Just check if it's not None/empty
            user_input_text = stt_response  # It's already the text
            print(f"User said: {user_input_text}")
        else:
            print("No speech detected or transcription failed")
            user_input_text = None
    else:
        # Handle non-STT case
        user_input_text = input("Type your input: ")


    if end_interaction:
        system_message = "System hint: This session reached the time limit, end the conversation in a nice way."
        llm_response_text = llm.request_response(user_input_text, role, system_message)
        print(f"LLM: {llm_response_text}")
    else:
        # LLM process the user input for next interaction turn
        llm_response_text = llm.request_response(user_input_text, 
                                                 role, 
                                                 addition_system_message="Please tell if the user missed something important around the description he gave you. Also tell if it repeat something. Also if the user mentions something that is in the image, tell him to describe things around that. Give him some hints. Always reply with a question and be positive.")
        print(f"LLM: {llm_response_text}")

    # TTS audio response
    if enable_TTS:
        tts_thread = threading.Thread(target=tts.play_text_audio, args=(llm_response_text,))
        tts_thread.start()
        audio_length = signal_queue.get()  # wait for TTS audio to load
        print(f"---> Audio length passed to Blossom: {audio_length}")
    else:
        audio_length = 0.05
         
    if enable_blossom:
        print(f"---> Passing audio_length={audio_length} to Blossom") 
        bl_thread = threading.Thread(
            target=bl.do_prompt_sequence_matching,
            # Delay for sync
            kwargs={"audio_length": audio_length, "delay_time": 2.5}
        )
        bl_thread.start()
    
    time.sleep(mic_time_offset) # for sync
    
    if enable_blossom:
        bl.reset()  # Cutoff Blossom's movement after audio ends
    print("Main thread wakes up.")

    if end_interaction:
        print("Existing Interaction Loop")
        break

    # Check timing
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.3f}, Time Limit: {session_time_limit}")
    if elapsed >= session_time_limit:
        print("Time limit hit. Ending interaction in next round.")
        end_interaction = True

# *** TEST CODE WITH STT/TSS ***
# while True:
#     # TODO: Get user input via voice or keyboard
#     if enable_STT:
#         stt_result = stt.get_voice_as_text(
#             pause_threshold=pause_threshold,
#             phrase_time_limit=phrase_time_limit
#         )
#         if stt_result["success"]:
#             user_input = stt_result["transcription"]["text"]
#         else:
#             print(f"[STT Error] {stt_result['error']}")
#             continue
#     else:
#         user_input = input("\n[You]: ").strip()
#         if not user_input:
#             print("Empty input. Exiting.")
#             break

#     # TODO: Request LLM response
#     if end_interaction:
#         system_msg = "System hint: Time limit reached. Please end the conversation politely."
#         llm_response = llm.request_response(user_input, addition_system_message=system_msg)
#     else:
#         llm_response = llm.request_response(user_input)

#     print(f"\n[Blossom]: {llm_response}")

#     # TODO: Play LLM response with TTS if enabled
#     if enable_TTS:
#         tts_thread = threading.Thread(target=tts.play_text_audio, args=(llm_response,))
#         tts_thread.start()
#         _ = signal_queue.get()

#     # Check time limit
#     elapsed = time.perf_counter() - start_time
#     if elapsed >= session_time_limit:
#         print("\n[System]: Session time limit reached.")
#         end_interaction = True

#     if end_interaction:
#         break

# *** TEST CODE WITHOUT STT/TSS ***
# while True:
#     user_input = input("You: ")

#     if not user_input.strip():
#         print("Empty input. Exiting.")
#         break

#     llm_response = llm.request_response(user_input)
#     print(f"Blossom: {llm_response}")

#     elapsed = time.perf_counter() - start_time
#     if elapsed >= session_time_limit:
#         print("\n[System]: Time limit reached. Ending interaction.")
#         break

