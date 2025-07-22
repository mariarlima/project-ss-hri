import os
import queue
import threading
import time
from dotenv import load_dotenv
from blossom_wrapper import BlossomWrapper # Added for Blossom robot integration

from STT import STT
from LLM import LLM
from TTS import TTS

from configuration import (
    enable_TTS,
    enable_STT,
    enable_blossom, # added for Blossom robot integration
    prompt,
    session_time_limit,
    phrase_time_limit,
    pause_threshold,
    mic_time_offset
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

# Added for Blossom robot integration
if enable_blossom:
    bl = BlossomWrapper()

bl_thread = None
bl_thread_target = None
bl_thread_kwargs = None

tts_thread = None

# TODO: Prompt the model to begin the session with text="Start"
# TODO: Play intro response if TTS enabled

# Main interaction loop
start_time = time.perf_counter()
end_interaction = False

while True:
    # TODO: Get user input via voice if enable_STT else keyboard
    # TODO: Request LLM response. Consider end_interaction case
    # TODO: Play LLM response with TTS if enabled. Use a thread to play audio asynchronously
    # TODO: Wait for the audio duration from the TTS module via signal_queue for synchronization

    # TODO: If enable_blossom, play a sequence with the Blossom robot
    # TODO: Use do_prompt_sequence_matching() from blossom_wrapper.py to choose motion duration based on audio length. Use audio_length from signal_queue.get()
    # TODO: Ensure you use phrase_time_limit, pause_threshold, mic_time_offset in your STT and TTS implementations


    # Check time limit
    elapsed = time.perf_counter() - start_time
    if elapsed >= session_time_limit:
        print("\n[System]: Session time limit reached.")
        end_interaction = True

    if end_interaction:
        break

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



