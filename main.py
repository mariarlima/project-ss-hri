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

    # Check time limit
    elapsed = time.perf_counter() - start_time
    if elapsed >= session_time_limit:
        print("\n[System]: Session time limit reached.")
        end_interaction = True

    if end_interaction:
        break

