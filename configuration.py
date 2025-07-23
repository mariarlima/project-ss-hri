# For testing with/without voice and/or robot 
enable_TTS = True
enable_blossom = True
enable_STT = True

delay_time = 3 # Start playing motion sequence after X seconds
mic_time_offset = 0.01 # Time offset for mic to start recording in seconds
pause_threshold = 2 # Only stop recording after X seconds of silence
phrase_time_limit = 5 # Max duration of audio clip
session_time_limit = 1.5 * 60 # min

# adapt the prompt to the interactive cognitive task context
prompt = [
    {
        "role": "system",
        "content": (
            "You are a friendly and patient companion designed to chat verbally with elderly people."
            "Speak clearly, use simple language, and be encouraging."
            "Keep your replies short (max 2 lines), warm, and easy to understand."
            "Tone Encourage participants to keep engaging with supportive comments and short hints. Use friendly language. Dont use emojis."
            "Start Greet the user. Ask if they are ready to play a game. Introduce the task with describe the image (Cookie theft)."
        )
    }
]

_image_path = "/home/pedrodias/Documents/git-repos/project-ss-hri/Cookie-Theft-Picture-4_W640.jpg"

whisper_model_id = "tiny.en"

TTS_config = {
    "unrealspeech":
    {
        "voice_id": "Liv",
        "bit_rate": "192k",
        "speed": -0.25,
        "pitch": 1.09,
    },
    "openai":
    {
        "model_id": "tts-1",
        "voice_id": "alloy"
    },
    "aws":
    {
        "voice_id": "Penelope",
        "speed": "80%",
        "pitch": "+0%",
    }
}

sequence_metadata = {
    "sequence_length_boundary_list": {"prompt": [2, 3, 4, 5, 6, 7, 8, 9, 12, 17, 20]},
    "sequence_list":
    {
        "start": ["cognitive/intro_01", "cognitive/intro_02", "cognitive/intro_03"],
        "prompt": ["cognitive/encouragement_01", "cognitive/encouragement_02", "cognitive/encouragement_03",
                   "cognitive/encouragement_04", "cognitive/encouragement_05",
                   "cognitive/encouragement_06"],
        "prompt_timed":
        [
            ["happy/happy_1_109", "happy/happy_2_109", "happy/happy_5_109", "happy/happy_8_109",
             "happy/happy_9_109"],
            ["yes", "fear/fear_startled"],
            ["happy/happy_nodding", "happy/happy",
             "anger/anger_dissapoint"],
            ["grand/grand4", "sesame/sesame12", "fear/fear"],
            ["cognitive/encouragement_04", "happy/happy_head_bobbing",
             "fear/fear_looking_around_1"],
            ["cognitive/encouragement_01", "happy/happy_daydream"],
            ["cognitive/encouragement_03",
             "cognitive/encouragement_05"],
            ["cognitive/encouragement_06", "cognitive/encouragement_02", "happy/happy_20181204_122044",
             "happy/happy_20181204_120338"],
            ["cognitive/end_01", "cognitive/end_02"],
            ["cognitive/end_03", "cognitive/intro_03"],
            ["cognitive/intro_01", "cognitive/intro_02"]
        ],
        "end": ["cognitive/end_01", "cognitive/end_02", "cognitive/end_03"],
    },
}

