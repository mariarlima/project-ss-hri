from pydub import AudioSegment
from io import BytesIO
import requests

def get_audio_length(audio_bytes):
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
    duration_in_milliseconds = len(audio)
    duration_in_seconds = duration_in_milliseconds / 1000
    return duration_in_seconds

def read_mp3_as_bytes(file_path):
    with open(file_path, 'rb') as file:
        audio_bytes = file.read()
    return audio_bytes

def read_mp3_as_bytes_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    audio_bytes = response.content
    return audio_bytes

