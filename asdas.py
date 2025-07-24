import sys
from pathlib import Path
controller_path = (Path(Path.cwd()) / "Blossom-Controller").resolve()
sys.path.insert(0, str(controller_path)) 
from blossom_wrapper import BlossomWrapper

bl = BlossomWrapper()

bl.reset()
bl.do_sequence("fear/fear_startled")

# bl.do_prompt_sequence_matching(
#     delay_time=0.5, audio_length=5)