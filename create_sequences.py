import sys
sys.path.append('Blossom-Controller')
from blossom_sequence_comb import combine_sequences

import os
output_sequence_dir = "./Blossom-Controller/blossom-public/blossompy/src/sequences/woody/cognitive/"
os.makedirs(output_sequence_dir, exist_ok=True)
sequence_dir = "./Blossom-Controller/blossom-public/blossompy/src/sequences/woody/"

# If this throws error, create cognitive directory under corresponding directory.
combine_sequences(["grand/grand1", "anger/anger_0_109", "yes", "anger/anger_scream"], "01", sequence_dir, output_sequence_dir)

combine_sequences(["amazon_demo/Introduction_NameAsking", "amazon_demo/Introduction_MakeFriends_Yes"], "intro_01", sequence_dir, output_sequence_dir)
combine_sequences(["amazon_demo/Introduction_MakeFriends", "amazon_demo/Personalization_BecomingFriends"], "intro_02", sequence_dir, output_sequence_dir)
combine_sequences(["amazon_demo/Introduction_MakeFriends_Yes", "amazon_demo/Personalization_BecomingFriends"],
                  "intro_03", sequence_dir, output_sequence_dir)

combine_sequences(["happy/happy_nodding", "happy/happy_8_109"], "encouragement_01", sequence_dir, output_sequence_dir)
combine_sequences(["happy/happy_20181204_130211"], "encouragement_02", sequence_dir, output_sequence_dir)
combine_sequences(["yes", "sesame/sesame10"], "encouragement_03", sequence_dir, output_sequence_dir)
combine_sequences(["happy/happy_8_109", "yes"], "encouragement_04", sequence_dir, output_sequence_dir)
combine_sequences(["happy/happy", "sesame/sesame10"], "encouragement_05", sequence_dir, output_sequence_dir)
combine_sequences(["sesame/sesame10", "happy/happy_nodding"], "encouragement_06", sequence_dir, output_sequence_dir)

combine_sequences(["grand/grand12", "happy/happy_20181204_120338"], "end_01", sequence_dir, output_sequence_dir)
combine_sequences(["happy/happy_daydream", "grand/grand8"], "end_02", sequence_dir, output_sequence_dir)
combine_sequences(["grand/grand4", "happy/happy", "grand/grand10", "happy/happy"], "end_03", sequence_dir, output_sequence_dir)
combine_sequences(["grand/grand4", "cognitive/encouragement_04"], "extra_01", sequence_dir, output_sequence_dir)
combine_sequences(["sesame/sesame10", "grand/grand12"], "extra_02", sequence_dir, output_sequence_dir)
combine_sequences(["happy/happy_8_109", "happy/happy_nodding"], "extra_03", sequence_dir, output_sequence_dir)
combine_sequences(["cognitive/extra_01", "happy/happy_nodding"], "extra_04", sequence_dir, output_sequence_dir)
combine_sequences(["cognitive/extra_02", "happy/happy_8_109"], "extra_05", sequence_dir, output_sequence_dir)
# new updated priming/intro sequence
combine_sequences(["cognitive/extra_01", "anger/anger_dissapoint", "sesame/sesame10", "yes"], "extra_06", sequence_dir, output_sequence_dir)
combine_sequences(["happy/happy_2_109", "happy/happy_8_109"], "extra_07", sequence_dir, output_sequence_dir)
combine_sequences(["grand/grand4", "yes"], "extra_08", sequence_dir, output_sequence_dir)
combine_sequences(["happy/happy", "happy/happy_9_109", "sad/sad_head_down"], "extra_09", sequence_dir, output_sequence_dir)
combine_sequences(["sesame/sesame12", "fear/fear", "happy/happy_1_109", "fear/fear"], "extra_10", sequence_dir, output_sequence_dir)
combine_sequences(["grand/grand4", "happy/happy_2_109"], "extra_11", sequence_dir, output_sequence_dir)
combine_sequences(["happy/happy_nodding", "fear/fear_startled"], "extra_12", sequence_dir, output_sequence_dir)
combine_sequences(["happy/happy_nodding", "anger/anger_dissapoint", "sesame/sesame10", "yes"], "extra_13", sequence_dir, output_sequence_dir)
combine_sequences(["fear/fear", "cognitive/encouragement_04"], "extra_14", sequence_dir, output_sequence_dir)


