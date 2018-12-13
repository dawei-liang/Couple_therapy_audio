"""
Created on Wed Dec 12 2018
@author: david

"""

import os
import subprocess

import check_dirs

dir_to_save = './wav_files/'
dir_to_load_mp4 = './T1_Audio_Files/'

check_dirs.check_dir(dir_to_save)
mp4_list = [x for x in os.listdir(dir_to_load_mp4) if x.endswith('.mp4')]   # Load mp4 files
print('Number of audio files:', len(mp4_list))


for item in mp4_list:
    # Format: 16-bit depth, sampling rate: from source
    command = "ffmpeg -i " + dir_to_load_mp4 + item + " -vn -sample_fmt s16 " + dir_to_save + item.strip('.mp4') + ".wav"
    subprocess.call(command, shell=True)

print('Done')



