#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 22:07:04 2018

@author: dawei
"""

import os

import check_dirs
import vggish_inference_demo
import tryread

dir_to_save_csv = './csv_files/'
dir_to_save_tf = './tf_records/'
dir_to_load_wav = './wav_files/'

check_dirs.check_dir(dir_to_save_csv)
check_dirs.check_dir(dir_to_save_tf)

if __name__ == '__main__':
    # Load wav files
    wav_list = [x for x in os.listdir(dir_to_load_wav) if x.endswith('.wav')]   
    print('Number of wav files:', len(wav_list))
    # Run the VGGish model
    for wav in wav_list:
        wav = wav.strip('.wav')
        try:
            vggish_object = vggish_inference_demo.vggish(dir_to_load_wav + wav, 
                                                         dir_to_save_tf + wav)
            vggish_object.set_all_flags()
            vggish_object.run()
        except:
            continue
        
    # Load tfrecords files
    tf_list = [x for x in os.listdir(dir_to_save_tf) if x.endswith('.tfrecords')]   
    print('Number of tf files:', len(tf_list))
    # Run the CSV converter
    for tf_file in tf_list:
        tf_file = tf_file.strip('.tfrecords')
        csv_object = tryread.csv_converter(dir_to_save_tf + tf_file,
                                           dir_to_save_csv + tf_file)
        csv_object.transform()
    print('Complete')
    