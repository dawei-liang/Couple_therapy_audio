#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 22:47:15 2018

@author: dawei
"""

import os

dir_to_load_mp4 = './T1_Audio_Files/'
mp4_list = [x for x in os.listdir(dir_to_load_mp4) if x.endswith('.mp4')]   
print('Number of MP4 files:', len(mp4_list))
mp4_list.sort()

dir_to_load_csv = './csv_files/'
csv_list = [x for x in os.listdir(dir_to_load_csv) if x.endswith('.csv')]   
print('Number of CSV files:', len(csv_list))
csv_list.sort()

count = 0
for i in range(len(mp4_list)):
    if mp4_list[i].split('.')[0] == csv_list[i].split('.')[0]:
        count += 1
print('Same:', count)