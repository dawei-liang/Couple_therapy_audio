# -*- coding: utf-8 -*-
"""
Modified on Tue Dec 27 17:20:00 2018

ADOPTED FROM: https://stackoverflow.com/questions/42703849/audioset-and-tensorflow-understanding

@modifier: david
"""


import tensorflow as tf
import csv
import numpy as np

class csv_converter:
    def __init__(self, root_dir, save_dir):
        self.root_dir = root_dir
        self.save_dir = save_dir
        
        self.audio_record = self.root_dir + '.tfrecords'
        self.vid_ids = []
        self.labels = []
        self.start_time_seconds = [] # in secondes
        self.end_time_seconds = []
        self.feat_audio = []
        self.count = 0
    
    def transform(self):
        for example in tf.python_io.tf_record_iterator(self.audio_record):
            tf_example = tf.train.Example.FromString(example)
            print(tf_example)
        #    vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))
        #    labels.append(tf_example.features.feature['labels'].int64_list.value)
        #    start_time_seconds.append(tf_example.features.feature['start_time_seconds'].float_list.value)
        #    end_time_seconds.append(tf_example.features.feature['end_time_seconds'].float_list.value)
        
            tf_seq_example = tf.train.SequenceExample.FromString(example)
            n_frames = len(tf_seq_example.feature_lists.feature_list['audio_embedding'].feature)
        
            sess = tf.InteractiveSession()
            audio_frame = []
            
            rows = tf.cast(tf.decode_raw(
                    tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[0].bytes_list.value[0],tf.uint8)
                              ,tf.float32).eval()   # the first row
            # iterate through frames
            for i in range(1, n_frames):
                each_row = tf.cast(tf.decode_raw(
                        tf_seq_example.feature_lists.feature_list['audio_embedding'].feature[i].bytes_list.value[0],tf.uint8)
                               ,tf.float32).eval()   # Obtain each row of features, 1*128, numpy array
                audio_frame.append(each_row)
                rows = np.vstack((rows,each_row))
        
            sess.close()
            self.feat_audio.append([])
        
            self.feat_audio[self.count].append(audio_frame) #此处 feat_audio = audio_frame
            self.count+=1
            
        print(each_row)
        #print(feat_audio)
        
        with open(self.save_dir + '.csv','w') as f:
            wr = csv.writer(f,lineterminator='\n')
            for i in range(n_frames):
                wr.writerow(rows[i,:])
        f.close()
        
