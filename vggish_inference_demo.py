# -*- coding: utf-8 -*-
"""
Modified on Tue Dec 27 17:20:00 2018

@Adopted from Google VGGish Github: https://github.com/tensorflow/models/tree/master/research/audioset
    modifier: david, research use only
"""

# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.
This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.
A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).
Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file
  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params
  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

#from __future__ import print_function

import numpy as np
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

class vggish:
    def __init__(self, dir_to_load, dir_to_save):
        self.dir_to_load = dir_to_load
        self.dir_to_save = dir_to_save
        
    def del_all_flags(self, FLAGS):
        self.FLAGS = FLAGS
        flags_dict = FLAGS._flags()
        keys_list = [keys for keys in flags_dict]
        for keys in keys_list:
            self.FLAGS.__delattr__(keys)
            
    def set_all_flags(self):
        self.del_all_flags(tf.flags.FLAGS)
        
        flags = tf.app.flags
        
        #dir_to_load = './wav_files/0001_T1_PS'
        #dir_to_save = './0001_T1_PS'
        
        flags.DEFINE_string(   # Set the input wav file path
            'wav_file', self.dir_to_load + '.wav',
            'Path to a wav file. Should contain signed 16-bit PCM samples. '
            'If none is provided, a synthetic sound is used.')
        
        flags.DEFINE_string(
            'checkpoint', 'vggish_model.ckpt',
            'Path to the VGGish checkpoint file.')
        
        flags.DEFINE_string(
            'pca_params', 'vggish_pca_params.npz',
            'Path to the VGGish PCA parameters file.')
        
        flags.DEFINE_string(   # Set the output tfrecords path
            'tfrecord_file', self.dir_to_save + '.tfrecords',
            'Path to a TFRecord file where embeddings will be written.')
        
        self.FLAGS = flags.FLAGS
    
    
    def vggish_main(self):
      # In this simple example, we run the examples from a single audio file through the model. 
      print ('hahaha')
      wav_file = self.FLAGS.wav_file 
      print(wav_file)   ###
    
      examples_batch = vggish_input.wavfile_to_examples(wav_file)
      print(examples_batch)
    
      # Prepare a postprocessor to munge the model embeddings.
      pproc = vggish_postprocess.Postprocessor(self.FLAGS.pca_params)
    
      # If needed, prepare a record writer to store the postprocessed embeddings.
      writer = tf.python_io.TFRecordWriter(
          self.FLAGS.tfrecord_file) if self.FLAGS.tfrecord_file else None
    
      with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, self.FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
    
        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        print(embedding_batch)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        print(postprocessed_batch)
    
        # Write the postprocessed embeddings as a SequenceExample, in a similar
        # format as the features released in AudioSet. Each row of the batch of
        # embeddings corresponds to roughly a second of audio (96 10ms frames), and
        # the rows are written as a sequence of bytes-valued features, where each
        # feature value contains the 128 bytes of the whitened quantized embedding.
        seq_example = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
                    #'c':
                        tf.train.FeatureList(
                            feature=[
                                    tf.train.Feature(
                                    bytes_list=tf.train.BytesList(value=[embedding.tobytes()]))
                                for embedding in postprocessed_batch
                            ]
                        )
                }
            )
        )
        print(seq_example)
        if writer:
          writer.write(seq_example.SerializeToString())
    
      if writer:
        writer.close()
        
    
    
    def run(self):
      tf.reset_default_graph()
      tf.app.run(self.vggish_main())
      tf.Session.close() 


