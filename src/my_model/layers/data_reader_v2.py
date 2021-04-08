from pathlib import Path
import tensorflow as tf
import functools
from my_model.abc.base import MY_BASE
#from my_model.layers.data_process import NerProcessor
from my_model.layers.base_process import FeatureExamples
from my_model.embeddings.bert_base.bert import tokenization
import pickle
import os
import collections
import six
class InputFn(FeatureExamples):
    def __init__(self,data_dir,tags_path,vocab_path,embeddings_type,tokenizer=None,max_seq_length=512,args=None):
       super().__init__(data_dir,tags_path,vocab_path,embeddings_type=embeddings_type,tokenizer=tokenizer,max_seq_length=max_seq_length,args=args)
    def tf_record_input_fn_builder(self,input_file, seq_length,len_label, is_training, drop_remainder,batch_size):
        if self.embeddings_type == 'bert':
            name_to_features = {
                "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
                "mask": tf.io.FixedLenFeature([seq_length], tf.int64),
                "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
                #"label_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
                "label_ids": tf.io.FixedLenFeature([len_label], tf.int64),
            }
        else:
            name_to_features = {
                "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
                #"input_ids": tf.io.FixedLenFeature([seq_length], tf.float32),
                "mask": tf.io.FixedLenFeature([seq_length], tf.int64),
                #"label_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
                "label_ids": tf.io.FixedLenFeature([len_label], tf.int64),
            }

        def _decode_record(record, name_to_features):
            example = tf.io.parse_single_example(record, name_to_features)
            for name in list(example.keys()):
                t = example[name]
                if t.dtype == tf.int64:
                    #t = tf.to_int32(t)
                    t = tf.cast(t,tf.int32)
                example[name] = t
            return example
    
        def input_fn(params):
            d = tf.data.TFRecordDataset(input_file)
            if is_training:
                d = d.repeat()
                d = d.shuffle(buffer_size=100)
            d = d.apply(tf.data.experimental.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder
            ))
            return d
        return input_fn
    def get_input_fn(self,example_file,batch_size,len_label,is_training,drop_remainder):
        input_fn = self.tf_record_input_fn_builder(
            input_file=example_file,
            seq_length=self.max_seq_length,
            len_label=len_label,
            is_training=is_training,
            drop_remainder=drop_remainder,
            batch_size=batch_size,
            )
        #return input_fn,label_list,len(examples)
        return input_fn
    
