from my_model.abc.model_abc import ABC_MODEL 
#import functools
import json
import logging
import sys
from my_model.embeddings.embedding import embedding_lookup,embedding
import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1
import os
from pathlib import Path
from my_model.abc.model_abc import ABC_MODEL
from my_model.abc.eval_abc import ABC_EVAL
from my_model.abc.export_abc import ABC_EXPORT
class BILSTM_CRF_NER_MODEL(ABC_MODEL):
    def __init__(self,args):
       #if model:
       #    rasie ValueError('model shoud be specified train or eval')
       super().__init__(args) 
       self.args = args

       if not self.args.vocab_words :
           self.args.vocab_words = os.path.join(self.args.data_dir,'vocab.words.txt')
       self.logging.info(args)
       if not self.args.vocab_tags :
           self.args.vocab_tags = os.path.join(self.args.data_dir,'vocab.tags.txt')
       if not self.args.vocab_chars :
           self.args.vocab_chars = os.path.join(self.args.data_dir,'vocab.chars.txt')
       if not self.args.glove :
           self.args.glove = os.path.join(self.args.data_dir,'glove.npz')
    #@staticmethod
    def model_fn_builder(self,args):
        # just for transfer args.
        if not args.vocab_words :
            args.vocab_words = os.path.join(args.eval_data_dir,'vocab.words.txt')
        if not args.vocab_tags :
            args.vocab_tags = os.path.join(args.eval_data_dir,'vocab.tags.txt')
        if not args.vocab_chars :
            args.vocab_chars = os.path.join(args.eval_data_dir,'vocab.chars.txt')
        if not args.glove :
            args.glove = os.path.join(args.eval_data_dir,'glove.npz')
        def model_fn(features, labels, mode, params):
            # For serving, features are a bit different
            if isinstance(features, dict):
                features = features['words'], features['nwords']
        
            # Read vocabs and inputs
            dropout = args.dropout
            words, nwords = features
            #tf.print(' '.join(words[4]), output_stream=sys.stderr)
            training = (mode == tf.estimator.ModeKeys.TRAIN)
            vocab_words = tf.contrib.lookup.index_table_from_file(
                #args.vocab_words)
                args.vocab_words, num_oov_buckets=args.num_oov_buckets)
            with Path(args.vocab_tags).open() as f:
                indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
                num_tags = len(indices) + 1
        
            word_ids = vocab_words.lookup(words)
            if args.embedding == 'word2id':
        	    # word2id
                with Path(args.vocab_words).open(encoding='utf-8') as f:
                    vocab_words_1 = f.readlines()
                    vocab_length = len(vocab_words_1)
                embeddings = embedding(word_ids,vocab_length,args)
                embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)
        
            else:
                # Word Embeddings
                # deafult
                glove = np.load(args.glove)['embeddings']  # np.array
                variable = np.vstack([glove, [[0.]*args.dim]])
                variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
                embeddings = tf.nn.embedding_lookup(variable, word_ids)
                embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)
            # LSTM
            t = tf.transpose(embeddings, perm=[1, 0, 2])
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(args.lstm_size)
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(args.lstm_size)
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
            output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.transpose(output, perm=[1, 0, 2])
            output = tf.layers.dropout(output, rate=dropout, training=training)
        
            # CRF
            logits = tf.layers.dense(output, num_tags)
            crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)
        
            if mode == tf.estimator.ModeKeys.PREDICT:
                # Predictions
                reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
                    args.vocab_tags)
                pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
                predictions = {
                    'pred_ids': pred_ids,
                    'tags': pred_strings
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            else:
                # Loss
                vocab_tags = tf.contrib.lookup.index_table_from_file(args.vocab_tags)
                tags = vocab_tags.lookup(labels)
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    logits, tags, nwords, crf_params)
                loss = tf.reduce_mean(-log_likelihood)
        
                # Metrics
                weights = tf.sequence_mask(nwords)
                metrics = {
                    'acc': tf.metrics.accuracy(tags, pred_ids, weights),
                    'precision': precision(tags, pred_ids, num_tags, indices, weights),
                    'recall': recall(tags, pred_ids, num_tags, indices, weights),
                    'f1': f1(tags, pred_ids, num_tags, indices, weights),
                }
                for metric_name, op in metrics.items():
                    tf.summary.scalar(metric_name, op[1])
        
                if mode == tf.estimator.ModeKeys.EVAL:
                    return tf.estimator.EstimatorSpec(
                        mode, loss=loss, eval_metric_ops=metrics)
        
                elif mode == tf.estimator.ModeKeys.TRAIN:
                    warmup_steps = args.warmup_steps
                    step = tf.to_float(tf.train.get_global_step())
                    if args.learning_rate_decay == 'sqrt':
                        lr_warmup = args.learning_rate_peak * tf.minimum(1.0,step/warmup_steps)
                        lr_decay = args.learning_rate_peak * tf.minimum(1.0,tf.sqrt(warmup_steps/step))
                        lr = tf.where(step < warmup_steps, lr_warmup, lr_decay)
                    elif args.learning_rate_decay == 'exp':
                        lr = tf.train.exponential_decay(args.learning_rate_peak,
                                global_step=step,
                                decay_steps=args.decay_steps,
                                decay_rate=args.decay_rate)
                    else:
                        self.logging.info('learning rate decay strategy not supported')
                        sys.exit()
                    tf.print(lr)
                    train_op = tf.train.AdamOptimizer(lr).minimize(
                        loss, global_step=tf.train.get_or_create_global_step())
                    return tf.estimator.EstimatorSpec(
                        mode, loss=loss, train_op=train_op)
        
        return model_fn


#class bilstmCRF_eval(ABC_EVAL):
#    def __init__(self,args):
#        super().__init__(args)
#    @staticmethod
#    def model_fn_builder(args):
#        if not args.vocab_words :
#            args.vocab_words = os.path.join(args.eval_data_dir,'vocab.words.txt')
#        if not args.vocab_tags :
#            args.vocab_tags = os.path.join(args.eval_data_dir,'vocab.tags.txt')
#        if not args.vocab_chars :
#            args.vocab_chars = os.path.join(args.eval_data_dir,'vocab.chars.txt')
#        if not args.glove :
#            args.glove = os.path.join(args.eval_data_dir,'glove.npz')
#        return BILSTM_CRF_NER_MODEL.model_fn_builder(args)
#    
#class bilstmCRF_export(ABC_EXPORT,):
#    def __init__(self,args):
#        super().__init__(args)
#    @staticmethod
#    def model_fn_builder(args):
#        return BILSTM_CRF_NER_MODEL.model_fn_builder(args)
    @staticmethod
    def serving_input_receiver_fn():
        words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
        nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
        receiver_tensors = {'words': words, 'nwords': nwords}
        features = {'words': words, 'nwords': nwords}
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    
