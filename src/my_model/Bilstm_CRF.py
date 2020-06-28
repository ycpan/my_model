"""GloVe Embeddings + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import functools
import json
import logging
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1
from MyUtils.EvaluateScore import EvaluateScore
from MyUtils.utils import get_data_from_file
from MyUtils.log_to_hbase import Log_To_HBase
from MyUtils.utils import get_file_list
from tensorflow.python.client import device_lib
import os
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #     ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        is_continue = True
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            if g is    None:
                # grad is None
                tf.logging.warning('grad is None')
                is_continue = False
                break 
            expanded_g = tf.expand_dims(g, 0)
            print('#######grad is {}##########'.format(g))
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

    # Average over the 'tower' dimension.
        if is_continue: 
            grad = tf.concat(grads, 0)
            grad = tf.reduce_sum(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads



class Bilstm_CRF(EvaluateScore, Log_To_HBase):
    def __init__(self, base_dir=None, input_dir = None, gpu='', label_list=['E', 'X'], params=None, off_predict=False):
        if base_dir is None:
            self.base_dir = './'
            # self.input_dir = '../dataset/pre-develop-bilstm-crf/model_input_data'
        else:
            self.base_dir = base_dir
        if input_dir:

            self.input_dir = input_dir
        else:
            self.input_dir = os.path.join(self.base_dir, 'model_input')

        #if len(gpu) == 0:
        #    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        #else:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
        print(gpu)
        self.devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
        self.num_gpus = len(self.devices)
        print(self.devices)
        print(self.num_gpus)
        self.params = params
        if self.params is None:
            self.params = {
                'dim': 256,
                'dropout': 0.1,
                'num_oov_buckets': 1,
                'epochs': 20,
                # 'batch_size': 20,
                'batch_size': 200,
                'buffer': 15000,
                # 'lstm_size': 100,
                'lstm_size': 512,
                'learnning_rate': 0.0002
                # 'words': str(Path(self.input_dir, 'vocab.words.txt')),
                # 'chars': str(Path(self.input_dir, 'vocab.chars.txt')),
                # 'tags': str(Path(self.input_dir, 'vocab.tags.txt')),
                # 'glove': str(Path(self.input_dir, 'glove.npz'))
            }
        self.params['words'] = str(Path(self.input_dir, 'vocab.words.txt'))
        self.params['chars'] = str(Path(self.input_dir, 'vocab.chars.txt'))
        self.params['tags'] = str(Path(self.input_dir, 'vocab.tags.txt'))
        self.params['glove'] = str(Path(self.input_dir, 'glove.npz'))
        print(self.params)

        self.indices = None
        self.estimator = None
        self.hook = True
        self.embeding = ''
        #self.embeding = 'glove'
        # if not self.score_data:
        self.score_dir = os.path.join(self.base_dir, 'score')
        if not os.path.exists(self.score_dir):
            os.mkdir(self.score_dir)

        self.tags_from_table = dict()
        # super(Bilstm_CRF, self).__init__(label_list, diff_dir=self.base_dir)
        abs_base_dir = os.path.abspath(self.base_dir)
        base_name = os.path.basename(abs_base_dir)
        row_key = 'Bilstm_CRF.{}'.format(base_name)
        table_cf = {str('params'): dict(max_versions=1024),
                    str('scores'): dict(max_versions=1024),
                    str('best_params'): dict(max_versions=32),
                    str('best_score'): dict(max_versions=32),
                    str('other1'): dict(max_versions=1024),
                    str('other2'): dict(max_versions=1024)}
        EvaluateScore.__init__(self, label_list=label_list, base_dir=self.base_dir)

        # Log_To_HBase.__init__(self, table_name='SequenceLabeling',row_key=row_key,column_families=table_cf)

        # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # DATADIR = './example1'

        # Logging
        self.Bilstm_CRF_log_dir = '{}/Bilstm_CRF_log'.format(self.base_dir)
        if not os.path.exists(self.Bilstm_CRF_log_dir):
            Path(self.Bilstm_CRF_log_dir).mkdir(exist_ok=True)
        tf.logging.set_verbosity(logging.INFO)
        handlers = [
            logging.FileHandler('{}/main.log'.format(self.Bilstm_CRF_log_dir)),
            logging.StreamHandler(sys.stdout)
        ]
        logging.getLogger('tensorflow').handlers = handlers
        # results
        self.Bilstm_CRF_results_dir = '{}/Bilstm_CRF_results'.format(self.base_dir)
        if not os.path.exists(self.Bilstm_CRF_results_dir):
            Path(self.Bilstm_CRF_results_dir).mkdir(exist_ok=True)

        with Path(self.params['words']).open(encoding='utf-8') as f:
            vocab_words = f.readlines()
            vocab_length = len(vocab_words)
        self.params['embeding_size'] = vocab_length + 1
        with Path(self.params['tags'], ).open(encoding='utf-8') as f:
            self.indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
            num_tags = len(self.indices) + 1
            # num_tags = len(self.indices)
        self.params['num_tags'] = num_tags
        with Path('{}/params.json'.format(self.Bilstm_CRF_results_dir)).open('w', encoding='utf-8') as f:
            json.dump(self.params, f, indent=4, sort_keys=True)

        with open(self.params['tags'], 'r', encoding='utf-8') as f:
            # iter = f.__iter__()
            for i, tag in enumerate(f.__iter__()):
                self.tags_from_table[i] = tag.strip()

        # self.write_data(self.params, self.table_name, self.row_key,'params')
        # self.save_data(self.params,'params')

    def parse_fn(self, line_words, line_tags):
        # Encode in Bytes for TF
        # print(line_words)
        # print(line_tags)

        words = [w.encode() for w in line_words.strip().split(' ')]
        tags = [t.encode() for t in line_tags.strip().split(' ')]

        assert len(words) == len(tags), "Words and tags lengths don't match:\n words:{}\n:tags:{}".format(words,tags)
        return (words[:500], len(words[:500])), tags[:500]


    def generator_fn(self, words, tags):
        with Path(words).open('r', encoding='utf-8') as f_words, Path(tags).open('r', encoding='utf-8') as f_tags:
            for line_words, line_tags in zip(f_words, f_tags):
                yield self.parse_fn(line_words, line_tags)

    def input_fn(self, words, tags, params=None, shuffle_and_repeat=False):
        params = params if params is not None else {}
        shapes = (([None], ()), [None])
        types = ((tf.string, tf.int32), tf.string)
        defaults = (('<pad>', 0), 'O')
        #defaults = (('<pad>', 0), 'S')

        dataset = tf.data.Dataset.from_generator(
            functools.partial(self.generator_fn, words, tags),
            output_shapes=shapes, output_types=types)

        if shuffle_and_repeat:
            dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])
            

        dataset = (dataset
                   .padded_batch(params.get('batch_size', 20)*self.num_gpus, shapes, defaults)
                   #.padded_batch(params.get('batch_size', 20), shapes, defaults)
                   .prefetch(1))
        return dataset

    def model_fn(self, features, labels, mode, params):
        # For serving, features are a bit different

        if isinstance(features, dict):
            features = features['words'], features['nwords']

        # Read vocabs and inputs
        dropout = params['dropout']
        words, nwords = features
        
        #nwords = tf.shape(words)[0]
        #print('###########tf.shape nwords:{}#######'.format(nwords))
        training = (mode == tf.estimator.ModeKeys.TRAIN)
        vocab_words = tf.contrib.lookup.index_table_from_file(
            params['words'], num_oov_buckets=params['num_oov_buckets'])
        if mode == tf.estimator.ModeKeys.PREDICT:
            # Word Embeddings
            word_ids = vocab_words.lookup(words)
            if self.embeding == 'glove':
                glove = np.load(params['glove'])['embeddings']  # np.array
                variable = np.vstack([glove, [[0.]*params['dim']]])
                variable = tf.Variable(variable, dtype=tf.float32, trainable=True)
                embeddings = tf.nn.embedding_lookup(variable, word_ids)
                embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

            # add by panyc
            # with Path(params['words']).open() as f:
            #     vocab_words = f.readlines()
            #     vocab_length = len(vocab_words)
            # end
            else:

                embeddings = tf.Variable(
                    # tf.random_uniform([vocab_length + 1, 300], -1.0, 1.0))
                    tf.random_normal([params['embeding_size'], 300], 0.0, 0.057735026918962574)
                )
                embeddings = tf.nn.embedding_lookup(embeddings, word_ids)
                embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

            # LSTM
            # t = tf.transpose(embed, perm=[1, 0, 2])
            t = tf.transpose(embeddings, perm=[1, 0, 2])
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
            output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.transpose(output, perm=[1, 0, 2])
            output = tf.layers.dropout(output, rate=dropout, training=training)

            # CRF
            logits = tf.layers.dense(output, params['num_tags'])
            crf_params = tf.get_variable("crf", [params['num_tags'], params['num_tags']], dtype=tf.float32)
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

            # Predictions
            reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
                params['tags'])
            pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
            predictions = {
                'pred_ids': pred_ids,
                'tags': pred_strings
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else:
            #print('##########nwords:{}###########'.format(nwords))
            ##words_shards = tf.split(words, self.num_gpus)
            ##labels_shards = tf.split(labels, self.num_gpus)
            #lr = None
            #optimizer = None
            #gradient_clip_value = params['gradient_clip_value']
            #step = tf.to_float(tf.train.get_global_step())
            #warmup_steps = params['warmup_steps']
            #if params['learning_rate_decay'] == 'sqrt':
            #    lr_warmup = params['learning_rate_peak'] * tf.minimum(1.0,step/warmup_steps)
            #    lr_decay = params['learning_rate_peak'] * tf.minimum(1.0,tf.sqrt(warmup_steps/step))
            #    lr = tf.where(step < warmup_steps, lr_warmup, lr_decay)
            #elif params['learning_rate_decay'] == 'exp':
            #    lr = tf.train.exponential_decay(params['learning_rate_peak'],
            #            global_step=step,
            #            decay_steps=params['decay_steps'],
            #            decay_rate=params['decay_rate'])
            #else:
            #    tf.logging.info('learning rate decay strategy not supported')
            #    sys.exit()
            #if params['optimizer'] == 'sgd':
            #    optimizer = tf.train.GradientDescentOptimizer(lr)
            #else params['optimizer'] == 'adam':
            #    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.997, epsilon=1e-09)
            ##else :
            ##    tf.logging.info('optimizer not supported')
            ##    sys.exit()
            #
            #optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, gradient_clip_value)
            #words = tf.cond(tf.less(tf.shape(words)[0], self.num_gpus), \
            #    lambda:tf.concat([words]*self.num_gpus,0),lambda:words)
            #nwords = tf.cond(tf.less(tf.shape(nwords)[0], self.num_gpus), \
            #    lambda:tf.concat([nwords]*self.num_gpus,0),lambda:nwords)
            #labels = tf.cond(tf.less(tf.shape(labels)[0], self.num_gpus), \
            #    lambda:tf.concat([labels]*self.num_gpus,0),lambda:labels)
            #n = (tf.shape(words)[0]//self.num_gpus ) * self.num_gpus
            #words = words[:n]
            #nwords = nwords[:n]
            #labels = labels[:n]
            #words_shards = tf.split(words, self.num_gpus)
            #nwords_shards = tf.split(nwords, self.num_gpus)
            #labels_shards = tf.split(labels, self.num_gpus)
            #loss_shards = []
            #grad_shards = []
            #metric_accuracy = []
            #accuracy_op = None
            #metric_precision = []
            #precision_op = None
            #metric_recall =  []
            #recall_op = None
            #metric_f1 = []
            #f1_op = None
            ##nwords = tf.div(nwords, self.num_gpus)
            ##nwords=10
            ##nwords = tf.constant([nwords,], dtype=tf.int32)
            #for i, device in enumerate(self.devices):
            #    with tf.variable_scope( tf.get_variable_scope(), reuse=True if i > 0 else None):
            #        with tf.device(device):
            #            words = words_shards[i]
            #            nwords = nwords_shards[i]
            #            labels = labels_shards[i]
            #            word_ids = vocab_words.lookup(words)
            #            if self.embeding == 'glove':
            #                glove = np.load(params['glove'])['embeddings']  # np.array
            #                variable = np.vstack([glove, [[0.]*params['dim']]])
            #                variable = tf.Variable(variable, dtype=tf.float32, trainable=True)
            #                embeddings = tf.nn.embedding_lookup(variable, word_ids)
            #                embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

            #            # add by panyc
            #            # with Path(params['words']).open() as f:
            #            #     vocab_words = f.readlines()
            #            #     vocab_length = len(vocab_words)
            #            # end
            #            else:

            #                embeddings = tf.Variable(
            #                    # tf.random_uniform([vocab_length + 1, 300], -1.0, 1.0))
            #                    tf.random_normal([params['embeding_size'], 300], 0.0, 0.057735026918962574)
            #                )
            #                embeddings = tf.nn.embedding_lookup(embeddings, word_ids)
            #                embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

            #            # LSTM
            #            # t = tf.transpose(embed, perm=[1, 0, 2])
            #            t = tf.transpose(embeddings, perm=[1, 0, 2])
            #            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
            #            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
            #            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            #            output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
            #            output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
            #            output = tf.concat([output_fw, output_bw], axis=-1)
            #            output = tf.transpose(output, perm=[1, 0, 2])
            #            output = tf.layers.dropout(output, rate=dropout, training=training)

            #            # CRF
            #            logits = tf.layers.dense(output, params['num_tags'])
            #            crf_params = tf.get_variable("crf", [params['num_tags'], params['num_tags']], dtype=tf.float32)
            #            pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

            #            # Loss

            #            vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
            #            # vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'], num_oov_buckets=params['num_oov_buckets'])
            #            tags = vocab_tags.lookup(labels)
            #            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            #                logits, tags, nwords, crf_params)
            #            loss = tf.reduce_mean(-log_likelihood)
            #            loss_shards.append(loss)
            #            print('############loss:{}##############'.format(loss))
            #            grads = optimizer.compute_gradients(loss)
            #            print('################grads{}###########'.format(grads))
            #            grad_shards.append(grads)
            #            weights = tf.sequence_mask(nwords,tf.shape(tags)[1])
            #            ## add by panyc
            #            #weights = tf.expand_dims(weights,axis=0)
            #            ## end
            #            val,accuracy_op = tf.metrics.accuracy(tags, pred_ids, weights)
            #            metric_accuracy.append([val])
            #            val,precision_op = precision(tags, pred_ids, params['num_tags'], self.indices, weights)
            #            metric_precision.append([val])
            #            val,recall_op = recall(tags, pred_ids, params['num_tags'], self.indices, weights)
            #            metric_recall.append([val])
            #            val,f1_op = f1(tags, pred_ids, params['num_tags'], self.indices, weights)
            #            metric_f1.append([val])

            #loss = tf.reduce_mean(loss_shards)
            #print('##############{}##############'.format(grad_shards))
            #grad = average_gradients(grad_shards)
            #metric_accuracy = tf.reduce_mean(metric_accuracy)
            #metric_precision = tf.reduce_mean(metric_precision)
            #metric_recall = tf.reduce_mean(metric_recall)
            #metric_f1 = tf.reduce_mean(metric_f1)
            #metrics = {
            #    'acc': (metric_accuracy,accuracy_op),
            #    'precision': (metric_precision,precision_op),
            #    'recall': (metric_recall, recall_op),
            #    'f1': (metric_f1, f1_op),
            #}
            ## Metrics
            ##weights = tf.sequence_mask(nwords)
            #for metric_name, op in metrics.items():
            #    print('############op##########')
            #    print(op)
            #    tf.summary.scalar(metric_name, op[1])
            word_ids = vocab_words.lookup(words)
            with Path(params['words']).open(encoding='utf-8') as f:
                vocab_words_1 = f.readlines()
                vocab_length = len(vocab_words_1)
            params['embeding_size'] = vocab_length + 1
            embeddings = tf.Variable(
                # tf.random_uniform([vocab_length + 1, 300], -1.0, 1.0))
                tf.random_normal([params['embeding_size'], 300], 0.0, 0.057735026918962574)
            )
            embeddings = tf.nn.embedding_lookup(embeddings, word_ids)
            embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

            # LSTM
            t = tf.transpose(embeddings, perm=[1, 0, 2])
            lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
            lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
            lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
            output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.transpose(output, perm=[1, 0, 2])
            output = tf.layers.dropout(output, rate=dropout, training=training)

            # CRF
            logits = tf.layers.dense(output, params['num_tags'])
            crf_params = tf.get_variable("crf", [params['num_tags'], params['num_tags']], dtype=tf.float32)
            pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

            #if mode == tf.estimator.ModeKeys.PREDICT:
            #    # Predictions
            #    reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            #        params['tags'])
            #    pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
            #    predictions = {
            #        'pred_ids': pred_ids,
            #        'tags': pred_strings
            #    }
            #    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            #else:
            # Loss
            vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
            tags = vocab_tags.lookup(labels)
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                logits, tags, nwords, crf_params)
            loss = tf.reduce_mean(-log_likelihood)

            # Metrics
            weights = tf.sequence_mask(nwords)
            metrics = {
                'acc': tf.metrics.accuracy(tags, pred_ids, weights),
                'precision': precision(tags, pred_ids, params['num_tags'], self.indices, weights),
                'recall': recall(tags, pred_ids, params['num_tags'], self.indices, weights),
                'f1': f1(tags, pred_ids, params['num_tags'], self.indices, weights),
            }
            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op[1])


            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=metrics)

            elif mode == tf.estimator.ModeKeys.TRAIN:

                #train_op = optimizer.apply_gradients(grad, global_step=tf.train.get_global_step())
                #ema = tf.train.ExponentialMovingAverage(decay=0.9)
                #with tf.control_dependencies([train_op]):
                #    train_op = ema.apply(tf.trainable_variables())

                #train_op = tf.train.AdamOptimizer(learning_rate=self.params['learnning_rate']).minimize(
                #    loss, global_step=tf.train.get_or_create_global_step())
                train_op = tf.train.AdamOptimizer().minimize(
                    loss, global_step=tf.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, train_op=train_op)

    def train_bilstm_crf_model(self):
        # Params

        # with Path(self.params['words']).open(encoding='utf-8') as f:
        #     vocab_words = f.readlines()
        #     vocab_length = len(vocab_words)
        # self.params['embeding_size'] = vocab_length + 1
        # with Path(self.params['tags'], ).open(encoding='utf-8') as f:
        #     self.indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        #     num_tags = len(self.indices) + 1
        #     # num_tags = len(self.indices)
        # self.params['num_tags'] = num_tags
        # with Path('{}/params.json'.format(self.Bilstm_CRF_results_dir)).open('w', encoding='utf-8') as f:
        #     json.dump(self.params, f, indent=4, sort_keys=True)
        #
        # with open(self.params['tags'], 'r', encoding='utf-8') as f:
        #     # iter = f.__iter__()
        #     for i, tag in enumerate(f.__iter__()):
        #         self.tags_from_table[i] = tag.strip()
        # Estimator, train and evaluate
        train_inpf = functools.partial(self.input_fn, self.f_words('train'), self.f_tags('train'),
                                       self.params, shuffle_and_repeat=True)
        eval_inpf = functools.partial(self.input_fn, self.f_words('testa'), self.f_tags('testa'))

        #session_config = tf.ConfigProto(device_count={'GPU': 0,'GPU':1,'GPU':2})
        session_config = tf.ConfigProto(allow_soft_placement = True,gpu_options=tf.GPUOptions(allow_growth = True))
        ''' With devices=None, MirroredStrategy will use all GPUs made availble to the process '''
        #train_distribution_strategy = tf.contrib.distribute.MirroredStrategy(devices=None)
        #train_distribution_strategy = tf.contrib.distribute.MirroredStrategy()
        #eval_distribution_strategy = tf.contrib.distribute.MirroredStrategy()
        #cfg = tf.estimator.RunConfig(save_checkpoints_steps=120,train_distribute=train_distribution_strategy,eval_distribute=eval_distribution_strategy, save_summary_steps=1,log_step_count_steps=10,keep_checkpoint_max=20).replace(session_config=session_config)
        cfg = tf.estimator.RunConfig(save_checkpoints_steps=120, save_summary_steps=1,log_step_count_steps=10,keep_checkpoint_max=20).replace(session_config=session_config)
        self.estimator = tf.estimator.Estimator(self.model_fn, '{}/model'.format(self.Bilstm_CRF_results_dir), cfg,
                                                self.params)
        Path(self.estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        if not self.hook:
            # hook = None
            train_spec = tf.estimator.TrainSpec(input_fn=train_inpf)

        else:
            hook = tf.contrib.estimator.stop_if_no_increase_hook(
                self.estimator, 'f1', 1000, min_steps=8000, run_every_secs=120)
            train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
        # tf.estimator.tr

        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

        # Write predictions to file

    def evaluate_all_train_file(self):
        # for file_path in get_file_list(self.input_data, file_type='words.txt',except_file_type='bak'):
        for name in ['train', 'testa', 'testb']:
            self.evaluate_train_file(name)

    def evaluate_train_file(self, name):
        self.predictions(name, self.f_predict_score(name))
        print('start compute {}'.format(name))

        self.compute_socres(self.f_predict_score(name))

    def save_score_file(self):
        pass

    def compute_socres(self, file_path):
        # ev = EvaluateScore(self.label_list)
        # ev.run(file_path)
        # self.score_file_name = os.path.basename(file_path)
        data = get_data_from_file(file_path)
        self.batch_process(data)
        save_score = {'f1':self.f1,'prediction':self.precision,'recall':self.recall}
        # self.save_data(save_score,'scores')
        self._get_timestamp()


    def predictions(self, name, save_name):
        # Path('results/score').mkdir(parents=True, exist_ok=True)
        # with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
        session_config = tf.ConfigProto(allow_soft_placement = True,gpu_options=tf.GPUOptions(allow_growth = True))
        cfg = tf.estimator.RunConfig(save_checkpoints_steps=120, save_summary_steps=1,log_step_count_steps=10,keep_checkpoint_max=20).replace(session_config=session_config)
        self.estimator = tf.estimator.Estimator(self.model_fn, '{}/model'.format(self.Bilstm_CRF_results_dir), cfg,
                                                self.params)
        with Path(save_name).open('wb') as f:
            test_inpf = functools.partial(self.input_fn, self.f_words(name), self.f_tags(name))
            golds_gen = self.generator_fn(self.f_words(name), self.f_tags(name))
            preds_gen = self.estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                # for word, tag, tag_pred in zip(words, tags, preds['tags']):
                # print(preds)

                tags_id = [self.tags_from_table[id] for id in preds['pred_ids']]
                for word, tag, tag_pred in zip(words, tags, tags_id):
                    f.write(b' '.join([word, tag, tag_pred.encode()]) + b'\n')
                f.write(b'\n')

    def f_words(self, name):
        return str(Path(self.input_dir, '{}.words.txt'.format(name)))

    def f_tags(self, name):
        return str(Path(self.input_dir, '{}.tags.txt'.format(name)))

    def f_predict_score(self, name):
        return str(Path(self.score_dir, 'score_{}'.format(name)))
    def predict_from_tf_serving(self, line):
        from grpc.beta import implementations
        from tensorflow_serving.apis import predict_pb2
        from tensorflow_serving.apis import prediction_service_pb2_grpc

        # def pretty_print(line, preds):
        #     words = line.strip().split()
        #     lengths = [max(len(w), len(p)) for w, p in zip(words, preds)]
        #     padded_words = [w + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
        #     padded_preds = [p.decode() + (l - len(p)) * ' ' for p, l in zip(preds, lengths)]
        #     # print('words: {}'.format(' '.join(padded_words)))
        #     # print('preds: {}'.format(' '.join(padded_preds)))
        #     # res1 = 'words: {}'.format(' '.join(padded_words))
        #     # res2 = 'preds: {}'.format(' '.join(padded_preds))
        #     res1 = ' '.join(padded_words)
        #     res2 = ' '.join(padded_preds)
        #     return res1, res2
        #
        # def predict_input_fn(line):
        #     # Words
        #     words = [w.encode() for w in line.strip().split()]
        #     nwords = len(words)
        #
        #     # Wrapping in Tensors
        #     # words = tf.constant([words], dtype=tf.string)
        #     # nwords = tf.constant([nwords], dtype=tf.int32)
        #
        #     # return (words, nwords), None
        #     return words, nwords
        def pretty_print(line, preds):
            line = repr(line).replace('\\', '/')[1:-1]
            line = list(line.replace(' ', ''))
            words = [w for w in line]
            lengths = [max(len(w), len(p)) for w, p in zip(words, preds)]
            padded_words = [w + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
            padded_preds = [p + (l - len(p)) * ' ' for p, l in zip(preds, lengths)]
            print('words: {}'.format('\t'.join(padded_words)))
            print('preds: {}'.format('\t'.join(padded_preds)))

        def predict_input_fn(line):
            # Words
            # line = ''.join([' ' + c + ' ' if len(c.encode()) > 1 else c for c in line]).split()
            line = repr(line).replace('\\', '/')[1:-1]
            line = list(line.replace(' ', ''))
            words = [w.encode() for w in line]
            nwords = len(words)

            # Wrapping in Tensors
            # words = tf.constant(words, shape=(1, nwords), dtype = tf.string)
            # nwords = tf.constant([nwords], shape=(1,), dtype=tf.int32)

            # return (words, nwords), None
            return words, nwords

        def build_sentence(data, tags):
            res = ''
            sub_res = []
            for word, tag in zip(data.strip().split(), tags.strip().split()):

                if tag == 'B':
                    if sub_res:
                        sub_str = ''.join(sub_res)
                        res = res + sub_str + ' '
                        # idx = i
                        sub_res = []
                    else:
                        # idx = i
                        sub_res.append(word)
                elif tag == 'S':
                    # res[i] = word
                    # ' '.join(res, word)
                    res = res + word + ' '
                    # idx = -1
                elif tag == 'M':
                    sub_res.append(word)

                    # idx = -1

                elif tag == 'E':
                    if len(sub_res) > 0:
                        sub_res.append(word)
                        sub_str = ''.join(sub_res)
                        res = res + sub_str + ' '

                        sub_res = []
                    else:
                        # res[idx] = word
                        res = res + word + ' '

            return res.strip()

        # @timecost
        # def predict(testStr):
        host = '43.247.185.201'
        # port='8031'
        port = '8036'
        # port='30000'
        channel = implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel._channel)

        request = predict_pb2.PredictRequest()
        request.model_spec.name = "saved_model"
        words, nwords = predict_input_fn(line)
        request.inputs['words'].CopyFrom(
            tf.contrib.util.make_tensor_proto(words, shape=(1, nwords), dtype=tf.string))
        # tf.contrib.util.make_tensor_proto(words, shape=words.shape, dtype=tf.string))
        request.inputs['nwords'].CopyFrom(
            tf.contrib.util.make_tensor_proto([nwords], shape=(1,), dtype=tf.int32))

        future = stub.Predict.future(request, 10.0)
        result = future.result()
        result_list = tf.make_ndarray(result.outputs["pred_ids"]).tolist()
        tags_id = [self.tags_from_table[id] for id in result_list[0]]
        pretty_print(line, tags_id)

        # 10 secs timeout
        # print(result)
    def Bilstm_CRF_save_model(self, save_path):

        def serving_input_receiver_fn():
            """Serving input_fn that builds features from placeholders

            Returns
            -------
            tf.estimator.export.ServingInputReceiver
            """
            words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
            nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
            receiver_tensors = {'words': words, 'nwords': nwords}
            features = {'words': words, 'nwords': nwords}
            return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)


        estimator = tf.estimator.Estimator(self.model_fn,'{}/model'.format(self.Bilstm_CRF_results_dir), params=self.params)
        estimator.export_saved_model('{}/saved_model'.format(save_path), serving_input_receiver_fn)
    def off_line_predict(self, line):
        def pretty_print(line, preds):
            line = repr(line).replace('\\', '/')[1:-1]
            line = list(line.replace(' ', ''))
            words = [w for w in line]
            lengths = [max(len(w), len(p)) for w, p in zip(words, preds)]
            padded_words = [w + (l - len(w)) * ' ' for w, l in zip(words, lengths)]
            padded_preds = [p + (l - len(p)) * ' ' for p, l in zip(preds, lengths)]
            print('words: {}'.format('\t'.join(padded_words)))
            print('preds: {}'.format('\t'.join(padded_preds)))

        def predict_input_fn(line):
            # Words
            # line = ''.join([' ' + c + ' ' if len(c.encode()) > 1 else c for c in line]).split()
            line = repr(line).replace('\\', '/')[1:-1]
            line = list(line.replace(' ', ''))
            words = [w.encode() for w in line]
            nwords = len(words)

            # Wrapping in Tensors
            words = tf.constant(words, shape=(1, nwords), dtype = tf.string)
            nwords = tf.constant([nwords], shape=(1,), dtype=tf.int32)

            return (words, nwords), None
            # return words, [nwords]
        def generate_data(path):
            data = get_data_from_file(path)
            for line in data:
                line = eval(line)
                print(line['text'])
                yield predict_input_fn(line['text'])

        def input_fn( path, params=None, shuffle_and_repeat=False):
            # params = params if params is not None else {}
            # shapes = (([None], ()), [None])
            # types = ((tf.string, tf.int32), tf.string)
            # # defaults = (('<pad>', 0), 'O')
            # defaults = (('<pad>', 0), 'O')
            #
            # dataset = tf.data.Dataset.from_generator(
            #     functools.partial(generate_data, path),
            #     output_shapes=shapes, output_types=types)
            #
            # # if shuffle_and_repeat:
            # #     dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])
            #
            # dataset = (dataset
            #            .padded_batch(params.get('batch_size', 20), shapes, defaults)
            #            .prefetch(1))
            data = generate_data(path)
            return data.__next__()



        estimator = tf.estimator.Estimator(self.model_fn, '{}/model'.format(self.Bilstm_CRF_results_dir), params=self.params)
        predict_inpf = functools.partial(predict_input_fn, line)
        # path = '{}testa'.format(self.forward_input_dir)
        # predict_inpf = functools.partial(input_fn, path)
        for pred in estimator.predict(predict_inpf):
            # print(pred['pred_ids'])
            tags_id = [self.tags_from_table[id] for id in pred['pred_ids']]
            pretty_print(line, tags_id)
            break
