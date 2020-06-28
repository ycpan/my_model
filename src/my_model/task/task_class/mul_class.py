# this is a multi class  method base on mulit_class_abc method,select one from N,N>2
from my_model.task.task_class.abc_mul_class import ABCMulClass
#import functools
import json
import logging
import sys
from my_model.embeddings.embedding import embedding_lookup,embedding
import numpy as np
import tensorflow as tf
from tf_metrics import precision, recall, f1
import my_model.metrics
import os
from pathlib import Path
#from my_model.abc.model_abc import ABC_MODEL
#from my_model.abc.eval_abc import ABC_EVAL
#from my_model.abc.export_abc import ABC_EXPORT
#from my_model.layers.data_reader_v1 import get_input_fn
#from my_model.layers.data_reader_v2 import get_input_fn
from my_model.layers.data_reader_v2 import InputFn
#from my_model.embeddings.bert_base.bert import modeling
#from my_model.embeddings.bert_base.bert import optimization
from my_model.embeddings.bert import modeling
from my_model.embeddings.bert import optimization
class MulClass(ABCMulClass):
    def __init__(self,args):
        #if model:
        #    rasie ValueError('model shoud be specified train or eval')
        args.task_name = 'ner'
        self.name = 'MulClass'
        #self.args = args
        if args.log_dir:
            self.log_dir = args.log_dir
        else:
            self.log_dir = ''
        if args.log_name:
            self.log_name = args.log_name
        else:
            log_name = '{}.log'.format(self.name) 

        self.args = args
        if not self.args.vocab_words :
            self.args.vocab_words = os.path.join(self.args.data_dir,'vocab.words.txt')
        if not self.args.vocab_tags :
            self.args.vocab_tags = os.path.join(self.args.data_dir,'vocab.tags.txt')
        if not self.args.vocab_chars :
            self.args.vocab_chars = os.path.join(self.args.data_dir,'vocab.chars.txt')
        if not self.args.glove :
            self.args.glove = os.path.join(self.args.data_dir,'glove.npz')
        super().__init__(data_dir=args.data_dir,output=args.output,tags_path=args.vocab_tags,vocab_path=args.vocab_words,embeddings_type=args.embedding,tokenizer=None,max_seq_length=args.max_seq_length,args=args) 
        self.num_labels = len(self.label_list)
        self.logging.info(args)
    #@staticmethod
    def train(self):

        if not self.args.train_data:
            self.args.train_data = os.path.join(self.args.data_dir,'train.txt')
        if not self.args.eval_data:
            self.args.eval_data = os.path.join(self.args.data_dir,'dev.txt')
        #if not self.args.test_data:
        #    self.args.test_data = os.path.join(self.args.data_dir,'test.txt')
        train_example_file = self.get_example_tf_record(self.args.train_data,is_write=False,set_type='train')
        eval_example_file = self.get_example_tf_record(self.args.eval_data,is_write=False,set_type='eval')
        #train_example_file = self.get_example_tf_record(self.args.train_data,is_write=True,set_type='train')
        #eval_example_file = self.get_example_tf_record(self.args.eval_data,is_write=True,set_type='eval')
        len_label = 1
        train_inpf = self.get_input_fn(train_example_file,self.args.batch_size,len_label=len_label,is_training=True,drop_remainder=False)
        eval_inpf = self.get_input_fn(eval_example_file,self.args.batch_size,len_label=len_label,is_training=False,drop_remainder=False)

        model_fn = self.model_fn_builder(self.args)

        #train_inpf = functools.partial(self.input_fn, self.fwords(self.args.data_dir,'train'), self.ftags(self.args.data_dir,'train'),
        #                               self.args, shuffle_and_repeat=True)
        #eval_inpf = functools.partial(self.input_fn, self.fwords(self.args.data_dir,'testa'), self.ftags(self.args.data_dir,'testa'),
        #                               self.args, shuffle_and_repeat=False)
    
        cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
        estimator = tf.estimator.Estimator(model_fn, '{}/model'.format(self.args.output), cfg, self.args)
        Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        #hook = tf.contrib.estimator.stop_if_no_increase_hook(
        #hook = tf.estimator.experimental.stop_if_no_increase_hook(
            #estimator, 'f1', 500, min_steps=80000, run_every_secs=120)
        #train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def predict(self):
        if not self.args.test_data:
            self.args.test_data = os.path.join(self.args.data_dir,'test.txt')
        #test_example_file = self.get_example_tf_record(self.args.test_data,is_write=False,set_type='test')
        test_example_file = os.path.join(self.data_dir, "{}.tf_record".format('test'))
        set_type='test'
        examples = self.get_examples(self.args.test_data,set_type)
        batch_tokens,batch_labels = self.filed_based_convert_examples_to_features(examples,test_example_file)
        test_inpf = self.get_input_fn(test_example_file,self.args.batch_size,is_training=False,drop_remainder=False)

        self.num_labels = len(self.label_list)
        model_fn = self.model_fn_builder(self.args)
        session_config = tf.ConfigProto(allow_soft_placement = True,gpu_options=tf.GPUOptions(allow_growth = True))
        cfg = tf.estimator.RunConfig(save_checkpoints_steps=120, save_summary_steps=1,log_step_count_steps=10,keep_checkpoint_max=20).replace(session_config=session_config)
        estimator = tf.estimator.Estimator(model_fn, '{}/model'.format(self.args.output), cfg, self.args)
        result = estimator.predict(input_fn=test_inpf)
        Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        output_predict_file = os.path.join(self.args.output, "label_test.txt")

        id2label = {value: key for key, value in self.label_map.items()}
        self.Writer(output_predict_file,result,batch_tokens,batch_labels,id2label)
        return output_predict_file
    def _write_base(self,batch_tokens,id2label,prediction,batch_labels,wf,i):
        token = batch_tokens[i]
        def get_more_tags(prediction,id2label):
            max_value = prediction.max()
            res = []
            for idx,v in enumerate(prediction):
                diff = max_value / v 
                if diff < 2 :
                    res.append(id2label.get(idx,'NULL'))
            return res
        #predict = id2label[prediction]
        try:
            predict = get_more_tags(prediction,id2label)
            #import ipdb
            #ipdb.set_trace()
            true_l = id2label.get(batch_labels[i],'NULL')
            if token!="[PAD]" and token!="[CLS]" and true_l!="X":
                #
                if predict=="X" and not predict.startswith("##"):
                    predict="O"
                
                line = "{}\t{}\t{}\n".format(' '.join(token),true_l,' '.join(predict))
                wf.write(line)
        except Exception as e:
           #import ipdb
           #ipdb.set_trace()
           self.logging.error(e)

    
    def Writer(self,output_predict_file,result,batch_tokens,batch_labels,id2label):
        with open(output_predict_file,'w') as wf:
    
            #if  FLAGS.crf:
            #    predictions  = []
            #    for m,pred in enumerate(result):
            #        predictions.extend(pred)
            #    for i,prediction in enumerate(predictions):
            #        _write_base(batch_tokens,id2label,prediction,batch_labels,wf,i)
            #        
            #else:
            for i,prediction in enumerate(result):
                self._write_base(batch_tokens,id2label,prediction,batch_labels,wf,i)
                

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
        def softmax_layer(logits,labels,num_labels,mask):
            logits = tf.reshape(logits, [-1, num_labels])
            labels = tf.reshape(labels, [-1])
            mask = tf.cast(mask,dtype=tf.float32)
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
            loss = tf.losses.softmax_cross_entropy(logits=logits,onehot_labels=one_hot_labels)
            loss *= tf.reshape(mask, [-1])
            loss = tf.reduce_sum(loss)
            total_size = tf.reduce_sum(mask)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights
            loss /= total_size
            # predict not mask we could filtered it in the prediction part.
            probabilities = tf.math.softmax(logits, axis=-1)
            predict = tf.math.argmax(probabilities, axis=-1)
            return loss, predict
        def crf_loss(logits,labels,num_labels,mask2len):
            """
            :param logits:
            :param labels:
            :param mask2len:each sample's length
            :return:
            """
            #TODO
            with tf.variable_scope("crf_loss"):
                trans = tf.get_variable(
                        "transition",
                        shape=[num_labels,num_labels],
                        initializer=tf.contrib.layers.xavier_initializer()
                )
            
            log_likelihood,transition = tf.contrib.crf.crf_log_likelihood(logits,labels,transition_params =trans ,sequence_lengths=mask2len)
            loss = tf.math.reduce_mean(-log_likelihood)
           
            return loss,transition
        def hidden2tag(hiddenlayer,numclass):
            linear = tf.keras.layers.Dense(numclass,activation=None)
            return linear(hiddenlayer)
        def create_model(embeddings, labels,mask,mode, training, num_labels, use_one_hot_embeddings):
        #    model = modeling.BertModel(
        #        config = bert_config,
        #        training=training,
        #        input_ids=input_ids,
        #        input_mask=mask,
        #        token_type_ids=segment_ids,
        #        use_one_hot_embeddings=use_one_hot_embeddings
        #        )
        #
            #output_layer = embeddings
            ##output_layer shape is
            #if training:
            #    output_layer = tf.keras.layers.Dropout(rate=0.1)(output_layer)
            #logits = hidden2tag(output_layer,num_labels)
            ## TODO test shape
            #logits = tf.reshape(logits,[-1,args.max_seq_length,num_labels])
            ##if FLAGS.crf:
            mask2len = tf.reduce_sum(mask,axis=1)
            #loss, trans = crf_loss(logits,labels,mask,num_labels,mask2len)
            #predict,viterbi_score = tf.contrib.crf.crf_decode(logits, trans, mask2len)
            #return (loss, logits,predict)
        
            ##else:
            ##    loss,predict  = softmax_layer(logits, labels, num_labels, mask)
            ##    return (loss, logits, predict)
            # LSTM
            nwords=mask2len
            output = embeddings
            if args.use_lstm:
                t = tf.transpose(output, perm=[1, 0, 2])#[batch_size,sequench_length,dim)
                lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(args.lstm_size)
                lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(args.lstm_size)
                lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
                output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
                output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
                output = tf.concat([output_fw, output_bw], axis=-1)#[sequench_length,batch_size,2*dim)
                output = tf.transpose(output, perm=[1, 0, 2])
                output = tf.layers.dropout(output, rate=args.dropout, training=training)
        

            _,seq,dim = output.shape
            output = tf.reshape(output,shape=[-1,seq * dim])
            #output = tf.transpose(output,perm=[1,0,2])#[sequench_length,batch_size,<2 * >dim)
            #import ipdb
            #ipdb.set_trace()
            #output = tf.reduce_mean(output,axis=0)#[batch_size,<2*>dim]
            # CRF
            logits = tf.layers.dense(output, num_labels)
            #logits = tf.layers.dense(output, 1)
            #crf_params = tf.get_variable("crf", [num_labels, num_labels], dtype=tf.float32)
            #pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)
            ##mask2len = tf.reduce_sum(mask,axis=1)
            #loss, trans = crf_loss(logits,labels,num_labels,mask2len)
            ##log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            ##    logits, tags, nwords, crf_params)
            ##loss = tf.reduce_mean(-log_likelihood)
            probabilities = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                labels = tf.reshape(labels,[-1])
                one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
                per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
                loss = tf.reduce_mean(per_example_loss)
                return (loss, logits,probabilities)
            else:
                return (None, logits,probabilities)
        
        def model_fn(features, labels, mode, params):
            # For serving, features are a bit different
            #if isinstance(features, dict):
            #    features = features['words'], features['nwords']
        
            # Read vocabs and inputs
            dropout = args.dropout
            #input_ids = features["input_ids"]
            #mask = features["mask"]
            #segment_ids = features["segment_ids"]
            #label_ids = features["label_ids"]
            ##words, nwords = features
            #tf.print(' '.join(words[4]), output_stream=sys.stderr)
            training = (mode == tf.estimator.ModeKeys.TRAIN)
            #vocab_words = tf.contrib.lookup.index_table_from_file(
            #    #args.vocab_words)
            #    args.vocab_words, num_oov_buckets=args.num_oov_buckets)
            #with Path(args.vocab_tags).open() as f:
            #    indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
            #    num_tags = len(indices) + 1
        
            ##word_ids = vocab_words.lookup(words)

            input_ids = features["input_ids"]
            mask = features["mask"]
            label_ids = None
            if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
                label_ids = features["label_ids"]

            if args.embedding == 'word2id':
        	    # word2id
                with Path(args.vocab_words).open(encoding='utf-8') as f:
                    vocab_words_1 = f.readlines()
                    vocab_length = len(vocab_words_1)
                #input_ids = features["input_ids"]
                ##label_ids = features["label_ids"]
                #label_ids = None
                #mask = features["mask"]
                embeddings = embedding(input_ids,vocab_length,args)
                embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)
                pass
        
            elif args.embedding == 'bert':
                from my_model.embeddings.embedding import get_bert_embedding
                #input_ids = features["input_ids"]
                #mask = features["mask"]
                segment_ids = features["segment_ids"]
                #label_ids = features["label_ids"]
                embeddings = get_bert_embedding(args.bert_config_file,training,input_ids,mask,segment_ids,use_one_hot_embeddings=False)

            else:
                # Word Embeddings
                # deafult
                #input_ids = features["input_ids"]
                #label_ids = features["label_ids"]
                #mask = features["mask"]
                #glove = np.load(args.glove)['embeddings']  # np.array
                glove = np.load(args.glove)['embeddings']  # np.array
                variable = np.vstack([glove, [[0.]*args.dim]])
                variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
                embeddings = tf.nn.embedding_lookup(variable, input_ids)
                embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)
                #embeddings = tf.nn.dropout(embeddings, rate=dropout, training=training)
                #embeddings = tf.compat.v1.layers.dropout(embeddings, rate=dropout, training=training)
                pass
            
            (total_loss, logits,predicts) = create_model(embeddings,label_ids,mask,mode, training, self.num_labels,use_one_hot_embeddings=False)
            tvars = tf.trainable_variables()
            initialized_variable_names=None
            scaffold_fn = None
            if args.init_checkpoint:
                (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,args.init_checkpoint)
                tf.train.init_from_checkpoint(args.init_checkpoint, assignment_map)
                self.logging.debug("**** Trainable Variables ****")
                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                    self.logging.debug("  name = %s, shape = %s%s", var.name, var.shape,
                                    init_string)
            if mode == tf.estimator.ModeKeys.TRAIN:
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
                elif args.learning_rate_decay == 'bert':
                    num_train_steps = int(
                        self.len_train_examples / args.batch_size * args.epochs)
                    #num_warmup_steps = int(num_train_steps * args.warmup_steps)
                    num_warmup_steps = int(num_train_steps * 0.1)
                    train_op = optimization.create_optimizer(total_loss, args.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)
                    output_spec = tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=total_loss,
                        train_op=train_op,
                        #scaffold_fn=scaffold_fn
                        )
                    return output_spec
                else:
                    self.logging.info('learning rate decay strategy not supported')
                    sys.exit()
                tf.print(lr)
                train_op = tf.train.AdamOptimizer(lr).minimize(
                    total_loss, global_step=tf.train.get_or_create_global_step())
                #return tf.estimator.EstimatorSpec(
                #    mode, loss=loss, train_op=train_op)
        

                #output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    #scaffold_fn=scaffold_fn
                    )

            elif mode == tf.estimator.ModeKeys.EVAL:
                #def metric_fn(label_ids, logits,num_labels,mask):
                #    predictions = tf.math.argmax(logits, axis=-1, output_type=tf.int32)
                #    cm = metrics.streaming_confusion_matrix(label_ids, predictions, num_labels-1, weights=mask)
                #    return {
                #        "confusion_matrix":cm
                #    }
                #    #
                #eval_metrics = (metric_fn, [label_ids, logits, self.num_labels, mask])
                #output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                # Metrics
                #weights = tf.sequence_mask(nwords)
                weights = mask
                #mask2len = tf.reduce_sum(mask,axis=1)
                #weights = tf.sequence_mask(mask2len)
                #pred_ids= tf.math.argmax(logits, axis=-1, output_type=tf.int32)
                pred_ids= tf.argmax(logits, axis=-1, output_type=tf.int32)
                num_label_ids = self.num_labels
                metrics = {
                    'acc': tf.metrics.accuracy(label_ids, pred_ids),
                    #'precision': tf.metrics.precision(label_ids, pred_ids, weights),
                    #'recall': tf.metrics.recall(label_ids, pred_ids, weights),
                    ##'f1': f1(label_ids, pred_ids, weights),
                    #'precision': precision(label_ids, pred_ids, self.num_labels),
                    #'recall': recall(label_ids, pred_ids, self.num_labels),
                    #'f1': f1(label_ids, pred_ids, self.num_labels),
                }
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=metrics
                    #scaffold_fn=scaffold_fn
                    )
            else:
                #output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode, predictions=predicts, 
                    #scaffold_fn=scaffold_fn
                )
            return output_spec

            # LSTM
            #t = tf.transpose(embeddings, perm=[1, 0, 2])
            #lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(args.lstm_size)
            #lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(args.lstm_size)
            #lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
            #output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
            #output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
            #output = tf.concat([output_fw, output_bw], axis=-1)
            #output = tf.transpose(output, perm=[1, 0, 2])
            #output = tf.layers.dropout(output, rate=dropout, training=training)
        
            ## CRF
            #logits = tf.layers.dense(output, num_tags)
            #crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
            #pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)
        
            #if mode == tf.estimator.ModeKeys.PREDICT:
            #    # Predictions
            #    reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            #        args.vocab_tags)
            #    pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
            #    predictions = {
            #        'pred_ids': pred_ids,
            #        'tags': pred_strings
            #    }
            #    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
            #else:
            #    # Loss
            #    vocab_tags = tf.contrib.lookup.index_table_from_file(args.vocab_tags)
            #    tags = vocab_tags.lookup(labels)
            #    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            #        logits, tags, nwords, crf_params)
            #    loss = tf.reduce_mean(-log_likelihood)
        
            #    # MetricS
            #    weights = tf.sequence_mask(nwords)
            #    metrics = {
            #        'acc': tftimator.EstimatorSpecmetrics.accuracy(tags, pred_ids, weights),
            #        'precision': precision(tags, pred_ids, num_tags, indices, weights),
            #        'recall': recall(tags, pred_ids, num_tags, indices, weights),
            #        'f1': f1(tags, pred_ids, num_tags, indices, weights),
            #    }
            #    for metric_name, op in metrics.items():
            #        tf.summary.scalar(metric_name, op[1])
        
            #    if mode == tf.estimator.ModeKeys.EVAL:
            #        return tf.estimator.EstimatorSpec(
            #            mode, loss=loss, eval_metric_ops=metrics)
        
            #    elif mode == tf.estimator.ModeKeys.TRAIN:
            #        warmup_steps = args.warmup_steps
            #        step = tf.to_float(tf.train.get_global_step())
            #        if args.learning_rate_decay == 'sqrt':
            #            lr_warmup = args.learning_rate_peak * tf.minimum(1.0,step/warmup_steps)
            #            lr_decay = args.learning_rate_peak * tf.minimum(1.0,tf.sqrt(warmup_steps/step))
            #            lr = tf.where(step < warmup_steps, lr_warmup, lr_decay)
            #        elif args.learning_rate_decay == 'exp':
            #            lr = tf.train.exponential_decay(args.learning_rate_peak,
            #                    global_step=step,
            #                    decay_steps=args.decay_steps,
            #                    decay_rate=args.decay_rate)
            #        else:
            #            self.logging.info('learning rate decay strategy not supported')
            #            sys.exit()
            #        tf.print(lr)
            #        train_op = tf.train.AdamOptimizer(lr).minimize(
            #            loss, global_step=tf.train.get_or_create_global_step())
            #        return tf.estimator.EstimatorSpec(
            #            mode, loss=loss, train_op=train_op)
        
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
#        return NER_CRF_MulClass.model_fn_builder(args)
#    
#class bilstmCRF_export(ABC_EXPORT,):
#    def __init__(self,args):
#        super().__init__(args)
#    @staticmethod
#    def model_fn_builder(args):
#        return NER_CRF_MulClass.model_fn_builder(args)
    #@staticmethod
    def get_serving_input_receiver_fn(self):
        def serving_input_receiver_fn():
            #words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
            #nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
            #receiver_tensors = {'words': words, 'nwords': nwords}
            #features = {'words': words, 'nwords': nwords}
            #return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
        
            input_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.args.max_seq_length], name='input_ids')
            mask = tf.placeholder(dtype=tf.int32, shape=[None, self.args.max_seq_length], name='mask')
            #segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.args.max_seq_length], name='segment_ids')
            #label_ids = tf.placeholder(dtype=tf.int32, shape=[None, ], name='unique_ids')
        
            receive_tensors = {'input_ids': input_ids, 'mask': mask}#, 'segment_ids': segment_ids}
                               #'label_ids': label_ids}
            features = {'input_ids': input_ids, 'mask': mask}#, 'segment_ids': segment_ids}# "label_ids": label_ids}
            #receive_tensors = {'input_ids': input_ids}
            #features = {'input_ids': input_ids}
            return tf.estimator.export.ServingInputReceiver(features, receive_tensors)
        return serving_input_receiver_fn
