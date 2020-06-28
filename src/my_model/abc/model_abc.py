from abc import ABC, abstractmethod
from argparse import ArgumentParser
from my_model.layers.data_reader import DATA_READER
from my_model.layers.data_reader_v2 import InputFn
from pathlib import Path
import tensorflow as tf
import functools
import codecs
import logging
import sys
import os
#class ABC_MODEL(DATA_READER):
class ABC_MODEL(InputFn):
    def __init__(self,data_dir,output,tags_path,vocab_path,embeddings_type,tokenizer=None,max_seq_length=256,args=None):
        self.indices = None
        self.estimator = None
        self.hook = True
        #import ipdb
        #ipdb.set_trace()
        self.args = args
        #data_dir = args.data_dir
        #tags_path = args.vocab_tags
        #vocab_path = args.vocab_words
        #embeddings_type = args.embedding

        #if not hasattr(self,'output'):
        #    self.output = './output'
        self.output = output
        if args:
            if hasattr(args,'save_path'):
                self.save_path = args.save_path
        Path(self.output).mkdir(exist_ok=True)
        super().__init__(data_dir,tags_path,vocab_path,embeddings_type=embeddings_type,tokenizer=tokenizer,max_seq_length=max_seq_length,args=self.args)
        self.logging = logging.getLogger('tensorflow')
        self._init_logging()
        #self.logging.info('########3')
        self.logging.info(self.output)
    @abstractmethod
    def model_fn_builder(self,args):
        raise NotImplementedError()

        
    def train(self):
        model_fn = self.model_fn_builder(self.args)
        train_inpf = functools.partial(self.input_fn, self.fwords(self.args.data_dir,'train'), self.ftags(self.args.data_dir,'train'),
                                       self.args, shuffle_and_repeat=True)
        eval_inpf = functools.partial(self.input_fn, self.fwords(self.args.data_dir,'testa'), self.ftags(self.args.data_dir,'testa'),
                                       self.args, shuffle_and_repeat=False)
    
        cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
        estimator = tf.estimator.Estimator(model_fn, '{}/model'.format(self.output), cfg, self.args)
        Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        #hook = tf.contrib.estimator.stop_if_no_increase_hook(
        #    estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
        #train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    #@staticmethod
    def compute_scores(self,output_predict_file,args):
        from my_model.abc import conlleval
        #import ipdb
        #ipdb.set_trace()
        eval_result = conlleval.return_report(output_predict_file)
        self.logging.info(''.join(eval_result))
        # 写结果到文件中
        with codecs.open(os.path.join(self.output, 'predict_score.txt'), 'w', encoding='utf-8') as fd:
            fd.write(''.join(eval_result))

    def predict(self):
        # Path('results/score').mkdir(parents=True, exist_ok=True)
        # with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
        session_config = tf.ConfigProto(allow_soft_placement = True,gpu_options=tf.GPUOptions(allow_growth = True))
        cfg = tf.estimator.RunConfig(save_checkpoints_steps=120, save_summary_steps=1,log_step_count_steps=10,keep_checkpoint_max=20).replace(session_config=session_config)
        model_fn = self.model_fn_builder(self.args)
        self.estimator = tf.estimator.Estimator(model_fn, '{}/model'.format(self.output), cfg)
        output_file = os.path.join(self.output,'preds_score.txt')
        
        with Path(output_file).open('wb') as f:
            test_inpf = functools.partial(self.input_fn, self.fwords(self.args.data_dir,self.args.eval_file), self.ftags(self.args.data_dir,self.args.eval_file),self.args)
            golds_gen = self.generator_fn(self.fwords(self.args.data_dir,self.args.eval_file), self.ftags(self.args.data_dir,self.args.eval_file))
            preds_gen = self.estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                # for word, tag, tag_pred in zip(words, tags, preds['tags']):
                # self.logging.info(preds)

                tags_tag = [tag for tag in preds['tags']]
                for word, tag, tag_pred in zip(words, tags, tags_tag):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')
            return  output_file 

    def eval_score(self):
    
        output_file = self.predict()
        self.compute_scores(output_file,self.args)
    #@staticmethod
    #@abstractmethod
    #def serving_input_receiver_fn():
    #    """Serving input_fn that builds features from placeholders
    #
    #    Returns
    #    -------
    #    tf.estimator.export.ServingInputReceiver
    #    """
    #    raise NotImplementedError()
    def export(self):

    
        Path(self.save_path).mkdir(exist_ok=True)
        super().__init__(data_dir,tags_path,vocab_path,embeddings_type=embeddings_type,tokenizer=tokenizer,max_seq_length=max_seq_length,args=self.args)
        self.logging = logging.getLogger('tensorflow')
        self._init_logging()
        #self.logging.info('########3')
        self.logging.info(self.output)
    @abstractmethod
    def model_fn_builder(self,args):
        raise NotImplementedError()

        
    def train(self):
        model_fn = self.model_fn_builder(self.args)
        train_inpf = functools.partial(self.input_fn, self.fwords(self.args.data_dir,'train'), self.ftags(self.args.data_dir,'train'),
                                       self.args, shuffle_and_repeat=True)
        eval_inpf = functools.partial(self.input_fn, self.fwords(self.args.data_dir,'testa'), self.ftags(self.args.data_dir,'testa'),
                                       self.args, shuffle_and_repeat=False)
    
        cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
        estimator = tf.estimator.Estimator(model_fn, '{}/model'.format(self.output), cfg, self.args)
        Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
        #hook = tf.contrib.estimator.stop_if_no_increase_hook(
        #    estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
        #train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
        train_spec = tf.estimator.TrainSpec(input_fn=train_inpf)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    #@staticmethod
    def compute_scores(self,output_predict_file,args):
        from my_model.abc import conlleval
        #import ipdb
        #ipdb.set_trace()
        eval_result = conlleval.return_report(output_predict_file)
        self.logging.info(''.join(eval_result))
        # 写结果到文件中
        with codecs.open(os.path.join(self.output, 'predict_score.txt'), 'w', encoding='utf-8') as fd:
            fd.write(''.join(eval_result))

    def predict(self):
        # Path('results/score').mkdir(parents=True, exist_ok=True)
        # with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
        session_config = tf.ConfigProto(allow_soft_placement = True,gpu_options=tf.GPUOptions(allow_growth = True))
        cfg = tf.estimator.RunConfig(save_checkpoints_steps=120, save_summary_steps=1,log_step_count_steps=10,keep_checkpoint_max=20).replace(session_config=session_config)
        model_fn = self.model_fn_builder(self.args)
        self.estimator = tf.estimator.Estimator(model_fn, '{}/model'.format(self.output), cfg)
        output_file = os.path.join(self.output,'preds_score.txt')
        
        with Path(output_file).open('wb') as f:
            test_inpf = functools.partial(self.input_fn, self.fwords(self.args.data_dir,self.args.eval_file), self.ftags(self.args.data_dir,self.args.eval_file),self.args)
            golds_gen = self.generator_fn(self.fwords(self.args.data_dir,self.args.eval_file), self.ftags(self.args.data_dir,self.args.eval_file))
            preds_gen = self.estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                # for word, tag, tag_pred in zip(words, tags, preds['tags']):
                # self.logging.info(preds)

                tags_tag = [tag for tag in preds['tags']]
                for word, tag, tag_pred in zip(words, tags, tags_tag):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')
            return  output_file 

    def eval_score(self):
    
        output_file = self.predict()
        self.compute_scores(output_file,self.args)
    #@staticmethod
    #@abstractmethod
    #def serving_input_receiver_fn():
    #    """Serving input_fn that builds features from placeholders
    #
    #    Returns
    #    -------
    #    tf.estimator.export.ServingInputReceiver
    #    """
    #    raise NotImplementedError()
    def export(self):

    
        model_fn = self.model_fn_builder(self.args)
        model_fn = self.model_fn_builder(self.args)
    
        estimator = tf.estimator.Estimator(model_fn,self.args.checkpoint_path)
        #estimator.export_saved_model(self.args.save_path, self.serving_input_receiver_fn)
        servering_input_fn = self.get_serving_input_receiver_fn()
        estimator.export_saved_model(self.args.save_path,servering_input_fn)
    #@abstractmethod
    #def eval(self):
        

    #@abstractmethod
    #def predict:

