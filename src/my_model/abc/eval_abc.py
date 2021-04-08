from my_model.layers.data_reader import DATA_READER
from abc import abstractmethod,ABC
import tensorflow as tf
import functools
import codecs
import os
from pathlib import Path
class ABC_EVAL(DATA_READER):
    def __init__(self,args):
        super().__init__()
        self.args = args
                                                #self.params)

    #def evaluate_all_train_file(self):
    #    # for file_path in get_file_list(self.input_data, file_type='words.txt',except_file_type='bak'):
    #    for name in ['train', 'testa', 'testb']:
    #        self.evaluate_train_file(name)

    #def evaluate_train_file(self, name):
    #    self.predictions(name, self.f_predict_score(name))
    #    print('start compute {}'.format(name))

    #    self.compute_socres(self.f_predict_score(name))

    #def save_score_file(self):
    #    pass

    @staticmethod
    def compute_scores(output_predict_file,args):
        from my_model.abc import conlleval
        eval_result = conlleval.return_report(output_predict_file)
        print(''.join(eval_result))
        # 写结果到文件中
        with codecs.open(os.path.join(args.output, 'predict_score.txt'), 'w', encoding='utf-8') as fd:
            fd.write(''.join(eval_result))


    @staticmethod
    @abstractmethod
    def model_fn_builder():
        raise NotImplementedError()
    def predict(self):
        # Path('results/score').mkdir(parents=True, exist_ok=True)
        # with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
        session_config = tf.ConfigProto(allow_soft_placement = True,gpu_options=tf.GPUOptions(allow_growth = True))
        cfg = tf.estimator.RunConfig(save_checkpoints_steps=120, save_summary_steps=1,log_step_count_steps=10,keep_checkpoint_max=20).replace(session_config=session_config)
        model_fn = self.model_fn_builder(self.args)
        self.estimator = tf.estimator.Estimator(model_fn, '{}/model'.format(self.args.output), cfg)
        output_file = os.path.join(self.args.output,'preds_score.txt')
        
        with Path(output_file).open('wb') as f:
            test_inpf = functools.partial(self.input_fn, self.fwords(self.args.eval_data_dir,self.args.eval_file), self.ftags(self.args.eval_data_dir,self.args.eval_file),self.args)
            golds_gen = self.generator_fn(self.fwords(self.args.eval_data_dir,self.args.eval_file), self.ftags(self.args.eval_data_dir,self.args.eval_file))
            preds_gen = self.estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                # for word, tag, tag_pred in zip(words, tags, preds['tags']):
                # print(preds)

                tags_tag = [tag for tag in preds['tags']]
                for word, tag, tag_pred in zip(words, tags, tags_tag):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    #def fwords(self, name):
    #    return str(Path(self.input_dir, '{}.words.txt'.format(name)))

    #def ftags(self, name):
    #    return str(Path(self.input_dir, '{}.tags.txt'.format(name)))

    #def f_predict_score(self, name):
            return  output_file 
    def eval_score(self):
    
        output_file = self.predict()
        self.compute_scores(output_file,self.args)
