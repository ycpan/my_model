import os
from argparse import ArgumentParser, Namespace
from logging import getLogger

#from my_model import SingleSentenceClassificationProcessor as Processor
#from my_model import TextClassificationPipeline, is_tf_available, is_torch_available
from my_model import is_tf_available
from my_model.commands import register_subcommand   
if not is_tf_available() :
    raise RuntimeError("At least TensorFlow 1.2+ should be installed to use CLI training")

# TF training parameters
USE_XLA = False
USE_AMP = False


def train_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.
    :return: ServeCommand
    """

    return TrainCommands(args)

#class TrainCommands(BaseTransformersCLICommand):
class TrainCommands:
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        train_parser = parser.add_parser("train", help="CLI tool to train a model on a task.")
        train_parser.add_argument(
            "--train_data",
            type=str,
            help="path to train "
        )
        train_parser.add_argument("--eval_data",type=str,help="eval score from name")
        register_subcommand(train_parser)
        train_parser.set_defaults(func=train_command_factory)
    def __init__(self, args: Namespace):
        self.logger = getLogger("my_model-cli/training")

        #self.framework = "tf" if is_tf_available() else "torch"

#        os.makedirs(args.output, exist_ok=True)
#        assert os.path.isdir(args.output)
#        self.output = args.output
#
#        self.column_label = args.column_label
#        self.column_text = args.column_text
#        self.column_id = args.column_id
#
#        self.logger.info("Loading {} pipeline for {}".format(args.task, args.model))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        self.model = None
        if args.task.lower() == "ner":
            #from my_model.task.ner.bilstm_crf_ner_model import BILSTM_CRF_NER_MODEL
            from my_model.task.ner.bert_bilstm_crf import NER_MODEL
            if args.model_name == 'bilstm_crf':
                self.model = NER_MODEL(args)
        elif args.task.lower() == "class":
            #from my_model.task.ner.bilstm_crf_ner_model import BILSTM_CRF_NER_MODEL
            if args.model_name == 'mul_class':
                from my_model.task.task_class.mul_class import MulClass
                self.model = MulClass(args)
            elif args.model_name == 'mul_tags':
                from my_model.task.task_class.mul_tags import MulTags
                self.model = MulTags(args)
        elif args.task.lower() == "regression":
            #from my_model.task.ner.bilstm_crf_ner_model import BILSTM_CRF_NER_MODEL
            if args.model_name == 'regression':
                from my_model.task.regression.base_regression import Regression
                self.model = Regression(args)
            if args.model_name == 'dnn_regression':
                from my_model.task.regression.dnn_regression import Regression
                self.model = Regression(args)
    def run(self):
        print('start train ...')
        self.model.train()
    
