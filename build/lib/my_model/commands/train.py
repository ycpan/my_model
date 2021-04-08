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
            required=True,
            help="path to train (and optionally evaluation) dataset as a csv with "
            "tab separated labels and sentences.",
        )
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
        if args.task == "ner":
            from my_model.task.ner.bilstm_crf_ner_model import BILSTM_CRF_NER_MODEL
            if args.model_name == 'bilstm_crf':
                self.model = BILSTM_CRF_NER_MODEL(args)
                #print('start train ...')
                #self.model.train()
#        elif args.task == "token_classification":
#            raise NotImplementedError
#        elif args.task == "question_answering":
#            raise NotImplementedError
#
#        self.logger.info("Loading dataset from {}".format(args.train_data))
#        self.train_dataset = Processor.create_from_csv(
#            args.train_data,
#            column_label=args.column_label,
#            column_text=args.column_text,
#            column_id=args.column_id,
#            skip_first_row=args.skip_first_row,
#        )
#        self.valid_dataset = None
#        if args.validation_data:
#            self.logger.info("Loading validation dataset from {}".format(args.validation_data))
#            self.valid_dataset = Processor.create_from_csv(
#                args.validation_data,
#                column_label=args.column_label,
#                column_text=args.column_text,
#                column_id=args.column_id,
#                skip_first_row=args.skip_first_row,
#            )
#
#        self.validation_split = args.validation_split
#        self.train_batch_size = args.train_batch_size
#        self.valid_batch_size = args.valid_batch_size
#        self.learning_rate = args.learning_rate
#        self.adam_epsilon = args.adam_epsilon
#
    def run(self):
        print('start train ...')
        self.model.train()
    
