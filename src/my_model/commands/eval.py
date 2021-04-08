from my_model.commands import register_subcommand   
from logging import getLogger
import os
from argparse import ArgumentParser,Namespace
def eval_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.
    :return: ServeCommand
    """

    return EvalCommands(args)

#class ExportCommands(BaseTransformersCLICommand):
class EvalCommands:
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        eval_parser = parser.add_parser("eval", help="CLI tool to eval a model on a task.")
        register_subcommand(eval_parser)
        eval_parser.add_argument("--save_path",type=str,default="scores",help="save path for evalions")
        eval_parser.add_argument("--test_data",type=str,help="test score from name")
        eval_parser.add_argument("--eval_file",type=str,default="testb",help="eval score from name")

        eval_parser.set_defaults(func=eval_command_factory)
    def __init__(self, args: Namespace):
        self.logger = getLogger("my_model-cli/evaling")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        self.model = None
        if args.task == "ner":
            #from my_model.task.ner.bilstm_crf_ner_model import bilstmCRF_eval
            from my_model.task.ner.bilstm_crf_ner_model import BILSTM_CRF_NER_MODEL
            if args.model_name == 'bilstm_crf':
                self.model = BILSTM_CRF_NER_MODEL(args)
        elif args.task == "class":
            #from my_model.task.ner.bilstm_crf_ner_model import bilstmCRF_eval
            from my_model.task.task_class.base_class import CLASS_MODEL
            if args.model_name == 'base_class':
                self.model = CLASS_MODEL(args)
    def run(self):
        print('start eval ...')
        self.model.eval_score()
    
