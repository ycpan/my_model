import os
from argparse import ArgumentParser, Namespace
from logging import getLogger
from my_model import is_tf_available
from my_model.commands import register_subcommand   
def export_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.
    :return: ServeCommand
    """

    return ExportCommands(args)

#class ExportCommands(BaseTransformersCLICommand):
class ExportCommands:
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        export_parser = parser.add_parser("export", help="CLI tool to export a model on a task.")
        register_subcommand(export_parser)
        export_parser.add_argument("--checkpoint_path",required=True,type=str,default="saved_model",help="checkpoint path")
        export_parser.add_argument("--save_path",required=True,type=str,default="saved_model",help="save path")

        export_parser.set_defaults(func=export_command_factory)
    def __init__(self, args: Namespace):
        self.logger = getLogger("my_model-cli/exporting")
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
        print(os.environ['CUDA_VISIBLE_DEVICES'])
        self.model = None
        if args.task == "ner":
            #from my_model.task.ner.bilstm_crf_ner_model import bilstmCRF_export
            from my_model.task.ner.bilstm_crf_ner_model import BILSTM_CRF_NER_MODEL
            if args.model_name == 'bilstm_crf':
                #self.model = bilstmCRF_export(args)
                self.model = BILSTM_CRF_NER_MODEL(args)
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
    def run(self):
        print('start export ...')
        self.model.export()
    
