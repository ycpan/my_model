"""Export model as a saved_model"""

__author__ = "Guillaume Genthial"

from pathlib import Path
import json
from abc import abstractmethod,ABC
from argparse import ArgumentParser
import tensorflow as tf

from my_model.commands.train import TrainCommands
from my_model.commands import register_subcommand
class ABC_EXPORT:
    def __init__(self,args):
        self.name = 'ABCExport'
        #self.checkpoint_path = args.checkpoint_path
        #self.save_path = save_path
        self.args = args
    @staticmethod
    @abstractmethod
    def serving_input_receiver_fn():
        """Serving input_fn that builds features from placeholders
    
        Returns
        -------
        tf.estimator.export.ServingInputReceiver
        """
        raise NotImplementedError()
    
    
    @staticmethod
    @abstractmethod
    def model_fn_builder(args):
        #model_fn = BILSTM_CRF_NER_MODEL.model_fn_builder(args)
        raise NotImplementedError()
    def export(self):
    
        model_fn = self.model_fn_builder(self.args)
    
        estimator = tf.estimator.Estimator(model_fn,self.args.checkpoint_path)
        estimator.export_saved_model(self.args.save_path, self.serving_input_receiver_fn)
