from my_model.commands import register_subcommand   
from my_model.commands import BaseCLICommand
#from logging import getLogger
import logging
import os
from argparse import ArgumentParser,Namespace
import subprocess 
import shlex
import time

def deploy_command_factory(args: Namespace):
    """
    Factory function used to instantiate serving server from provided command line arguments.
    :return: ServeCommand
    """

    return DeployCommands(args)

#class ExportCommands(BaseTransformersCLICommand):
class DeployCommands(BaseCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        deploy_parser = parser.add_parser("deploy", help="CLI tool to deploy a model on a task.")
        #register_subcommand(deploy_parser)
        deploy_parser.add_argument("--source_path",type=str,required=True,help="save path for model pb ")
        deploy_parser.add_argument(
            "--docker_port", type=int, default=8061, help="docker listening port in host server"
        )
        deploy_parser.add_argument("--target_path",type=str,default='/models',help="docker inner targert dir")
        deploy_parser.add_argument("--docker_name",type=str,required=True,help="docker running name")
        deploy_parser.add_argument("--model_name",type=str,default='saved_model',help="model pb dir name")
        deploy_parser.add_argument("--device_type",type=str,default='cpu',help="docker running on cpu or gpu")
        deploy_parser.add_argument("--device_map",type=str,default='-1',help="device map")
        deploy_parser.add_argument("--log_name",type=str,default='deploy.log',help="deploy logging name ")
        deploy_parser.add_argument("--log_dir",type=str,default='./',help="deploy logging dir")

        deploy_parser.set_defaults(func=deploy_command_factory)

    def __init__(self, args: Namespace):
        #super().__init__(args)
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
        print('cuda device map is {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        self.model = None
        if args.device_type == "cpu":
            cmd = 'setsid /usr/bin/docker run --name {} -p {}:8500 --mount type=bind,source={},target={} -e MODEL_NAME={}  -t tensorflow/serving'.format(args.docker_name,args.docker_port,os.path.join(args.source_path,args.model_name),os.path.join(args.target_path,args.model_name),args.model_name)
            args = shlex.split(cmd)
            pipe = subprocess.Popen(args)
            #pipe = subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            #self.log_subprocess_output(pipe.stdout)
            ##var = os.popen(cmd)
            #var = os.system(cmd)
            #print(var)

        elif args.gpu == "gpu":
            print('nothing running')
        else:
            raise ValueError("you just specify GPU or CPU,your device type is {}".format(args.device_type))
            #rasie ValueError("you just specify GPU or CPU")
            #print("you just specify GPU or CPU")
    def run(self):
        time.sleep(15)
        print('deploy finished.')
        #self.model.deploy_score()

    
