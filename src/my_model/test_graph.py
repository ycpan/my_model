import argparse
def train_func(args):
    print('ner')
    if args.embedding == 'bert':
        print("ner:bert")
    else:
        print("ner:glove")
def test_func(args):
    print('class')
    if args.embedding == 'bert':
        print("ner:bert")
    else:
        print("ner:glove")
def servering_func(args):
    print('class')
    if args.embedding == 'bert':
        print("ner:bert")
    else:
        print("ner:glove")
def evaluation_func(args):
    print('class')
    if args.embedding == 'bert':
        print("ner:bert")
    else:
        print("ner:glove")
parser = argparse.ArgumentParser(prog='test_parser', add_help=True)
#添加子命令 add
subparsers = parser.add_subparsers(help='sub-command help')
train = subparsers.add_parser('train', help='trainning a model')
train.add_argument('-task', required=True,choices=['ner','class'],type=str,help='specify a task name,such as ner,class')
train.add_argument('-embedding', choices=['bert','glove','word2vec'],default='bert',type=str,help='glove or bert or word2vec')
train.add_argument('-optimization',choices=['adam','gbdt'],default='adam',type=str,help='adam or gbdt')
#添加子命令 sub
servering = subparsers.add_parser('servering', help='do text class')
servering.add_argument('-x', type=int, help='x value')
servering.add_argument('-y', type=int, help='y value')
test = subparsers.add_parser('test', help='do text class')
test.add_argument('-x', type=int, help='x value')
test.add_argument('-y', type=int, help='y value')
evaluation = subparsers.add_parser('evaluation', help='do text class')
evaluation.add_argument('-x', type=int, help='x value')
evaluation.add_argument('-y', type=int, help='y value')
#embedding = parser.add_argument_group('Required parameters')
#embedding.add_argument('-embedding', choices=['bert','glove'],type=str,help='glove or bert or word2vec')
#embedding.add_argument('-optimization',type=str,help='adam or gbdt')
#network = parser.add_argument_group('Optional parameters')
#network.add_argument('-is_eval', action='store_true',default=True,help='if True ,do eval for epoche')
#network.add_argument('-is_servering', action='store_false',default=False,help='bilstmCRF help')
#
#parser.print_help()
train.set_defaults(func=train_func)
test.set_defaults(func=test_func)
args = parser.parse_args()
#执行函数功能
args.func(args)
