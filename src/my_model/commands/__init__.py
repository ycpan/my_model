from abc import ABC, abstractmethod
from argparse import ArgumentParser


class BaseCLICommand():
    def log_subprocess_output(self,pipe):
        for line in iter(pipe.readline, b''):  # b'\n'-separated lines,遇到b''时停止，否则一致迭代下去。
            line = line.decode()
            print(line)
    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
def register_subcommand(train_parser: ArgumentParser):
#def register_subcommand(train_parser):
    """
    Register this command to argparse so it's available for the transformer-cli
    :param parser: Root parser to register command-specific arguments
    :return:
    """

    train_parser.add_argument(
        "--column_label", type=int, default=0, help="Column of the dataset csv file with example labels."
    )
    train_parser.add_argument(
        "--column_text", type=int, default=1, help="Column of the dataset csv file with example texts."
    )
    train_parser.add_argument("--data_dir",type=str,default="./",help="train or eval  input dir")
    train_parser.add_argument(
        "--column_id", type=int, default=2, help="Column of the dataset csv file with example ids."
    )
    train_parser.add_argument(
        "--buffer", type=int, default=15000, help="select random dataset from buffer size sub data in all data")
    train_parser.add_argument(
        "--epochs", type=int, default=10, help="repeat  epochs size data train")
    train_parser.add_argument(
        "--max_seq_length", type=int, default=256, help="max_seq_length size")
    train_parser.add_argument(
        "--skip_first_row", action="store_true", help="Skip the first row of the csv file (headers)."
    )

    train_parser.add_argument("--validation_data", type=str, default="", help="path to validation dataset.")
    train_parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="if validation dataset is not provided, fraction of train dataset " "to use as validation dataset.",
    )

    train_parser.add_argument("--output", type=str, default="results", help="path to saved the trained model.")
    train_parser.add_argument("--init_checkpoint", type=str,  help="path to pretrained model.")
    train_parser.add_argument("--device_map", type=str, default="-1", help="which GPU model train on.")

    train_parser.add_argument(
        "--embedding", type=str, default="glove",choices=['glove','word2id','bert','plain'], help="Embedding to train the model on."
    )
    train_parser.add_argument(
        "--embedding_stddev", type=float, default=0.01, help="Embedding to train the model on."
    )
    train_parser.add_argument(
        "--task", type=str, default="text_classification", help="Task to train the model on."
    )
    train_parser.add_argument(
        "--model_name", type=str, default="bilstm_crf", help="to train which model."
    )
    train_parser.add_argument(
        "--log_dir", type=str, default="output", help="log dir "
    )
    train_parser.add_argument(
        "--log_name", type=str, default="my_model", help="log name "
    )
    #train_parser.add_argument(
    #    "--model", type=str, default="bert-base-uncased", help="Model's name or path to stored model."
    #)
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    train_parser.add_argument("--num_oov_buckets", type=int, default=1, help="num_oov_bucket.")
    train_parser.add_argument("--vocab_words", type=str,default='',  help="vocab_words path")
    train_parser.add_argument("--vocab_tags", type=str ,default='', help="vocab_tags path")
    train_parser.add_argument("--vocab_chars", type=str ,default='', help="vocab_chars path")
    train_parser.add_argument("--glove", type=str,  help="glove path")
    train_parser.add_argument("--bert_config_file", type=str,  help="bert_config_file path")
    train_parser.add_argument("--valid_batch_size", type=int, default=64, help="Batch size for validation.")
    train_parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate.")
    train_parser.add_argument("--learning_rate_decay", type=str, default='exp', help="decay method:exp or sqrt")
    train_parser.add_argument("--learning_rate_peak", type=float, default=0.01, help="初始化学习率 ")
    train_parser.add_argument("--decay_steps", type=int, default=100, help="每多少步衰减")
    train_parser.add_argument("--lstm_size", type=int, default=128, help="lstm cell size")
    train_parser.add_argument("--use_lstm", type=bool, default=False, help="if use_lstm ,pleash specify this,--use_lstm")
    train_parser.add_argument("--use_cnn", type=bool, default=False, help="if use_cnn ,pleash specify this,--use_cnn")
    train_parser.add_argument("--use_dense", type=bool, default=False, help="if use_dense ,pleash specify this,--use_dense")
    train_parser.add_argument("--decay_rate", type=float, default=0.95, help="每次衰减称的系数")
    train_parser.add_argument("--warmup_steps", type=int, default=500, help="warrm up")
    train_parser.add_argument("--dim", type=int, default=128, help="embedding size")
    train_parser.add_argument("--dropout", type=float, default=0.5, help="dropout")
    train_parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon for Adam optimizer.")
    #return train_parser
