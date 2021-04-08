import logging
from HbaseOperation.utils.hbase_operation import HBaseOperation


class MyBaseClass(logging, HBaseOperation):
    def __init__(self):
        self.__name__ = 'MyBaseClass'
