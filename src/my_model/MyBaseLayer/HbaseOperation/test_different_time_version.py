# from utils.log_to_hbase import Log_To_HBase
from .utils.hbase_operation import HBaseOperation
import datetime
import time
import pandas as pd
import numpy as np
def get_row_key():
    dt = datetime.datetime.time()
    return '{}.{}'.format(table_name, dt)

def test_DataFrame_cell():
    row_key = 'svc.test.DataFrame'
    indexs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k']
    df = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', '1', 'd', 'e'], index=indexs)
    df['c'] = 'hello world'
    df.at['a', 'f'] = None
    print("input1:")
    print(df)
    lh.write_data(df, table_name, row_key, 'test_DataFrame',1562067035309)

    df = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', '1', 'd', 'e'], index=indexs)
    print("input2:")
    print(df)
    lh.write_data(df, table_name, row_key, 'test_DataFrame',1562067035310)

    df = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'f', '1', 'd', 'e'], index=indexs)
    print("input3:")
    print(df)
    lh.write_data(df, table_name, row_key, 'test_DataFrame', 1562067035311)

    df = lh.get_specify_maximum_version_data(table_name, row_key, 'test_DataFrame', timestamp=1562067035311 + 1, include_timestamp=False)
    print("get maximum_version under 1562067035311 data:")
    print(df)
    print('\n\n\n')


    # print("test get_specify_maximum_version_data ")
    print("get all versions data under 15620670353012 output:")
    df = lh.get_specify_versions_data(table_name, row_key, 'test_DataFrame',timestamp=1562067035309 + 3, include_timestamp=True)
    for s ,ts in df:
        print(ts)
        print(s)
    print('\n\n\n')

    print("get maximum two versions data under 15620670353010 output:")
    df = lh.get_specify_versions_data(table_name, row_key, 'test_DataFrame', versions=2, timestamp=1562067035309 + 3, include_timestamp=True)
    for s, ts in df:
        print(ts)
        print(s)
    print('\n\n\n')

def test_dict():
    params = {
        'dim': 256,
        'dropout': 0.3,
        'num_oov_buckets': 1,
        'epochs': 35,
        # 'batch_size': 20,
        'batch_size': 100,
        'buffer': 15000,
        # 'lstm_size': 100,
        'lstm_size': 112,
        'learnning_rate': 0.0002

    }
    row_key = 'label_define.Bilstm_CRF'
    lh.write_data(params, table_name, row_key, 'params')
    # print(ts)
    print('input1:')
    print(params)
    params['learnning_rate'] = 0.005
    lh.write_data(params, table_name, row_key, 'params')

    print('input2')
    print(params)
    print('\n\n\n')

    df = lh.get_specify_maximum_version_data(table_name, row_key, 'params', include_timestamp=True)
    print('the maximum version data is:')
    print(df)
    df = lh.get_specify_versions_data(table_name, row_key, 'params', include_timestamp=True)
    print('all versions data is:')
    for s, ts in df:
        print(ts)
        print(s)
table_name = 'test_Bilstm_CRF'

table_cf = {str('params'): dict(max_versions=100),
            str('score'): dict(max_versions=100),
            str('others'): dict(max_versions=100),
            str('test_DataFrame'): dict(max_versions=100)}
lh = HBaseOperation('127.0.0.1', 9090)
ts = int(time.time())
# test_DataFrame_cell()
test_dict()
# a=lh.get_specify_maximum_version_data(table_name, 'svc.test.DataFrame', 'test_DataFrame',1562060397740,include_timestamp=True)
# print(a)
