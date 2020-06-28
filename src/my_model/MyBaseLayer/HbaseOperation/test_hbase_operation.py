from utils.hbase_operation import HBaseOperation
import pandas as pd
import numpy as np


def test_Series_cell():
    row_key = 'svc.test.Series'
    indexs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k']
    df = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', '1', 'd', 'e'], index=indexs)
    df['c'] = 'hello world'
    df = df['a']
    df['f'] = None
    print("input:")
    print(df)
    hb.write_data(df, table_name, row_key, 'test_Series')
    df = hb.get_data(table_name, row_key, 'test_Series')
    print("output:")
    print(df)


def test_DataFrame_cell():
    row_key = 'svc.test.DataFrame'
    indexs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k']
    df = pd.DataFrame(np.random.randn(10, 5), columns=['a', 'b', '1', 'd', 'e'], index=indexs)
    df['c'] = 'hello world'
    df.at['a', 'f'] = None
    print("input:")
    print(df)
    hb.write_data(df, table_name, row_key, 'test_DataFrame')
    df = hb.get_data(table_name, row_key, 'test_DataFrame')
    print("output:")
    print(df)


def test_others_cell():
    row_key = 'svc.test.others'
    accuracy = 0.5
    precison = 0.8
    recall = 0.2
    f1 = 0.9
    algrithm = 'svc.svm'
    column_content_list = [('accuracy', accuracy), ('precison', precison),
                           ('recall', recall), ('f1', f1),
                           ('algrithm', algrithm)]

    hb.write_data(column_content_list, table_name, row_key, 'others')
    df = hb.get_data(table_name, row_key, 'others')
    print("input:")
    print(column_content_list)
    print("output:")
    print(df)


def test_dic_cell():
    row_key = 'svc.test.dic'
    accuracy = 0.5
    precison = 0.8
    recall = 0.2
    f1 = 0.9
    algrithm = 'svc.svm'
    data = {'accuracy': accuracy, 'precison': precison,
            'recall': recall, 'f1': f1,
            'algrithm': algrithm}

    hb.write_data(data, table_name, row_key, 'others')
    df = hb.get_data(table_name, row_key, 'others')
    print("input:")
    print(data)
    print("output:")
    print(df)


if __name__ == "__main__":
    table_name = 'test_cell3'
    try:
        hb = HBaseOperation('127.0.0.1', 9090)
        table_cf = {str('test_Series'): dict(max_versions=10),
                    str('test_DataFrame'): dict(max_versions=10),
                    str('others'): dict(max_versions=10)}
        if not hb.is_table_exists(table_name):
            print('{} is not exist, creat it'.format(table_name))
            hb.create_HBase_table(table_name, table_cf)
        test_Series_cell()
        test_DataFrame_cell()
        test_others_cell()
        test_dic_cell()
    finally:
        print('end')
