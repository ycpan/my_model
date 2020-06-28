import pandas as pd
import pickle
import happybase
import struct
import numpy as np
import sys
# from influxdb import DataFrameClient
import re
import time
from datetime import datetime, timezone, timedelta


# from utils.conf import sql_db_configs


def get_csv_data(path, header=None):
    """load padas dataframe from csv file

    Arguments:
        path {str} -- filepath of the csv file

    Returns:
        pandas.DataFrame -- loaded data
    """
    return pd.read_csv(path, sep=',', encoding='utf-8', header=header)


def get_pickle_data(path):
    """load data from pickle file

    Arguments:
        path {str} -- filepath of the pickle file

    Returns:
        object -- loaded data
    """
    with open(path, 'rb') as file:
        return pickle.load(file)


def get_df_from_hbase(con, table_name, key, cf='hb', timestamp=None, include_timestamp=False):
    """Read a pandas DataFrame object from HBase table.

    Arguments:
        con {happybase.Connection} -- HBase connection object
        table_name {str} -- HBase table name to read
        key {str} -- row key from which the DataFrame should be read

    Keyword Arguments:
        cf {str} -- Column Family name (default: {'hb'})

    Returns:
        [pandas.DataFrame] -- Pandas DataFrame object read from HBase
    """

    table = con.table(table_name)

    column_dtype_key = key + 'columns'
    column_dtype = table.row(column_dtype_key, columns=[cf], timestamp=timestamp, include_timestamp=include_timestamp)
    columns = {col.decode().split(':')[1]: value.decode() for col, value in column_dtype.items()}

    column_order_key = key + 'column_order'
    column_order_dict = table.row(column_order_key, columns=[cf], timestamp=timestamp,
                                  include_timestamp=include_timestamp)
    column_order = list()
    for i in range(len(column_order_dict)):
        column_order.append(column_order_dict[bytes(':'.join((cf, str(i))), encoding='utf-8')].decode())

    # # row_start = key + 'rows' + struct.pack('>q', 0)
    # row_start = key + 'rows' + str(column_order(0))
    # # row_end = key + 'rows' + struct.pack('>q', sys.maxint)
    # row_end = key + 'rows' + str(column_order[len(column_order) - 1])
    row_key_template = key + '_rows_'
    # scan_columns = ['{}{}'.format(row_key_template, item) for item in column_order]
    HBase_rows = table.scan(row_prefix=bytes(row_key_template, encoding='utf-8'))
    # HBase_rows = table.scan(columns='cf:')
    df = pd.DataFrame()
    for row in HBase_rows:
        column_name = row[0].decode().split('_')[len(row[0].decode().split('_')) - 1]
        df_column = {key.decode().split(':')[1]: value.decode() for key, value in row[1].items()}
        pd_series = pd.Series(df_column)
        # df = df.append(df_column, ignore_index=True)
        df[column_name] = pd_series

    for column, data_type in columns.items():
        if len(list(columns.items())) == 1:
            column = df.columns[0]
        if column == '':
            continue
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            pass
        df[column] = df[column].astype(np.dtype(data_type))
    return df


def get_specify_maximum_version_from_cell(con, table_name, row_key, cf='hb', timestamp=None, include_timestamp=False):
    table = con.table(table_name)

    cell = table.row(row_key, columns=[cf], timestamp=timestamp, include_timestamp=include_timestamp)
    # cell1 = table.cells(row_key, column=cf, versions=5, timestamp=timestamp,
    #           include_timestamp=True)
    type_set = set()
    columnsOrder = None
    SeriesName = None
    columnsType = None
    columnsOrder_cf = None
    SeriesName_cf = None
    columnsType_cf = None
    res = None
    for cf, value in cell.items():
        if len(value) == 2:
            value, ts = value
        else:
            value = value
            ts = None
        cf_qualifier = cf.decode().split(':')[1]
        data_type = cf_qualifier.split('_')[0]
        type_set.add(data_type)
        data_content = cf_qualifier.split('_')[1]
        if data_content == 'columnsOrder':
            columnsOrder = eval(value.decode())
            columnsOrder_cf = cf
        if data_content == 'SeriesName':
            SeriesName = value.decode()
            SeriesName_cf = cf
        if data_content == 'columnsType':
            try:
                columnsType = eval(value.decode())
            except NameError:
                columnsType = value.decode()
            columnsType_cf = cf

    if columnsOrder_cf is not None:
        cell.pop(columnsOrder_cf)
    if SeriesName_cf is not None:
        cell.pop(SeriesName_cf)
    if columnsType_cf is not None:
        cell.pop(columnsType_cf)
    cell_keys = cell.keys()

    if columnsOrder_cf in cell_keys or SeriesName_cf in cell_keys or columnsType_cf in cell_keys:
        raise ValueError('more than one clean_log input one cell')
    # if len(type_set) > 2:
    #     raise ValueError('more than one clean_log input one cell')
    # if len(type_set) >= 2:
    #     raise ValueError('in one cell may have two data type, this can not deal it')

    if 'DataFrame' in type_set:
        if len(type_set) > 2:
            raise ValueError('in one cell may have two data type, this can not deal it')
        res = pd.DataFrame()
        for cf, value in cell.items():
            if len(value) == 2:
                value, _ = value
            else:
                value = value
            cf_qualifier = cf.decode().split(':')[1]
            data_index = cf_qualifier.split('_')[1]
            value = eval(value.decode())
            if 'None' in value:
                value = [None if v == 'None' else v for v in value]

            df_sub = pd.DataFrame(np.array(value).reshape(1, -1),
                                  columns=columnsOrder, index=[data_index])
            res = res.append(df_sub)
        for column, data_type in columnsType.items():
            if column == '':
                continue
            try:
                res[str(column)] = pd.to_numeric(res[str(column)])
            except ValueError:
                pass
            res[str(column)] = res[str(column)].astype(np.dtype(data_type))
    elif 'Series' in type_set:
        if len(type_set) > 2:
            raise ValueError('in one cell may have two data type, this can not deal it')
        res = pd.Series()
        for cf, value in cell.items():
            if len(value) == 2:
                value, _ = value
            else:
                value = value
            cf_qualifier = cf.decode().split(':')[1]
            data_index = cf_qualifier.split('_')[1]
            df_sub = pd.Series(value.decode(), index=[data_index])

            res = res.append(df_sub)
        if SeriesName is not None:
            res.name = SeriesName
        try:
            res = pd.to_numeric(res)
        except ValueError:
            pass
        res = res.astype(np.dtype(columnsType))
    elif 'dict' in type_set:
        if len(type_set) >= 2:
            raise ValueError('in one cell may have two data type, this can not deal it')
        # for cf, value in cell.items():
        if len(value) == 2:
            value, _ = value
        else:
            value = value
        res = eval(value.decode())

    else:
        res = dict()
        for cf, value in cell.items():
            if len(value) == 2:
                value, _ = value
            else:
                value = value
            cf_qualifier = cf.decode().split(':')[1]
            data_key = cf_qualifier.split('_')[1]
            # value = value.decode()
            value = value
            try:
                value = value.decode()
                value = eval(value)
            except:
                pass
            res[data_key] = value
    if ts is not None:
        return res, ts
    return res


def get_specify_versions_data_from_cell(con, table_name, row_key, cf='hb', versions=None, timestamp=None, include_timestamp=False):
    table = con.table(table_name)

    # cell = table.row(row_key, columns=[cf], timestamp=timestamp, include_timestamp=include_timestamp)
    cell = table.cells(row_key, column=cf, versions=versions, timestamp=timestamp,
              include_timestamp=True)
    ts_set = set()
    for _, ts in cell:
        ts_set.add(ts)
    res = []
    for ts in ts_set:
        res.append(get_specify_maximum_version_from_cell(con, table_name, row_key, cf, ts+1, include_timestamp))
    return res

