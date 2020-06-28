import pickle
import pandas as pd
import struct
import time


def save_to_pickle(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file)


def save_df_to_hbase(df, con, table_name, key, cf='hb',timestamp=None, wal=True):
    """Write a pandas DataFrame object to HBase table.
    :param df: pandas DataFrame object that has to be persisted
    :type df: pd.DataFrame
    :param con: HBase connection object
    :type con: happybase.Connection
    :param table_name: HBase table name to which the DataFrame should be written
    :type table_name: str
    :param key: row key to which the dataframe should be written
    :type key: str
    :param cf: Column Family name
    :type cf: str
    """
    table_names = con.tables()
    if bytes(table_name, encoding='utf-8') not in table_names:
        con.create_table(table_name, families={
            str(cf): dict(max_versions=10),
        })
    table = con.table(table_name)

    column_dtype_key = key + 'columns'
    column_dtype_value = dict()
    column_order_key = key + 'column_order'
    column_order_value = dict()
    row_key_template = key + '_rows_'
    if isinstance(df, pd.DataFrame):
        for column in df.columns:
            column_dtype_value[':'.join((cf, str(column)))] = df.dtypes[column].name
        for i, column_name in enumerate(df.columns.tolist()):
            column_order_value[':'.join((cf, str(i)))] = str(column_name)

        with table.batch(timestamp=timestamp,transaction=True,wal=wal) as b:
            b.put(column_dtype_key, column_dtype_value,wal=wal)
            b.put(column_order_key, column_order_value,wal=wal)
            for column in df.columns:

                row_key = row_key_template + str(column)
                row_value = dict()
                for index, value in df[column].items():
                    if not pd.isnull(value):
                        row_value[':'.join((cf, str(index)))] = str(value)
                        b.put(row_key, row_value,wal=wal)
    if isinstance(df, pd.Series):
        column_dtype_value['{}:'.format(cf)] = df.dtypes.name
        column_order_value[':'.join((cf, str(0)))] = str('')
        with table.batch(timestamp=timestamp,transaction=True,wal=wal) as b:
            b.put(column_dtype_key, column_dtype_value,wal=wal)
            b.put(column_order_key, column_order_value,wal=wal)
            count = 0

            column = df.name
            row_key = row_key_template + str(column)
            row_value = dict()
            for index, value in df.items():
                if not pd.isnull(value):
                    row_value[':'.join((cf, str(index)))] = str(value)
                    b.put(row_key, row_value,wal=wal)
            # b.put(row_key, row_value)
            print(count)
            count += 1

            b.put(row_key, row_value,wal=wal)
            print(count)
            count += 1
            print('tests')


def save_data_to_cell(con, input_data, table_name, row_key, cf,timestamp=None, wal=True):
    """
    
    :param con: 
    :param input_data:
    :type Dataframe, Series, dic, str,  Binary data or data tuple.
    data tuple list:[(describe str, data)]
    :param table_name: 
    :param row_key: 
    :param cf: 
    :return: 
    """
    import thriftpy
    # from builtins import BrokenPipeError
    try:
        table_names = con.tables()
    except thriftpy.transport.TTransportException:
        print("Exception:TTransportException, cf may be not exist")
        return -1
    except BrokenPipeError:
        print("Exception:BrokenPipeError, retry it")
        return -1
    if bytes(table_name, encoding='utf-8') not in table_names:
        raise NameError("the table {} have not in this database".format(table_name))
    table = con.table(table_name)
    # with table.batch(timestamp=timestamp,transaction=True,wal=wal) as b:
    #     b.delete(row_key, {cf})
    #     b.send()
    if isinstance(input_data, list) is False:
        input_data = [('default', input_data)]
    for desc, data in input_data:


        if isinstance(data, pd.DataFrame):
            dtype_column_qualifier = 'DataFrame_columnsType'
            order_column_qualifier = 'DataFrame_columnsOrder'
            dtype_column_value = dict()
            data_qualifier_prefix = "row_"
            type_dic = {column: data.dtypes[column].name for column in data.columns}

            dtype_column_value[':'.join((cf, str(dtype_column_qualifier)))] = str(type_dic)
            order_list = [str(column) for column in data.columns]
            order_column_value = dict()
            order_column_value[':'.join((cf, str(order_column_qualifier)))] = str(order_list)

            with table.batch(timestamp=timestamp,transaction=True,wal=wal) as b:
                b.put(row_key, dtype_column_value,wal=wal)
                b.put(row_key, order_column_value,wal=wal)
                data = data.fillna('None')
                for index, value in data.iterrows():
                    row_value = dict()
                    data_qualifier = data_qualifier_prefix + str(index)
                    row_value[':'.join((cf, str(data_qualifier)))] = str(value.tolist())
                    b.put(row_key, row_value,wal=wal)
        elif isinstance(data, pd.Series):
            dtype_column_qualifier = 'Series_columnsType'
            series_name_qualifier = 'Series_SeriesName'
            data_qualifier_prefix = "row_"
            dtype_column_value = dict()
            series_name_value = dict()
            dtype_column_value[':'.join((cf, dtype_column_qualifier))] = str(data.dtypes.name)
            series_name_value[':'.join((cf, series_name_qualifier))] = str(data.name)
            with table.batch(timestamp=timestamp,transaction=True,wal=wal) as b:
                b.put(row_key, dtype_column_value,wal=wal)
                b.put(row_key, series_name_value,wal=wal)
                row_value = dict()
                for index, value in data.items():
                    data_qualifier = data_qualifier_prefix + str(index)
                    row_value[':'.join((cf, str(data_qualifier)))] = str(value)
                    # b.put(row_key, row_value)
                b.put(row_key, row_value,wal=wal)
        # elif isinstance(data, dict):
        #     data_qualifier = 'dict' + desc

        else:
            data_type = type(data).__name__
            # data_qualifier = 'Others_' + desc
            data_qualifier = '{}_'.format(data_type) + desc
            # value = {':'.join((cf, data_qualifier)): str(data)}
            if isinstance(data,bytes):
                value = {':'.join((cf, data_qualifier)): data}
            else:
                value = {':'.join((cf, data_qualifier)): bytes(str(data), encoding='utf-8')}
            with table.batch(timestamp=timestamp,transaction=True,wal=wal) as b:
                # print('test')
                b.put(row_key, value)
            # table.put(row_key, value,timestamp=1562056346390,wal=False)
        # print('{} write successfully'.format(desc))
    return 0
