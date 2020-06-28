from happybase import Connection
from .save_data import save_data_to_cell
from .get_data import get_specify_maximum_version_from_cell
from .get_data import get_specify_versions_data_from_cell


class HBaseOperation(Connection):
    def __init__(self, host, port=9090):
        self.host = host
        self.port = port
        super(HBaseOperation, self).__init__(self.host, self.port, )

    def _re_init(self):
        import time
        time.sleep(1)
        self.close()
        super(HBaseOperation, self).__init__(self.host, self.port)
        self.open()

    def _write_data(self, data_tuple_list, table_name, row_key, cf, timestamp, wal):
        count = 2
        while count > 0:
            try:
                self._re_init()
                return save_data_to_cell(self, data_tuple_list, table_name, row_key, cf, timestamp=timestamp, wal=wal)
            except BrokenPipeError:
                count = count - 1
        raise OSError('retries reach max limits, retry failure')

    def write_data(self, input_data, table_name, row_key, cf, timestamp=None, wal=True):
        """
        :param input_data:
        :type Dataframe, Series, dic, str,  Binary data or data tuple.
        data tuple list:[(describe str, data)]
        :param table_name:
        :param row_key:
        :param cf:
        :return:
        """
        try:
            return save_data_to_cell(self, input_data, table_name, row_key, cf, timestamp=timestamp, wal=wal)
        except BrokenPipeError:
            print('Exception:save data happend BrokenPipeError, retry it')
            return self._write_data(input_data, table_name, row_key, cf, timestamp=timestamp, wal=wal)

    def _get_data(self, table_name, row_key, cf, timestamp, include_timestamp):
        count = 2
        while count > 0:
            try:
                self._re_init()
                return get_specify_maximum_version_from_cell(self, table_name, row_key, cf, timestamp=timestamp,
                                                             include_timestamp=include_timestamp)
            except BrokenPipeError:
                count = count - 1
        raise OSError('retries reach max limits, retry failure')

    def get_specify_maximum_version_data(self, table_name, row_key, cf, timestamp=None, include_timestamp=False):
        try:
            return get_specify_maximum_version_from_cell(self, table_name, row_key, cf, timestamp=timestamp,
                                                         include_timestamp=include_timestamp)
        except BrokenPipeError:
            print('Exception:get data happend BrokenPipeError, retry it')
            return self._get_data(table_name, row_key, cf, timestamp=timestamp, include_timestamp=include_timestamp)

    def get_data(self, table_name, row_key, cf, timestamp=None, include_timestamp=False):
        return self.get_specify_maximum_version_data(table_name, row_key, cf, timestamp, include_timestamp)

    def get_specify_versions_data(self, table_name, row_key, cf, versions=None, timestamp=None,
                                  include_timestamp=False):
        return get_specify_versions_data_from_cell(self, table_name, row_key, cf, versions=versions,
                                                   timestamp=timestamp,
                                                   include_timestamp=include_timestamp)

    @staticmethod
    def img_to_bin(canvas):
        """
        :param canvas:
        fig = plt.figure()
        plt.plot(x, y)
        canvas = fig.canvas
        :return:img bin data
        """
        import io
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        data = buffer.getvalue()
        return data

    @staticmethod
    def bin_to_img(bin_data, name):
        open('{}'.format(name), 'wb').write(bin_data)

    def save_img_for_pdf(self, canvas, table_name, row_key, cf):
        from utils.pdf_report import PDFReport
        data = PDFReport.img_to_data(canvas)
        self.write_data(str(data), table_name, row_key, cf)

    def get_img_for_pdf(self, table_name, row_key, cf):
        data = self.get_specify_maximum_version_data(table_name, row_key, cf)
        if data is None:
            return None
        return data['default']

    def create_HBase_table(self, table_name, table_cf: dict):
        self.create_table(table_name, families=table_cf)

    def is_table_exists(self, table_name):
        table_names = self.tables()
        if bytes(table_name, encoding='utf-8') not in table_names:
            return False
        else:
            return True
