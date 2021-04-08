import logging
import logging.handlers
import os
import sys
from pathlib import Path
class MY_BASE:
    def __init__(self,log_dir,log_name,args=None):
        self.__name__ = 'MY_BASE'
        self.logging = logging.getLogger('BASE')
        self.args = args
        self.log_dir = log_dir
        self.log_name = log_name
        Path(log_dir).mkdir(exist_ok=True)
        self._init_logging()
    def _init_logging(self,when="H",interval=1,backupCount=2):
        self.logging.setLevel(logging.INFO)
        self.log_path = os.path.join(self.log_dir,self.log_name)
        fh = logging.handlers.TimedRotatingFileHandler(filename=self.log_path, when=when, interval=interval, backupCount=backupCount)

        # 设置后缀名称，跟strftime的格式一样
        fh.suffix = "%Y%m%d-%H%M.log"
        #fh.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}.log$")
        fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
# 第三步，定义handler的输出格式
        formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)
        self.logging.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel((logging.DEBUG))  # 输出到console的log等级的开关
        #ch.setFormatter(formatter)
        self.logging.addHandler(ch)
        self.logging.info('logging file:{}'.format(self.log_path))

