#from .visualisation import ohlc_chart, view

#from .file_handler import open_file, load_products, dataframe

#from .density_table_models import OHLC, Density

#from .factory import norm, ohlc_chart, set_ATR, rolling_window, split


"""
여러 코드에서 공통적으로 사용하는 기능들 
"""
import sys
import inspect

def monitor_memory(dirr):
    """
    jupyter notebook에서 사용중인 오브젝트 들의 메모리 사용량 체크 
    reference:
    https://stackoverflow.com/questions/6946376/how-to-reload-a-class-in-python-shell
    
    실행법: 
    monitor_momery(dir())
    
    """
    # These are the usual ipython objects, including this one you are creating
    ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
    glob = inspect.stack()[1][0].f_globals

    # Get a sorted list of the objects and their sizes
    r = sorted([(x, int(sum([sys.getsizeof(glob.get(x))]+[sys.getsizeof(getattr(glob.get(x), attr)) for attr in dir(glob.get(x))])/1024)) for x in dirr if not x.startswith('_')\
         and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)

    print(r)