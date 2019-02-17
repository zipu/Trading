import os, logging, traceback, weakref
from collections import OrderedDict  
from datetime import datetime, timedelta
import tables as tb
import numpy as np
import pandas as pd

logger = logging.getLogger('Models')
logger.setLevel(logging.DEBUG)
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data')

class OHLC(tb.IsDescription):
    """
    OHLC data model
    Table structure:
        - date : POSIX 시간(초)을 Integer 형태로 저장
        - open : 시가
        - high: 고가
        - low: 저가
        - close: 종가
        - volume: 거래량
    """
    date = tb.Int64Col(pos=0)
    open = tb.Float64Col(pos=1)
    high = tb.Float64Col(pos=2)
    low = tb.Float64Col(pos=3)
    close = tb.Float64Col(pos=4)
    volume = tb.UInt64Col(pos=5)

class Minute(tb.IsDescription):
    """
    Minute data model
    Table structure:
        - date : POSIX 시간(초)을 Integer 형태로 저장
        - high: 고가
        - low: 저가
        - volume: 거래량
    """
    date = tb.Int64Col(pos=0)
    high = tb.Float64Col(pos=1)
    low = tb.Float64Col(pos=2)
    volume = tb.UInt64Col(pos=3)


class Products(dict):
    """
    전체 상품정보를 보관하는 dictionary 클래스
    """
    # DB FILES PATH
    RAWOHLCFILE = os.path.join(BASE_DIR, "rawohlc.db") #종목별 일봉데이터
    RAWMINUTEFILE = os.path.join(BASE_DIR, "rawminute.db") #종목별 분데이터
    OHLCFILE = os.path.join(BASE_DIR, "futures.db") #상품별 연결선물 OHLC 데이터
    MINUTEFILE = os.path.join(BASE_DIR, "minute.db") #상품별 연결분데이터

    # 일봉 데이터 수집 제외 상품목록
    EXCEPTIONS = [
        'E7', 'FDXM', 'HMCE', 'HMH', 'J7', 'M6A', 'M6B', 'M6E','MCD',
        'MGC', 'MJY', 'QC', 'QG', 'QM', 'SNS', 'XC', 'XK', 'XW', 'YG',
        'YI', 'FOAM', 'MP', 'SKU', 'HCHH', 'SCH', 'SNU','SUC'
    ]
    # 종목별 시장구분
    MARKETS = {
        'CURRENCY': ('AD','BP','BR','CD','CUS','DX','E7','J7','JY','M6A','M6B','M6E','MCD','MJY','MP','NE',
                     'RY','SF','SKU','URO','SIU','SUC'),
        'METAL': ('GC','HG','MGC','PA','PL','QC','QG','QM','RB','SI','YG','YI'),
        'MEATS':  ('FC','LC','LH', ),
        'PETROLEUM': ('HO','NG','OJ','CL'),
        'TROPICAL': ('CC','KC','SB'),
        'RATE_US': ('ZB','ZN','ZT','ZF'),
        'RATE_EU': ('ED','FBTP','FDAX','FDXM','FGBL','FGBM','FOAM','FOAT','FGBS'),
        'EQUITY_CHINA': ('HCEI','HCHH','HMCE','HMH','HSI','SCH','SCN'),
        'EQUITY_JAPAN': ('NIY','NKD','SNS','SNU','SSI'),
        'EQUITY_US': ('EMD','ES','NQ','RTY','YM'),
        'EQUITY_EU': ('FESX', ),
        'FIBER': ('CT',),
        'GRAIN': ('ZW','ZC','XC','XW','ZL','ZM','ZO','ZR','ZS','XK'),
        'ETC': ('FVS','SIN','SSG','STW','VX')
    }
    #E-mini 구분
    Emini = {
        'NG': ('QG'),
        'HG': ('QC'),
        'CL': ('QM'),
        'JY': ('J7','MJY'),
        'AD': ('M6A'),
        'GBP': ('M6B'),
        'EURO': ('M6E'),
        'CD':('MCD'),
        'GC':('MGC','YG'),
        'ZC': ('XC'),
        'ZK':('XK'),
        'ZW':('XW'),
        'SI':('YI'),
        'FDAX': ('FDXM'),
        'HCEI': ('HMCE'),
        'HSI': ('HMH'),
    }


    def __setitem__(self, key, value):
        if hasattr(self, key) and self.get(key) != value:
            logger.warning(f"The product {value} has changed from {self.get(key)} to {value}")
        super(Products, self).__setitem__(key, value)
        
    def get_name(self, symbol):
        #종목명을 반환
        return self.get_contract(symbol).name
    
    def symbols(self, stype):
        #전체 종목코드리스트를 반환
        if stype == 'all':
            return [symbol for product in self.values() for symbol in product.tradables]
        #DB 저장용 종목 코드리스트를 반환
        elif stype == 'db':
            symbols = []
            for name, product in self.items():
                if name not in Products.EXCEPTIONS:
                    symbols += product.tradables
            return symbols

    def contracts(self):
        #전체 종목 인스턴스를 반환하는 제너레이터
        for product in self.values():
            for contract in product.values():
                yield contract

    def get_contract(self, symbol):
        #해당 symbol의 contract를 반환
        c = [x for p in self.values() for x in p.values() if x.symbol == symbol]
        if len(c)>1:
            raise Exception(f"A symbol can not have multiple contracts. [{symbol}]")
        else:
            return c[0] if c else None

    def create_ohlc(self):
        """ 
        raw ohlc 로부터 연결선물 데이터 생성
        파일명: Products.OHLCFILE
        """
        def compensate_db(ohlc1, ohlc2, date, tbl, decimal_len):
            """ 
            db의 가격보정 함수
            최근 3거래일의 평균가격만큼 보정해줌
            ohlc1: 액티브 월물 데이터
            ohlc2: 차월물 데이터
            date: 보정날짜
            tbl: table instance
            decimal_len: 상품의 소숫점 자릿수
            """
            d = date
            prices1=[]
            prices2=[]
            cnt = 0
            while cnt < 2:
                data1 = ohlc1[ohlc1['date']==d]
                data2 = ohlc2[ohlc2['date']==d]
                if data1.size != 0 and data2.size != 0:
                    prices1.append(data1[0].tolist()[slice(1,5)])
                    prices2.append(data2[0].tolist()[slice(1,5)])
                    cnt += 1
                d -= 1
            diff = np.round(np.average(prices2) - np.average(prices1), decimal_len)
            tbl.cols.open[:] += diff
            tbl.cols.low[:] += diff
            tbl.cols.high[:] += diff
            tbl.cols.close[:] += diff
            #print(f"{tbl.title}[{tbl.name}] Data has been changed up by {diff} at {np.int32(date).astype('M8[D]')}")
            logger.info(f"{tbl.title}[{tbl.name}] Data has been changed up by {diff} at {np.int32(date).astype('M8[D]')}")

        filters = tb.Filters(complib='blosc', complevel=9)
        db = tb.open_file(Products.OHLCFILE, mode='a', filers=filters)
        
        for product in self.values():
            if product.is_exception:
                #print(f"The product {product.name} is excepted for ohlc data")
                logger.info(f"The product {product.name} is excepted for ohlc data")
                continue
        
            #print(f"Updating Continuous Futures Data on {product.name}[{product.symbol}]")
            logger.info(f"Updating Continuous Futures Data on {product.name}[{product.symbol}]")

            
            # table 없을시 생성, attrs에 active 월물의 symbol 저장
            if not hasattr(db.root, product.symbol):
                tbl = db.create_table('/', product.symbol, OHLC, product.name)
                tbl.cols.date.create_csindex()
                active = next(iter(product.values()))
                tbl.attrs.active = active.symbol
                start = min(active.ohlc()['date'])
            else:
                tbl = db.root[product.symbol]
                active = product[tbl.attrs.active]
                start = max(tbl.cols.date) + 1 #마지막 데이터 다음날부터
            end = np.datetime64(product.lastupdate).astype('M8[D]').astype('int32')
            
            contracts = iter(product.values()) # 월물 iterator
            while next(contracts).symbol != tbl.attrs.active: pass
            following = next(contracts, None) #차월물
        
            ohlc1 = active.ohlc() #액티브월물 데이터
            ohlc2 = following.ohlc() if following else None #차월물 데이터
            date = start
        
            while date <= end:
                # 해달날짜 데이터가 이미 있으면 에러
                mdate = date
                if tbl.read_where('date>=mdate').size:
                    raise ValueError(f"{product.name} has a duplicated data at {np.int32(date).astype('M8[D]')}")
            
            
                # 해당날짜가 액티브월물의 최종거래일 이후면 액티브 월물 변경후 다시 진행
                if date >= np.datetime64(active.final_tradeday).astype('M8[D]').astype('int32'):
                    compensate_db(ohlc1, ohlc2, date, tbl, product.decimal_len) #가격보정
                    active = following
                    tbl.attrs.active = active.symbol #db내의 액티브 월물 변경
                    following = next(contracts, None)
                    ohlc1 = active.ohlc()
                    ohlc2 = following.ohlc() if following else None
                    continue
            
                #2. 해당날짜의 데이터가 없으면 넘어감
                if ohlc1[ohlc1['date']==date].size == 0:
                    date += 1
                    continue

                #3. 차월물 데이터가 없거나 이전데이터 갯수가 1개 이하면 액티브 월물 저장
                if not following\
                or ohlc1[ohlc1['date'] < date].size == 0\
                or ohlc2[ohlc2['date']==date].size == 0\
                or ohlc2[ohlc2['date'] < date].size == 0:
                    tbl.append(ohlc1[ohlc1['date']==date])
                    date += 1
                    continue

                datum1 = ohlc1[ohlc1['date']==date] #액티브월물 당일 데이터
                datum2 = ohlc2[ohlc2['date']==date] #차월물 당일 데이터
            
                #4. 차월물의 거래량이 2일연속 많으면 액티브 월물 변경
                if datum1['volume'][0] < datum2['volume'][0]\
                   and ohlc1[ohlc1['date']<date]['volume'][-1] < ohlc2[ohlc2['date']<date]['volume'][-1]:
                    compensate_db(ohlc1, ohlc2, date, tbl, product.decimal_len)
                    tbl.append(datum2)
                    active = following
                    tbl.attrs.active = active.symbol
                    following = next(contracts, None)
                    ohlc1 = active.ohlc()
                    ohlc2 = following.ohlc() if following else None

                #5. 그 외에는 액티브 월물 저장
                else:
                    tbl.append(datum1)
                date += 1
            db.flush()
        db.close()



    @staticmethod
    def get_market(symbol):
        """ 상품이 속한 시장을 반환 """
        return [mk for mk, symbols in Products.MARKETS.items() if symbol in symbols][0]

class Product(OrderedDict):
    """
    해외선물 상품 정보
    dictionary class의  subclass 로써 각 월물 정보를 dict 형태로 저장
    """
    def __init__(self):
        self.tradables = []
        

    #ordereddict 를 base class로 작성하시 init variable의 pickling이 안되서
    #아래와 같이 reduce method를 반드시 override해야함
    #def __reduce__(self):
    #    state = super().__reduce__()
    #    newstate = (state[0],
    #                ([], ),
    #                self.__dict__,
    #                None,
    #                state[4])
    #    return newstate

    def updateinfo(self, info):
        self.name = info['pname']
        self.symbol = info['psymbol']
        self.currency = info['currency']
        self.excsymbol = info['excsymbol']
        self.exchange = info['exchange']
        self.market = Products.get_market(self.symbol)#info['market']
        self.notation = info['notation']
        self.tickunit = float(info['unit'])
        self.tickprice = float(info['price_per_unit'])
        self.rgltfactor = info['rgltfactor']
        self.opentime = info['opentime']
        self.closetime = info['closetime']
        self.decimal_len = int(info['decimal_len'])
        self.i_margin = info['initial_margin']
        self.m_margin = info['mntnc_margin']
        self.is_tradable = info['is_tradable']
        mini = [c for c, symbols in Products.Emini.items() if self.symbol in symbols]
        if mini:
            self.emini = True
            self.parent = mini[0]
        else:
            self.emini = False
            self.parent = None
        self.is_exception = True if self.symbol in Products.EXCEPTIONS else False
        self.lastupdate = datetime.now()

    def info(self):
        return {
            'name': self.name,
            'symbol': self.symbol,
            'currency': self.currency,
            'exchange symbol': self.excsymbol,
            'exchange': self.exchange,
            'market': self.market,
            'notation': self.notation,
            'tickunit': self.tickunit,
            'tickprice': self.tickprice,
            'regulator factor': self.rgltfactor,
            'opening time': self.opentime.strftime('%H:%M:%S'),
            'closing time': self.closetime.strftime('%H:%M:%S'),
            'decimal length': self.decimal_len,
            'initial margin': self.i_margin,
            'maintanance margin': self.m_margin,
            'is tratable': self.is_tradable,
            'last update': self.lastupdate.strftime('%Y-%m-%d %H:%M:%S'),
            'contracts': self.tradables,
            'emini': self.emini,
            'parent': self.parent,
            'is exception': self.is_exception
        }
    
    def ohlc(self):
        """ 연결선물 데이터 리턴"""
        db =  tb.open_file(Products.OHLCFILE, mode='r')
        data = db.root[self.symbol].read_sorted('date')
        db.close()
        return data

    def dataframe(self):
        """ pandas dataframe 형식으로 연결선물데이터 리턴"""
        data = self.ohlc()
        return pd.DataFrame(data[['open','high','low','close','volume']], index=data['date'].astype('M8[D]'))
        
        


    def __str__(self):
        return(f'{self.name}[{self.symbol}]')

    def __setattr__(self, name, value):
        if name in ['tradables', 'lastupdate']:
            pass
        elif not hasattr(self, name):
            logger.info(f"{value} has assigned to {name}.")
        elif getattr(self, name) != value:
            logger.info(f"{name} of {self.name} has changed from {getattr(self, name)} to {value}.")
        super(Product, self).__setattr__(name, value)
    

class Contract:
    """ 
    해외선물 월물별 세부정보
    """
    def __init__(self, symbol, name):
        
        # 신규 ohlc db table 생성
        filters = tb.Filters(complib='blosc', complevel=9)
        db = tb.open_file(Products.RAWOHLCFILE, mode='a', filters=filters)
        tbl = db.create_table('/', symbol, OHLC, name)
        tbl.cols.date.create_csindex()
        db.close()

        # 신규 minute db table 생성
        db = tb.open_file(Products.RAWMINUTEFILE, mode='a', filters=filters)
        tbl = db.create_table('/', symbol, Minute, name)
        tbl.cols.date.create_csindex()
        db.close()

        #logger.info(f"New contract {name}[{symbol}] has created")


    def __setattr__(self, name, value):
        if name == 'lastupdate':
            pass
        elif not hasattr(self, name):
            logger.info(f"{value} has assigned to {name}.")
        elif getattr(self, name) != value:
            logger.info(f"{name} has changed from {getattr(self, name)} to {value}.")
        super(Contract, self).__setattr__(name, value)


    def updateinfo(self, info):
        self.name = info['name']
        self.symbol = info['symbol']
        self.ecprice = info['ecprice'] #정산가
        self.appldate = info['appldate'] #종목배치수신일
        self.eccode = info['eccd'] #정산구분
        self.eminicode = info['eminicode'] #Emini구분
        self.year = info['year'] #상장년
        self.month = info['month'] #상장월
        self.seqno = info['seqno'] #월물순서
        self.expiration = info['expiration'] #만기일자
        self.final_tradeday = info['final_tradeday'] #최종거래일
        self.opendate = info['opendate'] #거래시작일자(한국)
        self.closedate = info['closedate'] #거래종료일자(한국)
        self.ovsstrday = info['ovsstrday'] #거래시작일 (현지)
        self.ovsendday = info['ovsendday'] #거래종료일(현지)
        self.is_tradable = info['is_tradable'] #거래가능구분코드
        self.lastupdate = datetime.now()

    def info(self):
        return {
            'name': self.name,
            'symbol': self.symbol,
            'ecprice': self.ecprice,
            'appl date': self.appldate.strftime('%Y-%m-%d'),
            'eccode': self.eccode,
            'eminicode': self.eminicode,
            'year': self.year,
            'month': self.month,
            'seqno': self.seqno,
            'expiration': self.expiration.strftime('%Y-%m-%d'),
            'final tradeday': self.final_tradeday.strftime('%Y-%m-%d'),
            'open date': self.opendate.strftime('%Y-%m-%d %H:%M:%S'),
            'close date': self.closedate.strftime('%Y-%m-%d %H:%M:%S'),
            'ovs start day': self.ovsstrday.strftime('%Y-%m-%d %H:%M:%S'),
            'ovs end day': self.ovsendday.strftime('%Y-%m-%d %H:%M:%S'),
            'is tradable': self.is_tradable,
            'lastupdate': self.lastupdate.strftime('%Y-%m-%d %H:%M:%S'),
        }

    def update_ohlc(self, data):
        #종목별 일봉 OHLC 데이터 업데이트
        if not data: 
            return logger.info("Nothing to update")
            
        db = tb.open_file(Products.RAWOHLCFILE, mode='a')
        table = db.root[self.symbol]
        for datum in data:
            sdate, open, high, low, close, volume = datum
            ohlc = list(map(float, [open,high,low,close]))
            #날짜가 이상하게 들어올때가 있음
            try: 
                ndate = np.datetime64(datetime.strptime(sdate, '%Y%m%d')).astype('M8[D]').astype('int32')
            except:
                logger.warning(f"{self.name} has a wrong DATE! at {sdate}")
                logger.error(traceback.format_exc())
                continue
            
            #중복 데이터 버림
            if table.read_where('date>=ndate').size:
                logger.info(f"{self.name} has a duplicated data at {sdate}")
            
            #거래량이 1미만이면 버림
            elif int(volume) < 0:
                logger.warning(f"{self.name} has a data with volume: {volume} at {sdate}")

            #잘못된 데이터 버림
            elif max(ohlc) != float(high) or min(ohlc) != float(low):
                logger.warning(f"{self.name} has a wrong data {datum}")

            #이제 업데이트 진행
            else:
                table.append([(ndate, open, high, low, close, volume)])
        table.flush()
        db.close()

    def ohlc(self, period='minute'):
        """pandas DataFrame 형식의 ohlc data 반환
           period : '30min' or 'day'
        """ 
        db = tb.open_file(Products.RAWMINUTEFILE, mode='r')
        data = db.root[self.symbol].read_sorted(sortby='date')
        db.close()
        df = pd.DataFrame(data[['open','high','low','close','volume']], index=data['date'].astype('M8[s]'))
        if period == 'minute':
            return df
        elif period == 'day':
            df= df.groupby(pd.Grouper(freq='24H',closed='left', label='left', base=8)).\
                   agg({'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume': 'sum'}).dropna()
            df.index = df.index.values.astype('M8[ns]').astype('M8[D]')
            return df
    def update_minute(self, data):
        """ 분데이터 업데이트 """
        if not data: 
            return logger.info("Nothing to update")
            
        lastdate = self.lastdate_in_db('Minute')
        db = tb.open_file(Products.RAWMINUTEFILE, mode='a')
        table = db.root[self.symbol]
        for datum in data:
            ndate, open, high, low, close, volume = datum
            ohlc = list(map(float, [open,high,low,close]))
            sdate = ndate.astype('M8[s]')
            
            #마지막 데이터보다 앞선 데이터 버림
            if ndate <= lastdate: #table.read_where('date>=ndate').size:
                continue

            #중복 데이터 버림
            elif table.read_where('date==ndate').size:
                continue
            
            #거래량이 1미만이면 버림
            elif int(volume) < 1:
                logger.warning(f"{self.name} has a data with volume: {volume} at {sdate}")

            #잘못된 데이터 버림
            elif max(ohlc) != float(high) or min(ohlc) != float(low):
                logger.warning(f"{self.name} has a wrong data {datum}")
            
            
            #이제 업데이트 진행
            else:
                table.append([(ndate, open, high, low, close, volume)])
                table.flush()
        db.close()
    
    def startday(self):
        # db에 저장된 마지막 날짜의 다음날짜를 YYYYMMdd 형식으로 반환
        # 마지막 날짜가 0이면 일주일 전 날짜 반환
        lastdate = self.lastdate_in_db('OHLC')
        startdate = lastdate.astype('M8[D]') + np.timedelta64(1,'D')
        return ''.join(str(startdate).split('-')) 
    
    def lastdate_in_db(self, dbname):
        # db에 저장된 마지막 날짜의 posix time을 반환
        if dbname == 'OHLC':
            dbfile = Products.RAWOHLCFILE
            default = np.datetime64(self.ovsendday - timedelta(7)).astype('M8[D]').astype('int32')
        elif dbname == 'Minute':
            dbfile = Products.RAWMINUTEFILE
            default = np.datetime64(self.ovsendday - timedelta(7)).astype('M8[s]').astype('int64')
        
        db = tb.open_file(dbfile, mode='r')
        lastdate = max(db.root[self.symbol].cols.date, default=default)
        db.close()
        return lastdate

    def __str__(self):
        return self.name


