"""
상품(Instrument) 의 Meta data 호출 및 관리

- data/instruments.csv 파일에 저장된 상품정보를 불러와 오브젝트화 함

"""
import csv, json
import os, warnings
from decimal import Decimal as D
#from datetime import datetime

import h5py
import pandas as pd

from .constants import DATADIR,\
                       INSTRUMENTS_CSV_PATH,\
                       KIBOT_FUTURES_CONTINUOUS_BV_DB_PATH, SRF_CONTINUOUS_BO_DB_PATH,\
                       SRF_CONTINUOUS_SO_DB_PATH, SRF_CONTRACTS_JSON_PATH, SRF_CONTRACTS_DB_PATH,\
                       SRF_ROLLOVER_BO_CSV_PATH

from .quotes import Quotes


    
class Instrument:
    """
    상품 인스턴스
    """
    def __init__(self, instrument, contracts):
        self._symbol = instrument['symbol']
        self._name = instrument['name']
        self._srf = instrument['srf'] #SRF 심볼
        self._ebest = instrument['ebest'] #이베스트 심볼
        self._kibot = instrument['kibot'] #kibot 심볼
        self._exchange = instrument['exchange']
        self._months = instrument['months'] #거래가능 월물
        self._tickunit = float(instrument['tickunit'] or 0)#D(instrument['tickunit']) if instrument['tickunit'] else D('0')
        self._tickvalue = float(instrument['tickvalue'] or 0)#D(instrument['tickvalue'])  if instrument['tickvalue'] else D('0')
        self._margin = int(instrument['margin'] or 0)#D(instrument['margin'])  if instrument['margin'] else D('0')
        self._currency = instrument['currency']
        self._tradable = True if instrument['tradable'] == '1' else False 
        self._number_system = int(instrument['number_system'] or 0)
        self._sector = instrument['sector']
        self._info = {
            'symbol': self._symbol,
            'name': self._name,
            'srf': self._srf,
            'ebest': self._ebest,
            'kibot': self._kibot,
            'exchange': self._exchange,
            'months': self._months,
            'tickunit': self._tickunit,
            'tickvalue': self._tickvalue,
            'margin': self._margin,
            'currency': self._currency,
            'tradable': self._tradable,
            'number_system': self._number_system,
            'sector': self._sector,
        }

        self.contracts = contracts 


    def __repr__(self):
        return f"[{self.symbol}] {self.name}"
    
    

    def quotes(self, db=SRF_CONTINUOUS_BO_DB_PATH, fields='ohlcvi', contract=None):
        """
        Database 에 저장된 일봉데이터 반환
        kwargs
         db: database 파일
         fields: 반환할 필드값 'ohlcv', 'ohlc', 'ohlcvi'
        """
        #filename = "futures-continuous-BV.hdf"
        #filepath = os.path.join(DATADIR, db, filename)
        #if db == 'kibot':
        #    file = h5py.File(KIBOT_FUTURES_CONTINUOUS_BV_DB_PATH, 'r')
        #    dset = file
        #    code = self.symbol
        
        #elif db == 'srf' and not contract:
        #    if method == 'bo':
        #        file = h5py.File(SRF_CONTINUOUS_BO_DB_PATH, 'r')
            
            #elif method == 'bv':
            #    file = h5py.File(SRF_CONTINUOUS_BV_DB_PATH, 'r')
            
        #    elif method == 'so':
        #        file = h5py.File(SRF_CONTINUOUS_SO_DB_PATH, 'r')

        #    else:
        #        raise ValueError(f"method: {method} does not exit")
            
        #    dset = file
        #    code = self.symbol 

        if contract:
            file = h5py.File(db, 'r')
            dset = file[self.symbol]
            code = contract

        else:
            file = h5py.File(db, 'r')
            dset = file
            code = self.symbol


        if not code in dset.keys():
            return None

        if fields == 'ohlcvi':
            data = dset[code][:]
        elif fields == 'ohlcv':
            data = dset[code]['date','open','high','low','close','volume']
        elif fields == 'ohlc':
            data = dset[code]['date','open','high','low','close'] 
        
        file.close()

        df = pd.DataFrame(data)
        #if 'date' in df.columns:
        df['date'] = df['date'].astype('M8[D]')
        df.set_index('date', inplace=True)
        df.columns.names = (['field'])
        df.rename(columns = {'open_interest':'oi'}, inplace = True)
        
        return Quotes(df, type='single')
        #if format == 'numpy':
        #    return data
        #if format == 'pandas':
        #    df = pd.DataFrame(data)
        #    df['date'] = df['date'].astype('M8[D]')
        #    df.set_index('date', inplace=True)
        #    return df

    def rolldates(self, method=None):
        """
        data/srf/SRF_rollover_bv.csv 로부터 월물 롤오버 정보를 취합하여 리턴
        Args:
         method: 'bo' (backward & open interest)
                 'bv' (backward & volume)
                 'so' (simple & open interest)
        """
        if method == 'bo' or method=='so':
            filepath = SRF_ROLLOVER_BO_CSV_PATH
        #elif method == 'bv':
        #    filepath = SRF_ROLLOVER_BV_CSV_PATH
        elif method == None:
            raise ValueError("roll over metod must be given")

        rolls = []
        with open(filepath, mode='r') as f:
            lines = csv.reader(f)
            for line in lines:
                if line[0] == self.symbol:
                    rolls.append(line)
        
        return rolls

    @property
    def decimal_places(self):
        """ 소숫점 자리수 """
        exponent = D(str(self.tickunit)).as_tuple().exponent
        return exponent * -1 if exponent < 0 else 0

   
    @property
    def info(self):
        """ 전체 상품 정보를 dictionary로 반환"""
        return self._info
    
    @property
    def symbol(self):
        """ 상품코드 """
        return self._symbol

    @property
    def name(self):
        """ 상품명 """
        return self._name

    @property
    def srf(self):
        """ SRF (Reference futures) 상품코드 """
        return self._srf

    @property
    def ebest(self):
        """ 이베스트 상품코드 """
        return self._ebest
    
    @property
    def kibot(self):
        """ 키봇 상품코드 """
        return self._kibot

    @property
    def exchange(self):
        """ 거래소 """
        return self._exchange

    @property 
    def months(self):
        """ 거래 월물 """
        return self._months

    @property
    def tickunit(self):
        """ 틱 단위 """
        return self._tickunit

    @property
    def tickvalue(self):
        """ 틱당 가치 """
        return self._tickvalue

    @property
    def margin(self):
        """ 개시 증거금 """
        return self._margin

    @property
    def currency(self):
        """ 거래통화 """
        return self._currency

    @property
    def tradable(self):
        """ 거래가능 여부 """
        return self._tradable
    @property
    def number_system(self):
        """ 진법 """
        return self._number_system

    @property
    def sector(self):
        """ 섹터분류 """
        return self._sector





class Instruments(dict):
    """
    종목(instrument) 정보를 나타내는 클래스
    root/data/instruments.csv 에 저장된 내용을 오브젝트화 한다
    계속 업데이트 필요
    """
    def __init__(self):

        #종목별 월물리스트 불러오기
        #[symbol, srf, code, from_date, to_date, refreshed_at]
        if os.path.exists(SRF_CONTRACTS_JSON_PATH):
            with open(SRF_CONTRACTS_JSON_PATH, 'r') as f:
                contracts = json.load(f)
        else:
            contracts={}
            warnings.warn("SRF_CONTRACTS_JSON_PATH 파일이 존재하지 않습니다.")


        with open(INSTRUMENTS_CSV_PATH, 'r') as file:
            for line in csv.DictReader(file):
                self[line['symbol']] = Instrument(line, contracts.get(line['symbol']))

        
    
    def __repr__(self):
        return "종목 정보 오브젝트"
    
       
    
    def get_symbols(self, *args):
        """
        kwargs에 들어있는 field 값을 가지고 있는 모든 종목의 심볼 리스트를 반환
        """
        lists =[]
        for instrument in self.values():
            if all(instrument.info[arg] for arg in args):
                lists.append(instrument.symbol)
        return lists

    def filter(self,ebestall=False, kibotall=False, srfall=False, **kwargs):
        """
        전체 상품 목록중 kwargs로 들어온 key,value 들과 매칭된 상품목록 리턴
        argument에 ebest(boolean) 또는 kibot(boolean)이 있으면, 
        ebest 코드 (또는 kibot 코드)가 있는 모든 상품리스트 반환
        ex) filter(symbol='AD', tradable=True) 
        """

        lists = []
        instruments = list(self.values())
        if ebestall:
            instruments = filter(lambda i: i.ebest, instruments)
        
        if kibotall:
            instruments = filter(lambda i: i.kibot, instruments)

        if srfall:
            instruments = filter(lambda i: i.srf, instruments)

        for instrument in instruments:
            if all(instrument.info[k] == v for k,v in kwargs.items()):
                lists.append(instrument)
        
        #print(f'Total {len(lists)} items selected')
        #return tuple(lists)
        return lists 
    
    def db_symbols(self, db=SRF_CONTINUOUS_BO_DB_PATH):
        """ 사용할 DB 안의 상품 리스트 반환"""
        file = h5py.File(db, 'r')
        return list(file.keys())

    def quotes(self, db=SRF_CONTINUOUS_BO_DB_PATH, symbols=None, start=None, end=None, fields='ohlcvi'):
        """
        여러 상품의 일봉정보를 돌려주는 함수
        *args
          db: 'srf' or 'kibot'
          symbols: 상품 리스트 (없으면 전체 데이터)
          start: 시작일자 (없으면 전체 데이터)
          end: 끝일자 
          fields: 반환할 데이터의 필드값 'ohlcvi', 'ohlcv', 'ohlc'
          method: 'bo' or 'so'
        
        *return
          pandas dataframe
        """
        #if db == 'kibot':
        #    filename = KIBOT_FUTURES_CONTINUOUS_BV_DB_PATH
        
        #elif db == 'srf' and method=='bo':
        #    filename = SRF_CONTINUOUS_BO_DB_PATH
        
        #elif db == 'srf' and method=='bv':
        #    filename = SRF_CONTINUOUS_BV_DB_PATH
        
        #elif db == 'srf' and method=='so':
        #    filename = SRF_CONTINUOUS_SO_DB_PATH

        #else:
        #    raise ValueError(f"'{method}' is not known method")

        file = h5py.File(db, 'r')


        if not symbols:
            symbols = list(file.keys())


        #pandas column name
        fieldnames = ['open','high','low','close','volume','oi']
        fieldnames = fieldnames[:len(fields)]
        #pandas multi index 형식
        indexes = [
            [],
            []
        ]

        df = pd.DataFrame()
        for symbol in symbols: 
            quote = file[symbol]
            
            indexes[0] = indexes[0] + [symbol]*len(fieldnames)
            indexes[1] = indexes[1] + fieldnames
            data = pd.DataFrame(quote[:])
            if 'open_interest' in data:
                data.rename(columns = {'open_interest':'oi'}, inplace = True)


            data['date'] = data['date'].astype('M8[D]')
            data.set_index('date', inplace=True)
            data = data[fieldnames] 

            df = pd.concat([df, data], axis=1)
        df.columns = indexes
        df.columns.names = (['symbol', 'field'])
        file.close()
        return Quotes(df[start:end], type='multiple')

    def kibot_contracts_list(self):
        """
        kibot의 월물 리스트 중 거래 및 사용가능한 상품에 대해
        각 상품의 월물 목록을 시간순으로 배열하여 dictionary로 반환 
        """
        from collections import defaultdict
        
        contracts = defaultdict(list)
        path = os.path.join(DATADIR, 'kibot','contracts-list.csv')
        with open(path, 'r') as file:
            wr = csv.DictReader(file)
            for item in wr:
                contracts[item['symbol']].append(item['contract'])
        return contracts

    def sort(self, contracts):
        """
        월물 리스트를 시간순서로 배열 후 리턴
        """
        return sorted(
            contracts,
            key=lambda x: (int(x[-2:]), self.month_code(x[-3])),
        )
                

    def month_code(self, month):
        """
        선물 월물기호를 숫자(월) 로 반환
        """
        code = {
            'F': 1,
            'G': 2,
            'H': 3,
            'J': 4,
            'K': 5,
            'M': 6,
            'N': 7,
            'Q': 8,
            'U': 9,
            'V': 10,
            'X': 11,
            'Z': 12,
        }
        return code[month]
    
 
       



"""
상품 정보 인스턴스를 생성하여 필요시 호출 후 바로 사용 가능
"""
instruments = Instruments()