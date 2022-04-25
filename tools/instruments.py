"""
상품(Instrument) 의 Meta data 호출 및 관리

- data/instruments.csv 파일에 저장된 상품정보를 불러와 오브젝트화 함

"""
import csv
import os
from decimal import Decimal as D
from datetime import datetime

import h5py
import pandas as pd

from .constants import DATADIR,\
                       INSTRUMENTS_CSV_PATH,\
                       KIBOT_FUTURES_CONTINUOUS_BV_DB_PATH, SRF_CONTINUOUS_BO_DB_PATH, SRF_CONTINUOUS_BV_DB_PATH,\
                       SRF_CONTRACTS_CSV_PATH, SRF_CONTRACTS_DB_PATH,\
                       SRF_ROLLOVER_BO_CSV_PATH, SRF_ROLLOVER_BV_CSV_PATH


class Instrument:
    """
    상품 인스턴스
    """
    def __init__(self, instrument):
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

        self.contracts = self._contracts_list()


    def __repr__(self):
        return f"[{self.symbol}] {self.name}"

    def quotes(self, db='srf', method='bo', format='pandas', fields='ohlcvi', contract=None):
        """
        Database 에 저장된 일봉데이터 반환
        kwargs
         db: database 종류 (default: kibot)
         format: 'numpy' or 'pandas'
         method: 연결방식 (bo 또는 bv)
         fields: 반환할 필드값 'ohlcv', 'ohlc', 'ohlcvi'
        """
        #filename = "futures-continuous-BV.hdf"
        #filepath = os.path.join(DATADIR, db, filename)
        if db == 'kibot':
            file = h5py.File(KIBOT_FUTURES_CONTINUOUS_BV_DB_PATH, 'r')
            dset = file
            code = self.symbol
        
        elif db == 'srf' and not contract:
            if method == 'bo':
                file = h5py.File(SRF_CONTINUOUS_BO_DB_PATH, 'r')
            elif method == 'bv':
                file = h5py.File(SRF_CONTINUOUS_BV_DB_PATH, 'r')
            dset = file
            code = self.symbol 

        elif db == 'srf' and contract:
            file = h5py.File(SRF_CONTRACTS_DB_PATH, 'r')
            dset = file[self.symbol]
            code = contract
        
        if not dset.get(code):
            """ 해당 상품 데이터가 없음 """
            return None

        
        
        if fields == 'ohlcvi':
            data = dset[code][:]
        elif fields == 'ohlcv':
            data = dset[code]['date','open','high','low','close','volume']
        elif fields == 'ohlc':
            data = dset[code]['date','open','high','low','close'] 
        
        file.close()

        if format == 'numpy':
            return data
        if format == 'pandas':
            df = pd.DataFrame(data)
            df['date'] = df['date'].astype('M8[D]')
            df.set_index('date', inplace=True)
            return df

    def rolldates(self, method=None):
        """
        data/srf/SRF_rollover_bv.csv 로부터 월물 롤오버 정보를 취합하여 리턴
        Args:
         method: 'bo' (backward & open interest)
                 'bv' (backward & volume)
        """
        if method == 'bo':
            filepath = SRF_ROLLOVER_BO_CSV_PATH
        elif method == 'bv':
            filepath = SRF_ROLLOVER_BV_CSV_PATH
        elif method == None:
            raise ValueError("roll over metod must be given")

        rolls = []
        with open(filepath, mode='r') as f:
            lines = csv.reader(f)
            for line in lines:
                if line[0] == self.symbol:
                    rolls.append(line)
        
        return rolls




    def _contracts_list(self):
        """
        월물 정보를 리스트로 반환
        return:
            [symbol, srf, code, from_date, to_date, refreshed_at, name]
        """
        contracts = []
        with open(SRF_CONTRACTS_CSV_PATH, encoding='utf-8', mode='r') as f:
            lines = csv.DictReader(f)
            for line in lines:
                if line['symbol'] == self.symbol:
                    from_date = datetime.strptime(line['from_date'], '%Y-%m-%d')
                    to_date = datetime.strptime(line['to_date'], '%Y-%m-%d')
                    contract = (line['symbol'], line['srf'], line['contract'], from_date,\
                                to_date, line['refreshed_at'], line['name'])
                    contracts.append(contract)
        return contracts




    
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
    #_filepath = os.path.join(DATADIR, 'instruments.csv')
    _filepath = INSTRUMENTS_CSV_PATH

    def __init__(self):
        #self.dict = dict()
        with open(Instruments._filepath, 'r') as file:
        #with open(INSTRUMENTS_CSV_PATH, 'r') as file:
            for line in csv.DictReader(file):
                self[line['symbol']] = Instrument(line)

    def __repr__(self):
        return "종목 정보 오브젝트"
    
    # deprecated - dict의 하위 클래스여서 기본적으로 장착하고 있음
    #def get(self, symbol):
    #    """ 
    #    종목코드에 해당하는 종목 정보를 반환
    #    symbol(string) : 종목코드
    #    """
    #    return self[symbol]
    
    def getlist(self, field):
        """
        field(string) 값을 리스트로 반환
        """
        lists = []
        for instrument in self.values():
            attr = getattr(instrument,field)
            if attr:
                lists.append(attr)
                
        return tuple(lists)

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
            #kwargs.pop('ebest')
            instruments = filter(lambda i: i.ebest, instruments)
        
        if kibotall:
            #kwargs.pop('kibot')
            instruments = filter(lambda i: i.kibot, instruments)

        if srfall:
            instruments = filter(lambda i: i.srf, instruments)

        for instrument in instruments:
            if all(instrument.info[k] == v for k,v in kwargs.items()):
                lists.append(instrument)
        
        #print(f'Total {len(lists)} items selected')
        #return tuple(lists)
        return lists 

    def quotes(self, db='srf', symbols=None, start=None, end=None, fields='ohlcvi', method='bo'):
        """
        여러 상품의 일봉정보를 돌려주는 함수
        *args
          db: 'srf' or 'kibot'
          symbols: 상품 리스트 (없으면 전체 데이터)
          start: 시작일자 (없으면 전체 데이터)
          end: 끝일자 
          fields: 반환할 데이터의 필드값 'ohlcvi', 'ohlcv', 'ohlc'
          method: 'bo' or 'bv'
        
        *return
          pandas dataframe
        """
        if db == 'kibot':
            filename = KIBOT_FUTURES_CONTINUOUS_BV_DB_PATH
        
        elif db == 'srf' and method=='bo':
            filename = SRF_CONTINUOUS_BO_DB_PATH
        
        elif db == 'srf' and method=='bv':
            filename = SRF_CONTINUOUS_BV_DB_PATH

        else:
            raise ValueError(f"'{method}' is not known method")

        file = h5py.File(filename, 'r')


        if not symbols:
            symbols = list(file.keys())
        

        #pandas column name
        # ex) 상품이 AD, EC 입력받은 fields 값이 'ohlc' 이면
        #  indexes = [
        #              ['AD','AD','AD','AD','EC','EC','EC','EC],
        #              ['open','high','low','close','open','high','low','close]
        #            ]
        fieldnames = ['open','high','low','close','volume','oi']
        fieldnames = fieldnames[:len(fields)]
        #indexes = [
        #    sum([ [instrument.symbol]*len(fieldnames) for instrument in symbols],[]),
        #    fieldnames*len(symbols)
        #]
        indexes = [
            [],
            []
        ]

        df = pd.DataFrame()
        for symbol, quote in file.items(): 
            indexes[0] = indexes[0] + [symbol]*len(fieldnames)
            indexes[1] = indexes[1] + fieldnames
            data = pd.DataFrame(quote[:])
            data['date'] = data['date'].astype('M8[D]')
            data.set_index('date', inplace=True)

            df = pd.concat([df, data], axis=1)
        df.columns = indexes
        file.close()
        return df[start:end]

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