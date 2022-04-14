"""
상품(Instrument) 의 Meta data 호출 및 관리

- data/instruments.csv 파일에 저장된 상품정보를 불러와 오브젝트화 함

"""
import csv
import os
from decimal import Decimal as D

import h5py
import pandas as pd

from .constants import DATADIR


class Instrument:
    """
    상품 인스턴스
    """
    def __init__(self, instrument):
        self._symbol = instrument['symbol']
        self._name = instrument['name']
        self._ebest = instrument['ebest'] #이베스트 심볼
        self._kibot = instrument['kibot'] #kibot 심볼
        self._exchange = instrument['exchange']
        self._months = instrument['months'] #거래가능 월물
        self._tickunit = D(instrument['tickunit']) if instrument['tickunit'] else D('0')
        self._tickvalue = D(instrument['tickvalue'])  if instrument['tickvalue'] else D('0')
        self._margin = D(instrument['margin'])  if instrument['margin'] else D('0')
        self._currency = instrument['currency']
        self._tradable = True if instrument['tradable'] == '1' else False 
        self._number_system = int(instrument['number_system'] or 0)
        self._sector = instrument['sector']
        self._info = {
            'symbol': self._symbol,
            'name': self._name,
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

    def __repr__(self):
        return f"[{self.symbol}] {self.name}"

    def quotes(self, db='kibot', format='numpy'):
        """
        Database 에 저장된 일봉데이터 반환
        kwargs
         db: database 종류 (default: kibot)
        """
        filepath = os.path.join(DATADIR,'kibot','quotes.hdf')
        file = h5py.File(filepath, 'r')
        data = file[self.symbol][:]
        file.close()

        if format == 'numpy':
            return data
        if format == 'pandas':
            df = pd.DataFrame(data)
            df['date'] = df['date'].astype('M8[D]')
            df.set_index('date', inplace=True)
            return df


    
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
    _filepath = os.path.join(DATADIR, 'instruments.csv')
    
    def __init__(self):
        #self.dict = dict()
        with open(Instruments._filepath, 'r') as file:
            for line in csv.DictReader(file):
                self[line['symbol']] = Instrument(line)

    def __repr__(self):
        return "종목 정보 오브젝트"
    
    def get(self, symbol):
        """ 
        종목코드에 해당하는 종목 정보를 반환
        symbol(string) : 종목코드
        """
        return self[symbol]
    
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

    def filter(self, **kwargs):
        """
        전체 상품 목록중 kwargs로 들어온 key,value 들과 매칭된 상품목록 리턴
        argument에 ebest(boolean) 또는 kibot(boolean)이 있으면, 
        ebest 코드 (또는 kibot 코드)가 있는 모든 상품리스트 반환
        ex) filter(symbol='AD', tradable=True) 
        """
        lists = []
        instruments = self.values()
        if 'ebest' in kwargs.keys():
            kwargs.pop('ebest')
            instruments = filter(lambda i: i.ebest, instruments)
        
        if 'kibot' in kwargs.keys():
            kwargs.pop('kibot')
            instruments = filter(lambda i: i.kibot, instruments)

        for instrument in instruments:
            if all(instrument.info[k] == v for k,v in kwargs.items()):
                lists.append(instrument)
        
        #print(f'Total {len(lists)} items selected')
        return tuple(lists)

    def quotes(self, symbols=None, start=None, end=None):
        """
        여러 상품의 일봉정보를 돌려주는 함수
        *args
          db: 데이터 소스
          symbols: 상품 리스트
          start: 시작일자 (없으면 전체 데이터)
          end: 끝일자 
        
        *return
          pandas dataframe
        """
        if symbols is None:
            symbols = self.filter(kibot=True)
        

        #pandas column name
        indexes = [
            sum([ [instrument.symbol]*6 for instrument in symbols],[]),
            ['open','high','low','close','volume','open_interest']*len(symbols)
        ]

        df = pd.concat([i.quotes(format='pandas') for i in symbols], axis=1)
        df.columns = indexes
        return df[start:end]






        



            
                

"""
상품 정보 인스턴스를 생성하여 필요시 호출 후 바로 사용 가능
"""
instruments = Instruments()