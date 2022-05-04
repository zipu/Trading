"""
시장의 일별 데이터를 담고 있는 오브젝트
기본적으로 ohlc 값을 가지고 있으며
다른 지표들 ex) ma, ema 등을 반환하거나 기존 테이블에 추가할 수 있다.


"""

import pandas as pd
import numpy as np

from .datatools import norm

class Quotes(pd.DataFrame):
    """
     일별 상품 정보 (OHLC 등) 클래스로,
     pandas DataFrame의 하위 클래스
    """
    def __init__(self, data, type='multiple'):
        """
        type: single or multi
        """
        super().__init__(data) 
        self.type = type

        #if 'open_interest' in self.columns:
        #    self.rename(columns = {'open_interest':'oi'}, inplace = True)
        #
        #if type == 'single':
        #    # 날짜가 컬럼에 있는경우 인덱스로 변경
        #    if 'date' in self.columns:
        #        self['date'] = self['date'].astype('M8[D]')
        #        self.set_index('date', inplace=True)
        #    self.columns.names = (['field'])
        #if type == 'multiple':
        #    self.columns.names = (['symbol', 'field'])


    def MA(self, window, ref='close', inplace=False, fieldname=None):
        """
        N-day Moving Average
        window: 기준일 (window)
        ref : 기준 가격 (open, high, low, close)
        inplace: True 시 기존 quote에 병합하여 리턴 
        fieldname: pandas dataframe column name
        """
        fieldname = fieldname if fieldname else f'ma{window}'
        
        
        if self.type == 'single':
            ma = self[ref].rolling(window, min_periods=1).mean()
            ma.name = fieldname
            
            if inplace:
                self[fieldname] = ma
            else: 
                return ma

        if self.type == 'multiple':
            ma = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].rolling(window, min_periods=1).mean()
                df.name = (symbol, fieldname)
                ma.append(df)
            
            ma = pd.concat(ma, axis=1)
            if inplace:
                self[ma.columns] = ma
                self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
            else:
                return ma

    
    def EMA(self, window, ref='close', inplace=False, fieldname=None):
        """
        N-day Exponential Moving Average
        window: 기준일 (window)
        ref : 기준 가격 (open, high, low, close)
        inplace: True 시 기존 quote에 병합하여 리턴 
        fieldname: dataframe column name
        """
        fieldname = fieldname if fieldname else f'ema{window}'
        
        if self.type == 'single':
            ema = self[ref].ewm(span=window).mean()
            ema.name = fieldname
            if inplace:
                self[fieldname] = ema
            else: 
                return ema

        if self.type == 'multiple':
            ema = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].ewm(span=window).mean()
                df.name = (symbol, fieldname)
                ema.append(df)
            
            ema = pd.concat(ema, axis=1)
            if inplace:
                self[ema.columns] = ema
                self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

            else:
                return ema

    
    def ATR(self, window=20, inplace=False, fieldname=None):
        """
        Average True Range (변동성 지표)
        fieldname: pandas dataframe column name
        """
        fieldname = fieldname if fieldname else f'atr{window}'

        if self.type == 'single':
            df = pd.DataFrame()
            df['hl'] = self['high'] - self['low']
            df['hc'] = np.abs(self['high'] - self['close'].shift(1))
            df['lc'] = np.abs(self['low'] - self['close'].shift(1))
            #df['TR'] = df.max(axis=1)
            atr = df.max(axis=1).ewm(span=window).mean()
            
            #normalization (z-score) - 지난 1년 데이터 기준
            atr = (atr - atr.rolling(window=250, min_periods= 1).mean())/atr.rolling(window=250, min_periods= 1).std()
            #atr = norm(atr)
            atr.name = fieldname
            if inplace:
                self[fieldname] = atr
            else:
                return atr

        
        elif self.type == 'multiple':
            atrs=[]
            for symbol, quote in self.groupby(level=0, axis=1):
                df = pd.DataFrame()
                df['hl'] = quote[symbol, 'high'] - quote[symbol, 'low']
                df['hc'] = np.abs(quote[symbol, 'high'] - quote[symbol, 'close'].shift(1))
                df['lc'] = np.abs(quote[symbol, 'low'] - quote[symbol, 'close'].shift(1))
                atr = df.max(axis=1).ewm(span=window).mean()
                atr = (atr - atr.rolling(window=200, min_periods= 1).mean())/atr.rolling(window=200, min_periods= 1).std()
                atr.name = (symbol, fieldname)
                atrs.append(atr)
            atrs = pd.concat(atrs, axis=1)

            if inplace:
                self[atrs.columns] = atrs
                self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

            else:
                return atrs


    def MIN(self, window, ref='low', inplace=False, fieldname=None):
        """
        최근 N일 저가
        """
        fieldname = fieldname if fieldname else f'min{window}'

        if self.type == 'single':
            ind = self[ref].rolling(window, min_periods=1).min()
            ind.name = fieldname
            if inplace:
                self[fieldname] = ind
            else: 
                return ind

        if self.type == 'multiple':
            ind = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].rolling(window, min_periods=1).min()
                df.name = (symbol, fieldname)
                ind.append(df)
            
            ind = pd.concat(ind, axis=1)
            if inplace:
                self[ind.columns] = ind
                self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

            else:
                return ind

    def MAX(self, window, ref='high', inplace=False, fieldname=None):
        """
        최근 N일 고가
        return: data, type
        """
        fieldname = fieldname if fieldname else f'max{window}'

        if self.type == 'single':
            ind = self[ref].rolling(window, min_periods=1).max()
            ind.name = fieldname
            if inplace:
                self[fieldname] = ind
            else: 
                return ind

        if self.type == 'multiple':
            ind = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].rolling(window, min_periods=1).max()
                df.name = (symbol, fieldname)
                ind.append(df)
            
            ind = pd.concat(ind, axis=1)
            if inplace:
                self[ind.columns] = ind
                self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

            else:
                return ind



        


