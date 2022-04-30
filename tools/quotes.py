"""
시장의 일별 데이터를 담고 있는 오브젝트
기본적으로 ohlc 값을 가지고 있으며
다른 지표들 ex) ma, ema 등을 반환하거나 기존 테이블에 추가할 수 있다.


"""

import pandas as pd
import numpy as np

class Quotes(pd.DataFrame):
    """
     일별 상품 정보 (OHLC 등) 클래스로,
     pandas DataFrame의 하위 클래스
    """
    def __init__(self, data, type='single'):
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


    def MA(self, window, ref='close', inplace=False):
        """
        N-day Moving Average
        window: 기준일 (window)
        ref : 기준 가격 (open, high, low, close)
        inplace: True 시 기존 quote에 병합하여 리턴 
        """
        if self.type == 'single':
            ma = self[ref].rolling(window, min_periods=1).mean()
            ma.name = f'ma{window}'
            if inplace:
                self[f'ma{window}'] = ma
            else: 
                return ma

        if self.type == 'multiple':
            ma = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].rolling(window, min_periods=1).mean()
                df.name = (symbol, f'ma{window}')
                ma.append(df)
            
            ma = pd.concat(ma, axis=1)
            if inplace:
                self[ma.columns] = ma
                self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)


            else:
                return ma

    
    def EMA(self, window, ref='close', inplace=False):
        """
        N-day Exponential Moving Average
        window: 기준일 (window)
        ref : 기준 가격 (open, high, low, close)
        inplace: True 시 기존 quote에 병합하여 리턴 
        """
        if self.type == 'single':
            ema = self[ref].ewm(span=window).mean()
            ema.name = f'ema{window}'
            if inplace:
                self[f'ema{window}'] = ema
            else: 
                return ema

        if self.type == 'multiple':
            ema = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].ewm(span=window).mean()
                df.name = (symbol, f'ema{window}')
                ema.append(df)
            
            ema = pd.concat(ema, axis=1)
            if inplace:
                self[ema.columns] = ema
                self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

            else:
                return ema

    
    def ATR(self, window=20, inplace=False):
        """
        Average True Range (변동성 지표)
        """
        if self.type == 'single':
            df = pd.DataFrame()
            df['hl'] = self['high'] - self['low']
            df['hc'] = np.abs(self['high'] - self['close'].shift(1))
            df['lc'] = np.abs(self['low'] - self['close'].shift(1))
            #df['TR'] = df.max(axis=1)
            atr = df.max(axis=1).ewm(span=window).mean()
            if inplace:
                self[f'atr{window}'] = atr
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
                atr.name = (symbol, f'atr{window}')
                atrs.append(atr)
            atrs = pd.concat(atrs, axis=1)

            if inplace:
                self[atrs.columns] = atrs
                self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

            else:
                return atrs


    def MIN(self, window, ref='close', inplace=False):
        """
        최근 N일 저가
        """
        if self.type == 'single':
            ind = self[ref].rolling(window, min_periods=1).min()
            ind.name = f'min{window}'
            if inplace:
                self[f'min{window}'] = ind
            else: 
                return ind

        if self.type == 'multiple':
            ind = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].rolling(window, min_periods=1).min()
                df.name = (symbol, f'min{window}')
                ind.append(df)
            
            ind = pd.concat(ind, axis=1)
            if inplace:
                self[ind.columns] = ind
                self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

            else:
                return ind

    def MAX(self, window, ref='close', inplace=False):
        """
        최근 N일 저가
        """
        if self.type == 'single':
            ind = self[ref].rolling(window, min_periods=1).max()
            ind.name = f'max{window}'
            if inplace:
                self[f'max{window}'] = ind
            else: 
                return ind

        if self.type == 'multiple':
            ind = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].rolling(window, min_periods=1).max()
                df.name = (symbol, f'max{window}')
                ind.append(df)
            
            ind = pd.concat(ind, axis=1)
            if inplace:
                self[ind.columns] = ind
                self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

            else:
                return ind



        


