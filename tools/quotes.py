"""
시장의 일별 데이터를 담고 있는 오브젝트
기본적으로 ohlc 값을 가지고 있으며
다른 지표들 ex) ma, ema 등을 반환하거나 기존 테이블에 추가할 수 있다.


"""
import os
import pandas as pd
import numpy as np
import h5py

from tools.constants import DATADIR

from .datatools import norm

class Quotes(pd.DataFrame):
    """
     일별 상품 정보 (OHLC 등) 클래스로,
     pandas DataFrame의 하위 클래스
    """
    def __init__(self, data):
        """
        type: single or multi
        """
        super().__init__(data) 

        if 'symbol' in data.attrs:
            self.attrs['symbol'] = data.attrs['symbol']

        #metric 값들의 형식을 저장 (plotting 용도로 사용)
        self.attrs['axes'] = {
            'ohlc': ['EMA','MA','MAX','MIN', 'PD'],
            'atr': ['ATR'],
            'trend': ['TREND']
        }

        if len(self.columns.names) == 1:
            self.type = 'single'
        elif  len(self.columns.names) == 2:
            self.type = 'multiple'

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


    def MA(self, period, ref='close', inplace=False, fieldname=None):
        """
        N-day Moving Average
     period: 기준일  period)
        ref : 기준 가격 (open, high, low, close)
        inplace: True 시 기존 quote에 병합하여 리턴 
        fieldname: pandas dataframe column name
        """
        fieldname = fieldname if fieldname else f'ma{period}'
        
        
        if self.type == 'single':
            ma = self[ref].rolling(period, min_periods=1).mean()
            ma.name = fieldname
            
            if inplace:
                return Quotes(self.join(ma, how='left'))
                #self[fieldname] = ma
            else: 
                return ma

        if self.type == 'multiple':
            ma = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].rolling(period, min_periods=1).mean()
                df.name = (symbol, fieldname)
                ma.append(df)
            
            ma = pd.concat(ma, axis=1)
            if inplace:
                #self[ma.columns] = ma
                #self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
                return Quotes(self.join(ma, how='left').sort_index(axis=1, level=0, sort_remaining=False))
            else:
                return ma

    
    def EMA(self, period, ref='close', inplace=False, fieldname=None):
        """
        N-day Exponential Moving Average
        period: 기준일  period)
        ref : 기준 가격 (open, high, low, close)
        inplace: True 시 기존 quote에 병합하여 리턴 
        fieldname: dataframe column name
        """
        fieldname = fieldname if fieldname else f'ema{period}'
        
        if self.type == 'single':
            ema = self[ref].ewm(span=period).mean()
            ema.name = fieldname
            if inplace:
                return Quotes(self.join(ema, how='left'))
                #self[fieldname] = ema
            else: 
                return ema

        if self.type == 'multiple':
            ema = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].ewm(span=period).mean()
                df.name = (symbol, fieldname)
                ema.append(df)
            
            ema = pd.concat(ema, axis=1)
            if inplace:
                #self[ema.columns] = ema
                #self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
                return Quotes(self.join(ema, how='left').sort_index(axis=1, level=0, sort_remaining=False))
            else:
                return ema

    
    def ATR(self, period=20, inplace=False, fieldname=None):
        """
        Average True Range (변동성 지표)
        fieldname: pandas dataframe column name
        """
        fieldname = fieldname if fieldname else f'atr{period}'

        if self.type == 'single':
            df = pd.DataFrame()
            df['hl'] = self['high'] - self['low']
            df['hc'] = np.abs(self['high'] - self['close'].shift(1))
            df['lc'] = np.abs(self['low'] - self['close'].shift(1))
            #df['TR'] = df.max(axis=1)
            atr = df.max(axis=1).ewm(span=period).mean()
            
            #normalization (z-score) - 지난 1년 데이터 기준
            atr = (atr - atr.rolling(period=250, min_periods= 1).mean())/atr.rolling(period=250, min_periods= 1).std()
            #atr = norm(atr)
            atr.name = fieldname
            if inplace:
                return Quotes(self.join(atr, how='left'))
            else:
                return atr

        
        elif self.type == 'multiple':
            atrs=[]
            for symbol, quote in self.groupby(level=0, axis=1):
                df = pd.DataFrame()
                df['hl'] = quote[symbol, 'high'] - quote[symbol, 'low']
                df['hc'] = np.abs(quote[symbol, 'high'] - quote[symbol, 'close'].shift(1))
                df['lc'] = np.abs(quote[symbol, 'low'] - quote[symbol, 'close'].shift(1))
                atr = df.max(axis=1).ewm(span=period).mean()
                atr = (atr - atr.rolling(period=200, min_periods= 1).mean())/atr.rolling(period=200, min_periods= 1).std()
                atr.name = (symbol, fieldname)
                atrs.append(atr)
            atrs = pd.concat(atrs, axis=1)

            if inplace:
                #self[atrs.columns] = atrs
                #self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
                return Quotes(self.join(atrs, how='left').sort_index(axis=1, level=0, sort_remaining=False))

            else:
                return atrs


    def MIN(self, period, ref='low', inplace=False, fieldname=None):
        """
        최근 N일 저가
        """
        fieldname = fieldname if fieldname else f'min{period}'

        if self.type == 'single':
            ind = self[ref].rolling(period, min_periods=1).min()
            ind.name = fieldname
            if inplace:
                #self[fieldname] = ind
                return Quotes(self.join(ind, how='left'))
            else: 
                return ind

        if self.type == 'multiple':
            ind = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].rolling(period, min_periods=1).min()
                df.name = (symbol, fieldname)
                ind.append(df)
            
            ind = pd.concat(ind, axis=1)
            if inplace:
                #self[ind.columns] = ind
                #self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
                return Quotes(self.join(ind, how='left').sort_index(axis=1, level=0, sort_remaining=False))

            else:
                return ind

    def MAX(self, period, ref='high', inplace=False, fieldname=None):
        """
        최근 N일 고가
        return: data, type
        """
        fieldname = fieldname if fieldname else f'max{period}'

        if self.type == 'single':
            ind = self[ref].rolling(period, min_periods=1).max()
            ind.name = fieldname
            if inplace:
                return Quotes(self.join(ind, how='left'))
            else: 
                return ind

        if self.type == 'multiple':
            ind = []
            for symbol, quote in self.groupby(level=0, axis=1):
                df = quote[symbol, ref].rolling(period, min_periods=1).max()
                df.name = (symbol, fieldname)
                ind.append(df)
            
            ind = pd.concat(ind, axis=1)
            if inplace:
                #self[ind.columns] = ind
                #self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
                return Quotes(self.join(ind, how='left').sort_index(axis=1, level=0, sort_remaining=False))

            else:
                return ind
    

    def PD(self, percentile, tau=50, inplace=False, fieldname=None):
        """
        Price Density
        return: data, type
        """
        fieldname = fieldname if fieldname else f'pd{percentile}'
        db_path = os.path.join(DATADIR,'price density',f'percentiles_Tau_{tau}.hdf')

        if self.type == 'single':
            symbol = self.attrs['symbol']
            file = h5py.File(db_path, 'r')
            data = file[symbol]['percentiles'][:,percentile]
            dates = file[symbol]['dates'][:].astype('M8[s]')
            file.close()
            ind = pd.Series(data=data, index=dates)
            ind.name = fieldname
            if inplace:
                #self[fieldname] = ind
                return Quotes(self.join(ind, how='left'))
            else: 
                return ind

        if self.type == 'multiple':
            ind = []
            symbols = self.columns.levels[0]
            for symbol in symbols:
                file = h5py.File(db_path, 'r')
                data = file[symbol]['percentiles'][:,percentile]
                dates = file[symbol]['dates'][:].astype('M8[s]')
                file.close()
                df = pd.Series(data=data, index=dates)
                df.name = (symbol, fieldname)
                ind.append(df)
            
            ind = pd.concat(ind, axis=1)
            if inplace:
                #self[ind.columns] = ind
                #self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
                return Quotes(self.join(ind, how='left').sort_index(axis=1, level=0, sort_remaining=False))

            else:
                return ind
            
    def TREND(self, period, direction, threshold, inplace=False, fieldname=None):
        """
        N-days Trend Index
        direction: 'down' or 'side' or 'up'
        period: integer , only 50 now
        """
        def calc_trend(data):
            #연속지속일수 계산
            # threshold 값보다 크거나 같으면 1(추세) 작으면 0(비추세)
            data[data < threshold] = 0
            data[data >= threshold] = 1
            data = data.astype('int')

            # 추세가 바뀌는 인덱스 찾기
            diff=np.diff(data)
            idx=np.where( (diff==1) | (diff==-1))[0]+1 

            # 추세 그룹으로 나눔
            groups = np.split(data, idx)
            # 각추세마다 지속일수를 계산하고 다시 합침
            return np.concatenate([group.cumsum() for group in groups])

        fieldname = fieldname if fieldname else f'trend{direction}{period}'
        db_path = os.path.join(DATADIR,'trend index','trend_index.hdf')
        tidx = ['','down','side','up'].index(direction)

        if self.type == 'single':
            symbol = self.attrs['symbol']
            file = h5py.File(db_path, 'r')
            data = file[symbol][f"trend{period}"][:, tidx]
            dates = file[symbol][f"trend{period}"][:, 0].astype('M8[s]')
            file.close()

           
            ind = pd.Series(data=calc_trend(data), index=dates)
            ind.name = fieldname
            if inplace:
                #self[fieldname] = ind
                return Quotes(self.join(ind, how='left'))
            else: 
                return ind
            
        if self.type == 'multiple':
            ind = []
            symbols = self.columns.levels[0]
            for symbol in symbols:
                file = h5py.File(db_path, 'r')
                data = file[symbol][f"trend{period}"][:, tidx]
                dates = file[symbol][f"trend{period}"][:, 0].astype('M8[s]')
                file.close()
                df = pd.Series(data=calc_trend(data), index=dates)
                df.name = (symbol, fieldname)
                ind.append(df)
            
            ind = pd.concat(ind, axis=1)
            if inplace:
                #self[ind.columns] = ind
                return Quotes(self.join(ind, how='left').sort_index(axis=1, level=0, sort_remaining=False))
                #self.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

            else:
                return ind





        


