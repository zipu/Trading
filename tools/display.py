import os
import re

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.dates as mdates
from matplotlib.ticker import Formatter, FormatStrFormatter



class MyFormatter(Formatter):
    """
        OHLC 차트 표현시 X축 표시형식을 나타내는 formatter
        matplotlib 의 dateformatter 로는 주말 또는 휴일이 공백으로 보이기 때문에
        trading day들을 Integer index로 표현한후 format을 datetime으로 변경
    """
    def __init__(self, dates, fmt='%Y-%m-%d'):
        self.dates = dates
        self.fmt = fmt
        self.xidx = []

    def __call__(self, x, pos):
        'Return the label for time x at position pos'
        ind = int(np.round(x))
        self.xidx.append(ind)
        
        if ind < 0:
            return ''
        
        if ind >= len(self.dates):
            #print(self.xidx)
            diff = self.xidx[1] - self.xidx[0]
            #print((self.dates.iloc[-1] + pd.Timedelta(days=diff)).strftime(self.fmt))
            return (self.dates[-1] + pd.Timedelta(days=diff)).strftime(self.fmt)

        
        return self.dates[ind].strftime(self.fmt)




"""
바 차트
"""
def ohlc_chart(ax, quotes, period='day', linewidth=1, colors=['k','k'], background='lightgoldenrodyellow'):
    """ 
    OHLC Bar Chart
    ax: matplot lib axes instance
    quotes: pandas dataframe 형식의 OHLC 데이터로써 index는 날짜여야 하고, 
            column 이름으로 일봉은 open, high, low, close를 30분봉은 high, low를 포함해야 함
    colors: 상승, 하락봉 색깔
    background: 배경색
    """
    #from pandas.plotting import register_matplotlib_converters
    #register_matplotlib_converters()

    # 인덱스가 날짜이면 이를 컬럼으로 옮김
    
    #if np.issubdtype(quotes.index.values.dtype, np.datetime64):
        #quotes.reset_index(inplace=True)
        
        # x 축 티커를 날짜로 표시 
    
    dates = np.array(range(len(quotes.index)))
    myformatter = MyFormatter(quotes.index)
    ax.xaxis.set_major_formatter(myformatter)
    
    #dates = quotes.index.values
  
    o = quotes['open'].values
    h = quotes['high'].values
    l = quotes['low'].values
    c = quotes['close'].values
    
    if period == 'day':
        #quotes['close']>= quotes['open']
        
        #x축 세팅 bar와 bar사이의 간격 설정
        #if np.issubdtype(dates.dtype, np.datetime64): # date column 의 type이 datetime형식인 경우
        #    offset = np.timedelta64(8, 'h')
        
        #elif np.issubdtype(dates.dtype, np.integer):
        offset = 0.3 # 오른쪽, 왼쪽으로 뻗은 선의 길이
        
        if colors[0] == colors[1]:
            ax.vlines(dates, l, h, linewidth=linewidth, color=colors[0])
            ax.hlines(o, dates-offset, dates, linewidth=linewidth, color=colors[0])
            ax.hlines(c, dates, dates+offset, linewidth=linewidth, color=colors[0])
    
        else:
            cond =  c >= o
            #상승bar drawing
            for i, idx in enumerate(iter([cond, ~cond])):
                #dates = data.index.values
                ax.vlines(dates[idx], l[idx], h[idx], linewidth=linewidth, color=colors[i])
                ax.hlines(o[idx], dates[idx]-offset, dates[idx], linewidth=linewidth, color=colors[i])
                ax.hlines(c[idx], dates[idx], dates[idx]+offset, linewidth=linewidth, color=colors[i])
    
    elif period == 'minute':
        #drawing bars
        #dates = quotes.index.values
        ax.vlines(dates, l, h, linewidth=linewidth, color='k')

    #style
    ax.grid(linestyle='--')
    ax.set_facecolor(background)
    ax.yaxis.tick_right()
    #ax.xaxis.set_major_locator(mdates.MonthLocator())
    #ax.xaxis.set_minor_locator(mdates.DayLocator())
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    #ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    #x값을 날짜로 변환

    return ax

def oi_chart(ax, quotes, period='day', linewidth=1, colors=['k','k'], background='lightgoldenrodyellow'):
    """
    미결제 약정을 보여주는 보조 차트
    """
    #formatting
    dates = np.array(range(len(quotes.index)))
    myformatter = MyFormatter(quotes.index)
    ax.xaxis.set_major_formatter(myformatter)
    
    #data
    oi = quotes.values /1000
    
    #drawing
    ax.vlines(dates, 0, oi, linewidth=linewidth, color='k')
    
    #style
    ax.grid(linestyle='--')
    ax.set_facecolor('lightgoldenrodyellow')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d k'))
    ax.yaxis.tick_right()
    ax.set_title('Open interest', loc='left')
    
    return ax

def indicator_chart(ax, quotes,name, linewidth=1):
    """
    지표 차트
    """
    dates = np.array(range(len(quotes.index)))
    
    if isinstance(quotes, pd.Series):
        ax.plot(dates, quotes, label=name)
    
    else:
        for column in quotes.columns:
            ax.plot(dates, quotes[column])
    
    ax.legend(loc=2)
    return ax

def index_chart(ax, quotes, name, color='purple'):
    """
    인덱스 차트: 별도의 plot에 생성
    """
    dates = np.array(range(len(quotes.index)))

    if isinstance(quotes, pd.Series):
        ax.plot(dates, quotes, color=color, label=name)
    
    else:
        for column in quotes.columns:
            ax.plot(dates, quotes[column], color=color)

    #legend
    ax.legend(loc=2)
    
    #style
    ax.grid(linestyle='--')
    ax.set_facecolor('lightgoldenrodyellow')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.tick_right()
    #ax.set_title(name, x=0.03, y=1.0, pad=-14)
    
    return ax





def view(data, period='day', size=(10,6), colors=['k','k'],\
         metrics=None, title='OHLC Chart'):
    """
    pandas dataframe 의 일데이터를 차트로 그려주는 함수
    data: 일 데이터
    """
    has_metrics = True if metrics is not None else None
    
    # 에러: indicator와 data가 매칭이 안되면 에러
    if has_metrics and not (data.index == metrics.index).all():
        raise ValueError("일봉 데이터와 지표 데이터가 매칭 되지 않습니다 ")
    
    # figure의 grid 결정 (plot 갯수 확인)
    # index 타입인 indicator의 갯수만큼 axes 를 늘림
    cnt = 0
    if has_metrics:
        indicator_type = metrics.attrs['type']
        cnt += len(['flag' for v in indicator_type.values() if v == 'index'])

    # x축 format을 datetime 형식으로 할 경우 거래가 없는 주말에 빈공간으로 나옴. 
    # 따라서 index를 integer 형식으로 나타낸 후, 해당 숫자에 해당하는 날짜로 x축 
    # tick format만 변경함
    
    #fig = plt.figure(figsize=size)
    #ax = plt.gca()
    xsize = size[0]
    ysize = size[1]*(1 + 0.2*cnt)
    fig, (ax) = plt.subplots(2+cnt,1, figsize=(xsize, ysize), sharex=True, gridspec_kw = {'height_ratios':[4,1]+[1]*cnt})
    ohlc_axes = ax[0]
    oi_axes = ax[1]

   
    ohlc_chart(ohlc_axes, data, period=period, colors=colors)
    
    # 컬럼 명을 두종류를 사용해서 종종 문제가 됨.. 언젠간 해결해야.
    if 'open_interest' in data.columns:
        oi = 'open_interest'
    elif 'oi' in data.columns:
        oi = 'oi'
    
    oi_chart(oi_axes, data[oi], period=period, colors=colors)

    if has_metrics:
        indicator_type = metrics.attrs['type']
        
        cnt=0
        for name in metrics.columns:
            if indicator_type[name] == 'price':
                ind_axes = ax[0]
                indicator_chart(ind_axes, metrics[name], name)

            elif indicator_type[name] == 'index':
                index_axes = ax[2+cnt]
                index_chart(index_axes, metrics[name], name)
                cnt += 1
        
    
    ohlc_axes.set_title(title, loc='left')
    fig.autofmt_xdate()
    plt.close()
    return fig