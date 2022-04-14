import os

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    
    dates = quotes.index.values
    o = quotes['open'].values
    h = quotes['high'].values
    l = quotes['low'].values
    c = quotes['close'].values
    
    if period == 'day':
        #quotes['close']>= quotes['open']
        
        #x축 세팅 bar와 bar사이의 간격 설정
        if np.issubdtype(dates.dtype, np.datetime64): # date column 의 type이 datetime형식인 경우
            offset = np.timedelta64(8, 'h')
        elif np.issubdtype(dates.dtype, np.integer):
            offset = 0.3
        
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
    return ax


def view(data, period='day', size=(10,6), colors=['k','k']):
    """
    datetime index의 ohlc dataframe 인풋
    """
    fig = plt.figure(figsize=size)
    ax = plt.gca()
    ohlc_chart(ax, data,period=period, colors=colors)
    fig.autofmt_xdate()