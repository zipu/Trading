import os

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


"""
바 차트
"""
def ohlc_chart(ax, quotes, linewidth=1, colors=['k','k'], background='lightgoldenrodyellow'):
    """ 
    OHLC Bar Chart
    ax: matplot lib axes instance
    quotes: pandas dataframe 형식의 OHLC 데이터로써 index는 날짜여야 하고, 
            column 이름으로 open, high, low, close를 포함해야 함
    colors: 상승, 하락봉 색깔
    background: 배경색
    """
    cond = quotes['close']>= quotes['open']
    #x축 세팅 bar와 bar사이의 간격 설정
    if quotes.index.dtype == 'M8[ns]':
        offset = np.timedelta64(8, 'h')
    elif quotes.index.dtype == 'int32':
        offset = 0.3
    
    #상승bar drawing
    for i, data in enumerate(iter([quotes[cond], quotes[~cond]])):
        dates = data.index
        ax.vlines(dates, data['low'], data['high'], linewidth=linewidth, color=colors[i])
        ax.hlines(data['open'], dates-offset, dates, linewidth=linewidth, color=colors[i])
        ax.hlines(data['close'], dates, dates+offset, linewidth=linewidth, color=colors[i])
    
    #style
    ax.grid(linestyle='--')
    ax.set_facecolor(background)
    ax.yaxis.tick_right()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    return ax


def view(data, size=(10,6)):
    """
    datetime index의 ohlc dataframe 인풋
    """
    fig = plt.figure(figsize=size)
    ax = plt.gca()
    ohlc_chart(ax, data, colors=['r','b'])
    fig.autofmt_xdate()