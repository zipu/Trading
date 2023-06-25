"""
사용자가 등록한 시스템을 오브젝트화 함
"""
#from multiprocessing import log_to_stderr
#from os import stat
import os
from datetime import datetime
from collections import defaultdict
import re
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

from tools.instruments import instruments
from .trades import TradesBook
from .equity import EquityBook
from .heat import DefaultHeat

LONG = 1
SHORT = -1


class System:
    def __init__(self, abstract, quotes, id):
        """
        abstract: 시스템 설정 내용
        qutoes: indicator 생성용
        """
        self.id = id
        self.abstract = abstract
        self.from_date = datetime.strptime(abstract['from_date'], '%Y-%m-%d')
        
        if not abstract['to_date']:
            self.to_date = datetime.strptime(quotes.index[-1].strftime('%Y-%m-%d'), '%Y-%m-%d')
        else:
            self.to_date = datetime.strptime(abstract['to_date'], '%Y-%m-%d')

        self.name = abstract['name']
        self.description = abstract['description']
        
        # 설정파일에 거래 상품 리스트가 없으면 db 전체 상품으로 진행
        if abstract['instruments']:
            self.symbols = abstract['instruments'] #상품 코드 목록
        else:
            self.symbols = quotes.columns.levels[0].to_list()
        
        self.instruments = [instruments[symbol] for symbol in self.symbols]
        
        #섹터정보
        self.sectors = defaultdict(list)
        if abstract['sectors'] == 'default':
            for symbol in self.symbols:
                self.sectors[instruments[symbol].sector].append(symbol)
        
        
        #매매환경
        self.commission = abstract['commission']
        self.skid = abstract['skid']

        #자본금
        self.principal = abstract['principal'] 
        
        #추후 Heat System을 모듈화 해서 사용할 계획 
        #한도 설정 
        # 시스템 허용 위험한도 : 전체 자산 대비 
        self.heat = eval(abstract['heat_system'])( 
            abstract['max_system_heat'],
            abstract['max_sector_heat'],
            abstract['max_trade_heat'],
            abstract['max_lots']
        )
        
        #재무 상태 내역서 
        self.equity = EquityBook(self.principal, self.from_date)

        # 매매 내역서 
        self.trades = TradesBook(self.id)
        

        #지표 생성
        self.metrics = pd.DataFrame() #지표 생성
        self.metrics.attrs['name'] = self.name
        self.create_metrics(abstract['metrics'], quotes)
        
        #생성된 지표 일봉데이터 결합
        quotes = pd.concat([quotes, self.metrics], axis=1) 

        #매매 시그널 생성
        self.signals = pd.DataFrame() #시그널 생성
        self.signals.attrs['name'] = self.name

        self.entry_rule = abstract['entry_rule']
        if self.entry_rule['long']:
            self.create_signal(self.entry_rule['long'], 'enter_long', quotes)
        
        if self.entry_rule['short']:
            self.create_signal(self.entry_rule['short'], 'enter_short', quotes)
        
        self.exit_rule = abstract['exit_rule']
        if self.exit_rule['long']:
            self.create_signal(self.exit_rule['long'], 'exit_long', quotes)
        if self.exit_rule['short']:
            self.create_signal(self.exit_rule['short'], 'exit_short', quotes)


        self.stop_rule = abstract['stop_rule']
        if self.stop_rule['long']:
            self.create_stops(self.stop_rule['long'], 'stop_long', quotes)
        if self.stop_rule['short']:
            self.create_stops(self.stop_rule['short'], 'stop_short', quotes)

        # OHLC 데이터 없는날, index type등 처리
        self.compensate_signals(quotes)

        #self.set_nans(self.metrics, quotes)
        #self.set_nans(self.signals, quotes)


    def __repr__(self):
        return f"<<시스템: {self.name}>>"



    def create_metrics(self, metrics, quotes):
        """
        configurator 파일에 등록된 사용 인디케이터를 pd Dataframe으로 생성
        - metrics: 등록할 지표 저보
        - qutoes: 시장 일봉데이터
        """
        lists = []
        for indicator in metrics:
            #window = [x.split('=')[1] for x in indicator if 'window' in x][0]
            fieldname = indicator[0]
            ind = indicator[1]
            params = ','.join(indicator[2:])
            #fieldname = f"{indicator[0]}{window}_{system['name']}"
            
            #kwargs = ','.join(indicator[1:])
            params = eval(f'dict({params})')
            series = getattr(quotes, ind)(fieldname=fieldname, **params)
            lists.append(series)

        df = pd.concat([self.metrics]+lists, axis=1)
        self.metrics = df.sort_index(axis=1, level=0, sort_remaining=False)

        # plotting시 추가 axes 생성여부를 판단하기 위해 각 metric의 형식을 attribute로 저장
        metric_types = quotes.attrs['metric_types']
        numeric_type = {}
        for metric in self.abstract['metrics']:
            numeric_type[metric[0]] = [k for k,v in metric_types.items() if metric[1] in v][0]
        self.metrics.attrs['type'] = numeric_type

        # OHLC 데이터가 없는 거래일에 metric 값은 Averaging 결과로써 존재할 수 있음. 
        # 이를 NaN 값으로 변경 해줌
        """
        ohlc 데이타가 nan 인데도, averaging 과정에서 시그널이 있을 수 있음.
        이를 nan으로 바꾸는 함수
        """
        for symbol in self.symbols:
            flag = quotes[symbol][['open','high','low','close']].isna().any(axis=1)
            self.metrics.loc[flag, symbol] = np.nan



    def create_signal(self, rules, name, quotes):
        """
        사용자 정의된 매수, 매매 규칙에 해당되는 날에 시그널(True or False) 생성
        - rules: 매매규칙
        - name: 매매이름
        - quotes: 시장 일봉데이터
        """
        if not rules:
            signals = pd.DataFrame()
            for symbol in self.symbols:
                signals[symbol,name] = float('nan')
            self.signals = pd.concat([self.signals, signals], axis=1)
            self.signals.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
            return
        
        #quotes = pd.concat([quotes, self.metrics], axis=1)
        signals = []
        for symbol in self.symbols:
            condition = rules
            df = quotes[symbol]
            fields = df.columns.to_list()
            for field in fields:
                condition = re.sub(f"(?<![A-Za-z0-9])({field})(?![A-Za-z0-9])", f"df['{field}']", condition)
            #self.quotes[symbol,'buy_signal'] = eval(rule)
            signal = eval(condition)
            signal.name = (symbol, name)
            signals.append(signal)
        signals = pd.concat(signals, axis=1)
        self.signals = pd.concat([self.signals, signals], axis=1)
        #self.quotes[signals.columns] = signals
        self.signals.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

    def create_stops(self, rules, name, quotes):
        """
        사용자 정의된 규칙에 따른 스탑 가격 생성
        """
        if not rules:
            signals = pd.DataFrame()
            for symbol in self.symbols:
                signals[symbol,name] = float('nan')
            self.signals = pd.concat([self.signals, signals], axis=1)
            self.signals.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
            return

        #quotes = pd.concat([quotes, self.metrics], axis=1)
        signals = []
        for symbol in self.symbols:
            df = quotes[symbol][rules]
            df.name = (symbol, name)
            signals.append(df)
        
        signals = pd.concat(signals, axis=1)
        self.signals = pd.concat([self.signals, signals], axis=1)
        self.signals.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)
        #self.signals.index = self.signals.index.astype('M8[ns]')

    def compensate_signals(self, quotes):
        """
        1. 인덱스 형식 object -> datetime 으로 변경
        """
        self.signals.index = self.signals.index.astype('M8[ns]')

        """
        1. ohlc 데이타가 nan 이면 signal도 모두 nan으로 변경
        2. signal이 모두 nan인 날짜의 signal을 직전 거래일 값으로 변경

        """
        starts = {} #첫 거래 시작일
        for symbol in self.symbols:
            flag = quotes[symbol][['open','high','low','close']].isna().any(axis=1)
            self.signals.loc[flag, symbol] = np.nan
            starts[symbol] = quotes[symbol][~flag].index[0]
        
        for symbol in self.symbols:
            start = starts[symbol]
            while self.signals[symbol].loc[start:].isna().all(axis=1).any():
                idx = self.signals[symbol].loc[start:].isna().all(axis=1)
                mask = idx[idx==True].index
                self.signals.loc[mask, symbol] = self.signals[symbol].shift(1).loc[mask].values



    
    def trade(self, quote):
        """
        * 매매 진행
        yesterday: 전거래일(yesterday) 장 종료 후 매매여부 판단 후
        quote: 오늘 날짜의 quote 가격으로 매매 진행 
        
        %% 매매시 고려 사항 및 진행 순서 %%
        1. 진행중인 매매가 있는가?
            Yes -> 청산신호 발생? Yes -> 청산
                                 No -> pass
            No -> pass

        2. 신규진입 시그널이 있는가?
            No -> pass
            Yes -> heat 계산 -> 계약수 결정 -> 매매진입 -> heat 갱신
        """
        today = quote.name
        datesindex = self.metrics.index
        yesterday = datesindex[datesindex.get_loc(today) - 1]
        
        # 전일 장 종료 후 시그널로 매매 여부 판단
        signals = self.signals.loc[yesterday]

        """
        1. 거래 가능 상품 선별
          
          상품마다 특정 날짜에 데이터가 존재하지 않을 수 있음
          이 경우 해당 상품의 당일 시그널을 전일 시그널과 같도록 변경하고 거래 상품 목록에서 제외
        """
        #symbols = []
        mask = quote.isna().groupby('symbol').all()
        symbols = list(mask[~mask].index)
        #print(symbols)

        #for symbol in self.symbols:
        #    ohlc = quote[symbol][['open','high','low','close']]

        #    if ohlc.isna().any():
        #        self.signals.loc[today, symbol] = self.signals.loc[yesterday, symbol].values
        #    else:
        #        symbols.append(symbol)

        """
        2. 진행 중인 매매 청산

          진행중인 매매 상품 중 당일 거래 가능하면, 전일 청산 신호가 발생 했을 시 시초가 + skid 에 청산
        """
        
        #진행중인 매매목록
        fires = self.trades.get_on_fires() 
        fires = [fire for fire in fires if fire.symbol in symbols]
        
        for fire in fires:
            symbol = fire.symbol
            if fire.position == LONG and signals[symbol]['exit_long']:
                self.exit(fire, quote[symbol], type='exit')
                #자산 상태 업데이트
                #if self.equity.update(today, self.trades, self.heat):
                #  return True #시스템 가동 중단

            elif fire.position == SHORT and signals[symbol]['exit_short']:
                self.exit(fire, quote[symbol], type='exit')
                #자산 상태 업데이트
                #if self.equity.update(today, self.trades, self.heat):
                #    return True #시스템 가동 중단
        
        #자산 상태 업데이트
        if self.equity.update(today, self.trades, self.heat):
                    return True #시스템 가동 중단
        
        """
        3. 매매 진입

          전일 진입 신호 발생시, 시초가(+skid) 진입
        """
        for symbol in symbols:
            if signals[symbol].get('enter_long'):
                #stopprice = signals_today[symbol]['stop_long']
                self.enter(symbol, quote[symbol], LONG)
                if self.equity.update(today, self.trades, self.heat):
                    return True #시스템 가동 중단
                
            elif signals[symbol].get('enter_short'):
                #stopprice = signals_today[symbol]['stop_short']
                self.enter(symbol, quote[symbol], SHORT)
                if self.equity.update(today, self.trades, self.heat):
                    return True #시스템 가동 중단
                
            


        #3. STOP 청산 및 stop 가격 업데이트
        
        #오늘 거래가능 종목중 진행중인 매매목록
        fires = self.trades.get_on_fires() 
        fires = [fire for fire in fires if fire.symbol in symbols]

        for fire in fires:
            if fire.position == LONG and fire.stopprice >= quote[fire.symbol]['low']:
                self.trades.add_exit(fire, today, fire.stopprice, fire.lots, 'stop' )

            elif fire.position == SHORT and fire.stopprice <= quote[fire.symbol]['high']:
                self.trades.add_exit(fire, today, fire.stopprice, fire.lots, 'stop' )

            else:
                if fire.position == LONG:
                    stopprice = self.signals.loc[today][fire.symbol]['stop_long']
                elif fire.position == SHORT:
                    stopprice = self.signals.loc[today][fire.symbol]['stop_short']

                fire.stopprice == stopprice
                fire.update_status(quote[fire.symbol]['close'], stopprice)

        if self.equity.update(today, self.trades, self.heat):
            return True #시스템 가동 중단

    
                
    def enter(self, symbol, quote, position):
        """
        매매 진입
        """

        # 해당 상품을 이미 최대 계약수로 매매하고 있다면 스킵
        if sum([fire.lots for fire in self.trades.get_on_fires(symbol=symbol)]) >= self.heat.max_lots:
            return


        today = quote.name
        sector = instruments[symbol].sector
        order = {
            'entrydate': today,
            'symbol': symbol,
            'sector': sector,
            'position': position,
            'entryprice': self.price(symbol, quote, position, 'enter'),
        }

        if position == LONG:
            order['stopprice'] = self.signals.loc[today][symbol]['stop_long']
        elif position == SHORT:
            order['stopprice'] = self.signals.loc[today][symbol]['stop_short']

        risk_trade, risk_ticks = self.price_to_value(
            symbol, position, order['stopprice'], order['entryprice'], 1)
        
        #order['entryrisk'] = risk
        order['entryrisk_ticks'] = risk_ticks
        
        if risk_ticks < 0 :
            #self.trades.reject(symbol, today, sector, position, order['entryprice'])
            #return
            raise ValueError(f"리스크가 음수 또는 0일 수 없음: {order}")
        
        elif risk_ticks == 0:
            return 

        #계약수 결정
        lots = self.heat.calc_lots(symbol, sector, risk_ticks, self.equity)
        if lots == 0:
            #self.trades.reject(symbol, today, sector, position, order['entryprice'])
            return
        
        order['entrylots'] = lots
        order['entryrisk'] = risk_trade * lots
        order['commission'] = self.commission * lots
        
        self.trades.add_entry(**order)
        #self.update_status()
    

    def price_to_value(self, symbol, position, initial_price, final_price, lots):
        """
        상품 가격 차이로부터 가치를 계산
        """
        tickunit = instruments[symbol].tickunit
        tickvalue = instruments[symbol].tickvalue
        value_ticks = round(position*(final_price-initial_price)/tickunit)
        value = value_ticks * tickvalue * lots

        return value, value_ticks

    
    
    def exit(self, fire, quote, type='exit'):
        """
        청산 

        - 분할 매도 코드 추후 삽입
        """
        exitdate = quote.name
        exitprice = self.price(fire.symbol, quote, fire.position, 'exit')
        
        #####################################
        #추후 분할 청산 시스템 사용시 변경 필요#
        #####################################
        exitlots = fire.lots
        #####################################

        self.trades.add_exit(fire, exitdate, exitprice, exitlots, type)


    def price(self,symbol, quote, position, type):
        """ 
        매매가격결정 
         position: LONG or SHORT
         type: enter or exit
            1)'buy' : open + (high - open) * skid
            2)'sell': open - (open - low) * skid 
        """
        tickunit = instruments[symbol].tickunit

        if position==LONG and type == 'enter':
            tick_diff = (quote['high'] - quote['open'])/tickunit
            skid = round(tick_diff*self.skid)*tickunit
            return quote['open'] + skid

        elif position == LONG and type == 'exit':
            tick_diff = (quote['open'] - quote['low'])/tickunit
            skid = round(tick_diff*self.skid)*tickunit
            return quote['open'] - skid

        elif position == SHORT and type == 'enter':
            tick_diff = (quote['open'] - quote['low'])/tickunit
            skid = round(tick_diff*self.skid)*tickunit
            return quote['open'] - skid

        elif position == SHORT and type == 'exit':
            tick_diff = (quote['high'] - quote['open'])/tickunit
            skid = round(tick_diff*self.skid)*tickunit
            return quote['open'] + skid
        
    #def set_nans(self, df, quotes):
    #    """
    #    ohlc 데이타가 nan 인데도, averaging 과정에서 시그널이 있을 수 있음.
    #    이를 nan으로 바꾸는 함수
    #    df: self.signals 또는 self.metrics
    #    """
    #    for symbol in self.symbols:
    #        flag = quotes[symbol][['open','high','low','close']].isna().any(axis=1)
    #        df.loc[flag==True, symbol] = np.nan

    
    def equity_plot(self):
        
        equitylog = pd.DataFrame(self.equity.log()).set_index('date')
        equity = equitylog.groupby(by='date').last()
        x = equity.index.values
        capital = equity.capital.values
        fixed_capital = equity.fixed_capital.values
        principal = (capital - equity.flame).values #매매직전원금
        max_capital = equity.max_capital.values
        #commission = equity.commission.values #누적 수수료
        p = self.principal #투자원금

        fig, ax = plt.subplots(1,1, figsize=(15, 12))
        ax.fill_between(x, p, fixed_capital, where=fixed_capital>=p, facecolor='green', alpha=0.4, interpolate=True, label='fixed value=total value - risk')
        ax.fill_between(x, p, fixed_capital, where=fixed_capital<p, facecolor='red', alpha=0.6, interpolate=True)
        ax.fill_between(x, capital, max_capital, color='grey', alpha=0.2)

        ax.plot(x, principal, color='black',alpha=0.7, linewidth=1, label='cash')
        ax.plot(x, capital, color='orange',alpha=0.7, linewidth=1, label='total value = cash + flame - commission')
        #ax.plot(x, commission, color='grey', alpha=0.7, linestyle='--', label='commission')

        ax.set_xlim([x.min(), x.max()])

        #reference curve
        rate = 0.1 #annual interest rate
        refx = (x-x[0])/np.timedelta64(365,'D')
        refy = p*np.exp(rate*refx)
        ax.plot(x, refy, color='magenta', linestyle='--', label='reference (10%)')

        #labels
        ax.legend(loc='upper left', fontsize='large')
        ax.set_title('Equity Chart', fontsize=17)

        #ax.set_xlabel('Date', fontsize=15)
        ax.set_ylabel('equity ($)', fontsize=12)
        ax.yaxis.set_label_position("right")
        #style
        ax.grid(linestyle='--')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.yaxis.tick_right()
        fig.autofmt_xdate()
                
        #plt.show()
        return fig

    def summary(self, level=0):
        if level == 0:
            self.equity_plot()
            #plt.show()
            return self.total_result()
        
        elif level == 1:
            return self.trade_result(level='sector')
        
        elif level == 2:
            return self.trade_result(level='name')


    def total_result(self):
        win_trades = self.trades.get(result='WIN')
        lose_trades = self.trades.get(result='LOSE')
        trades = win_trades + lose_trades
        cnt = len(trades)


        total = {
            '투자금': self.principal,
            '최종자산': self.equity.capital,
            '총손익': (self.equity.capital / self.principal) - 1,
            'Bliss': self.equity.cagr/self.equity.mdd,
            'CAGR': self.equity.cagr,
            'MDD': self.equity.mdd,
            '손익비': sum([t.profit for t in win_trades])/sum([t.profit for t in lose_trades]),
            '승률': len(win_trades) / cnt,
            '위험대비손익': sum([t.profit/t.entryrisk for t in trades])/cnt,
            '평균손익': sum([t.profit for t in trades])/cnt,
            '평균수익': sum([t.profit for t in win_trades])/len(win_trades),
            '평균손실': sum([t.profit for t in lose_trades])/len(lose_trades),
            #'손익표준편차': trade.profit.std(),
            '보유기간': sum(t.duration for t in trades)/cnt,
            '매매회수': cnt
        }
        
        report = pd.DataFrame([total], index=['Result'])
        
        report = report.style.format({
                    '투자금': lambda x: "{:,.0f}".format(x) if x else '',
                    '최종자산': lambda x: "{:,.0f}".format(x) if x else '',
                    '총손익': lambda x: "{:.1%}".format(x) if x else '',
                    'Bliss': lambda x: "{:,.3f}".format(x) if x else '',
                    'CAGR': lambda x: "{:,.1%}".format(x) if x else '',
                    'MDD': lambda x: "{:,.1f}".format(x)+"%" if x else '',
                    '손익비': "{:.2f}",
                    '승률': "{:.1%}",
                    '위험대비손익': "{:,.1%}",
                    '평균손익': "{:,.0f}",
                    '평균수익': "{:,.0f}",
                    #'손익표준편차': lambda x: "{:,.0f}".format(x) if x else '',
                    '평균손실': "{:,.0f}",
                    '보유기간': "{:,.0f} 일",
                    #'# trades': "{:.1f}"
                })
        

        return report

    def trade_result(self, level='name'):
        result = []
        tradelog = pd.DataFrame(self.trades.log())
        tradelog = tradelog[tradelog['result'] != 'REJECT']
        for lev, table in tradelog.groupby(level):
            
            #period = table['duration'].mean()
            #ave_num = len(table)/len(np.unique(table.symbol))
            
            trade = {
                #'symbol': symbol,
                '구분': lev,#self.pinfo[symbol]['name'],
                '총손익': table.profit.sum()+table.flame.sum(),
                '평균손익': table.profit.mean(),
                '표준편차': table.profit.std(),
                '위험대비손익': (table.profit/table.entryrisk).mean(),
                '승률': len(table[table.profit > 0])/len(table),
                '보유기간': table.duration.mean(),
                '매매회수': len(table)
            }
            result.append(trade)
        df = pd.DataFrame(result)
        df.set_index('구분', inplace=True)
        #del df.index.name
        #styling
        df = df.style.format({
                    '총손익': "{:.0f}",
                    '평균손익': "{:.0f}",
                    '표준편차': "{:.2f}",
                    '위험대비손익': "{:.2%}",
                    '승률': "{:.1%}",
                    #'# trades': "{:.f}"
                })
        df.data.sort_values(by='총손익', ascending=False, inplace=True)
        return df

    def detail_result(self, symbol, start=None, end=None):
        """
        상품별 매매 결과 결과를 그래프로 출력함
        """
        from functools import reduce
        from tools.display import MyFormatter
        from IPython.display import display


        #0. 환경 설정. 데이터 가공. 
        start = self.from_date if start == None else start
        end = self.to_date if end == None else end
        symbol = symbol
        name = instruments[symbol].name

        quotes = instruments.quotes()[symbol].loc[start:end]
        quotes.dropna(subset=['open','high','low','close'], inplace=True)

        datesindex = quotes.index
        quotes = quotes.reset_index()

        metrics = self.metrics[symbol].loc[datesindex]

        trades = self.trades.log(symbol=symbol)
        trades = pd.DataFrame(trades)
        #, columns=['entrydate','exitdate','position', 'entryprice', 'entrylots','entryrisk','#exits', 'profit','profit_ticks','duration','result'])
        

        if len(trades) == 0:
            print(f"No trades have made in given period: {symbol}")
            return (None, None)
        
        else:
            trades = trades[(trades.entrydate>=start) & (trades.entrydate<=end)]
            if len(trades) == 0:
                print(f"No trades have made in given period: {symbol}")
                return (None, None)

        #1. 차트 환경 설정
        index_rows = sum([1 for i in metrics.attrs['type'].values() if i == 'index'])
        xsize = 15
        ysize = 15*(1 + 0.2*index_rows)
        linewidth = 1

        fig, (ax) = plt.subplots(3+index_rows, 1, figsize=(xsize, ysize),
                    gridspec_kw = {'height_ratios':[4,1,1]+[1]*index_rows})
        
        #2. 성능차트
        win_masks = []
        lose_masks =[]

        for idx, row in trades.iterrows():
            mask = (quotes['date'] >= row['entrydate']) & (quotes['date'] <= row['exitdate'])
            if row['result'] == 'WIN':
                win_masks.append(mask)
            elif row['result'] == 'LOSE':
                lose_masks.append(mask)

        win_mask = reduce(np.logical_or, win_masks) if win_masks else quotes['date'] == -1
        lose_mask = reduce(np.logical_or, lose_masks) if lose_masks else quotes['date'] == -1

        win_dates = quotes[win_mask].index
        lose_dates = quotes[lose_mask].index
        neutral_dates = quotes[~win_mask & ~lose_mask].index
        dates = [win_dates, lose_dates, neutral_dates]
        colors = ['r', 'b', 'k']

        offset = 0.4

        for date, color in zip(dates, colors):
            o = quotes.loc[date, 'open'].values
            h = quotes.loc[date, 'high'].values
            l = quotes.loc[date, 'low'].values
            c = quotes.loc[date, 'close'].values

            ax[0].vlines(date, l, h, linewidth=linewidth, color=color)
            ax[0].hlines(o, date-offset, date, linewidth=linewidth, color=color)
            ax[0].hlines(c, date, date+offset, linewidth=linewidth, color=color)
        
        #3. 지표 그리기
        metric_type = {
            'price':[k for k, v in metrics.attrs['type'].items() if v=='price' ],
            'index':[k for k, v in metrics.attrs['type'].items() if v=='index' ]
        }

        for metric in metric_type['price']:
            ax[0].plot(quotes.index, metrics[metric], label=metric)


        # formats
        title = f"Trade Performance: {name} ({symbol})"
        ax[0].set_title(title, fontsize=20)
        ax[0].set_ylabel('Price')
        ax[0].legend(loc=2)
        myformatter = MyFormatter(datesindex)
        ax[0].xaxis.set_major_formatter(myformatter)

        #4.Index Chart
        for axis, name in zip(ax[1:-1], metric_type['index']):
            axis.set_title(name, loc='left')
            #axis.set_ylabel(name, fontsize=15)
            axis.plot(quotes.index, metrics[name])
            axis.set_xticks([])


        #5. Cummulative Profit Chart
        cumprofit = trades.profit_ticks.cumsum()
        idx = cumprofit.index
        ax[-2].plot(cumprofit, color='blue')
        ax[-2].axhline(y=0, linewidth=1, color='k')
        ax[-2].set_title('Cummulative Profit(tick)', loc='left')


        #4. Tick-Profit Chart
        #tick profit chart
        num_trades = len(trades)
        ax[-1].bar(np.arange(1,num_trades+1), np.where(trades.position=='Long', trades.profit_ticks, 0), 0.3, color='red', alpha=0.6 )
        ax[-1].bar(np.arange(1,num_trades+1), np.where(trades.position=='Short', trades.profit_ticks, 0), 0.3, color='blue', alpha=0.6 )
        ax[-1].set_title('Profit (tick)', loc='left')
        ax[-1].axhline(y=0, linewidth=1, color='k')
        #labels

        #common styles
        for axis in ax:
            axis.yaxis.tick_right()
            axis.set_facecolor('lightgoldenrodyellow')
            #ax[1].set_xticks(range(1,num_trades+1))
            axis.grid(linestyle='--')

        result = {
                #'symbol': symbol,
                '총손익': round(trades.profit.sum()),
                '총손익(틱)': round(trades.profit_ticks.sum()),
                '평균손익(틱)': round(trades.profit_ticks.mean()),
                #'표준편차(틱)': round(trades.profit_ticks.std()),
                '위험대비손익': (trades.profit/trades.entryrisk).mean(),
                '승률': len(trades[trades['result'] == 'WIN'])/len(trades),
                '보유기간': trades.duration.mean(),
                '매매회수': len(trades)
            }

        df = pd.DataFrame(result, index=['결과'])
        df = df.style.format({
                    '총손익': "{:,}",
                    '총손익(틱)': "{:,}",
                    '평균손익(틱)': "{:,}",
                    #'표준편차(틱)': "{:,}",
                    '위험대비손익': "{:.2%}",
                    '승률': "{:.1%}",
                    '보유기간': "{:.0f}"
                })
        #return df
        
        #display(df)
        return (fig, result)
        #return trades

    def create_report(self):
        """ 결과 보고서 작성 """
        #폴더 생성
        foldername = self.name + '_' + datetime.today().strftime('%Y%m%d%H%M')
        savedir = os.path.join('report',foldername) 
        os.mkdir(savedir)

        #1. 종합 결과
        #equity chart 이미지 파일 생성 및 저장 
        fig = self.equity_plot()
        fig.tight_layout()
        fig.savefig(os.path.join(savedir,'equity_chart.svg')) #equity_chart
        plt.close()


        result = self.summary().data.to_dict()
        for k,v in result.items():
            result[k] = v['Result']

        #2. 섹터 결과
        sector_table_rows=""
        for name, row in self.summary(level=1).data.iterrows():
            sector_table_rows += f"""\
            <tr align='center'>\
            <td>{name}</td>\
            <td>{row['총손익']:,.0f}</td>\
            <td>{row['평균손익']:,.0f}</td>\
            <td>{row['표준편차']:,.0f}</td>\
            <td>{row['위험대비손익']:.2f}</td>\
            <td>{row['승률']*100:.2f} %</td>\
            <td>{row['보유기간']:.0f}</td>\
            <td>{ row['매매회수']:.0f}</td>\
            </tr>\
            """
        
        #3. 종목별 결과
        product_table_rows=""
        for name, row in self.summary(level=2).data.iterrows():
            product_table_rows += f"""\
            <tr align='center'>\
            <td>{name}</td>\
            <td>{row['총손익']:,.0f}</td>\
            <td>{row['평균손익']:,.0f}</td>\
            <td>{row['표준편차']:,.0f}</td>\
            <td>{row['위험대비손익']:.2f}</td>\
            <td>{row['승률']*100:.2f} %</td>\
            <td>{row['보유기간']:.0f}</td>\
            <td>{ row['매매회수']:.0f}</td>\
            </tr>\
            """
        
        #4. 종목 결과 상세
        product_detail=""
        for symbol in self.symbols:
            fig, product_result = self.detail_result(symbol)
            if not fig:
                continue
            fig.tight_layout()
            fig.savefig(os.path.join(savedir,f'trade_chart_({symbol}).svg')) #trade_chart
            plt.close()
            product_detail += f"""
            <div align="center">
                    <img src="trade_chart_({symbol}).svg" width="1000px;" style="margin-left:50px; padding:0">
                    <div align="center">
                        <table border="1" width="800px" style="border-collapse:collapse" >
                            <tr align="center">
                                <th width="14.3%">총손익</th>
                                <th width="14.3%">총손익(틱)</th>
                                <th width="14.3%">평균손익(틱)</th>
                                <th width="14.3%">위험대비손익</th>
                                <th width="14.3%">승률</th>
                                <th width="14.3%">보유기간</th>
                                <th width="14.3%">매매횟수</th>
                            </tr>
                            <tr align="center">
                                <td width="14.3%">{product_result['총손익']:,.0f}</td>
                                <td width="14.3%">{product_result['총손익(틱)']:,.0f}</td>
                                <td width="14.3%">{product_result['평균손익(틱)']:,.1f}</td>
                                <td width="14.3%">{product_result['위험대비손익']:,.0f}</td>
                                <td width="14.3%">{product_result['승률']:.2f} %</td>
                                <td width="14.3%">{product_result['보유기간']:.0f}</td>
                                <td width="14.3%">{product_result['매매회수']:.0f}</td>
                            </tr>
                        </table>
                    </div>
            </div><br><br><br>
            """
            pd.DataFrame(self.trades.log(symbol=symbol)).to_csv(os.path.join(savedir,f'trade_history_({symbol}).csv'))

        #템플릿파일 로드
        with open('report_template.html', encoding='utf-8') as f:
            template = f.read()


        # 최종 html 파일 작성
        abstract = self.abstract
        report = template.format(
            today = datetime.today().strftime('%Y년 %m월 %d일'),
            system_name = abstract['name'],
            system_description = abstract['description'],
            system_sector=abstract['sectors'],
            system_instruments=', '.join(abstract['instruments']),
            system_from_date=abstract['from_date'],
            system_to_date=abstract['to_date'],
            system_principal=abstract['principal'],
            system_heat_system=abstract['heat_system'],
            system_max_system_heat=abstract['max_system_heat'],
            system_max_sector_heat=abstract['max_sector_heat'],
            system_max_trade_heat=abstract['max_trade_heat'],
            system_max_lots=abstract['max_lots'],
            system_commission=abstract['commission'],
            system_skid= abstract['skid'],
            system_metrics='<br>'.join(['  ,  '.join(i) for i in abstract['metrics']]),
            system_entry_rule_long=abstract['entry_rule']['long'].replace('<','&lt').replace('>','&gt') if abstract['entry_rule']['long'] else None,
            system_entry_rule_short=abstract['entry_rule']['short'].replace('<','&lt').replace('>','&gt') if abstract['entry_rule']['short'] else None,
            system_exit_rule_long=abstract['exit_rule']['long'].replace('<','&lt').replace('>','&gt') if abstract['exit_rule']['long'] else None ,
            system_exit_rule_short=abstract['exit_rule']['short'].replace('<','&lt').replace('>','&gt') if abstract['exit_rule']['short'] else None,
            system_stop_rule_long=abstract['stop_rule']['long'],
            system_stop_rule_short=abstract['stop_rule']['short'],
            equity_chart='equity_chart.svg',
            principal=result['투자금'],
            capital=result['최종자산'],
            profit=result['총손익'],
            bliss=result['Bliss'],
            cagr=result['CAGR'],
            mdd=result['MDD'],
            ptr=result['손익비'],
            winrate=result['승률']*100,
            rtp=result['위험대비손익'],
            avg_profit=result['평균손익'],
            avg_win=result['평균수익'],
            avg_loss=result['평균손실'],
            sector_table_rows = sector_table_rows,
            product_table_rows=product_table_rows,
            product_detail = product_detail
        )

        with open(os.path.join(savedir, '00_report.html'), 'w') as f:
            f.write(report)
            
        # 상세 거래내역 생성 및 저장
        """
        history = []
        for trade in self.trades.book:
            history.append([
                trade.id,
                trade.name,
                trade.entrydate.strftime('%Y-%m-%d'),
                trade.symbol,
                trade.sector,
                trade.position,
                trade.entryprice,
                trade.entrylots,
                trade.entryrisk,
                trade.entryrisk_ticks,
                trade.currentprice,
                trade.stopprice,
                trade.risk,
                trade.lots,
                trade.flame,
                trade.exits[0]['exitdate'].strftime('%Y-%m-%d') if trade.exits else "",
                trade.exits[0]['exitprice'] if trade.exits else "",
                trade.exits[0]['exitlots'] if trade.exits else "",
                trade.exits[0]['profit'] if trade.exits else "",
                trade.exits[0]['profit_ticks'] if trade.exits else "",
                trade.exits[0]['duration'] if trade.exits else "",
                trade.exits[0]['result'] if trade.exits else "",
                trade.commission,
                trade.exittype,
                trade.on_fire,
                
            ])
        columns = ['id','name','entrydate','symbol','sector','position','entryprice','entrylots',\
                    'entryrisk','entryrisk_ticks','currentprice','stopprice','risk','lots','flame','exitdate',\
                    'exitprice','exitlots','profit','profit_ticks','duration','result','commission','exittype','on_fire']
        df = pd.DataFrame(columns=columns, data=history)
        """
        pd.DataFrame(self.trades.log()).to_csv(os.path.join(savedir,f'trade_history.csv'))

        # 자산 내역 생성 및 저장
        pd.DataFrame(self.equity.log()).to_csv(os.path.join(savedir,f'equity_history.csv'))




       
   



        
