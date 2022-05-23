"""
사용자가 등록한 시스템을 오브젝트화 함
"""
#from multiprocessing import log_to_stderr
#from os import stat

from datetime import datetime
from collections import defaultdict
import re
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

from tools.instruments import instruments
from .book import TradesBook, EquityBook, DefaultHeat

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
        self.symbols = abstract['instruments'] #매매상품 코드 목록
        if not self.symbols: #코드 목록이 없으면 srf 전체 목록으로 매매 진행
            self.symbols = instruments.get_symbols('srf')
        
       #self.instruments = [instruments[symbol] for symbol in self.symbols]
        
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
        
        #한도 설정
        # 시스템 허용 위험한도 : 전체 자산 대비 
        self.heat = eval(abstract['heat_system'])( 
            abstract['max_system_heat'],
            abstract['max_sector_heat'],
            abstract['max_trade_heat'],
            abstract['max_lots']
        )
        
        #재무 정보 
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
        self.create_signal(self.entry_rule['long'], 'enter_long', quotes)
        self.create_signal(self.entry_rule['short'], 'enter_short', quotes)
        
        self.exit_rule = abstract['exit_rule']
        self.create_signal(self.exit_rule['long'], 'exit_long', quotes)
        self.create_signal(self.exit_rule['short'], 'exit_short', quotes)


        self.stop_rule = abstract['stop_rule']
        self.create_stops(self.stop_rule['long'], 'stop_long', quotes)
        self.create_stops(self.stop_rule['short'], 'stop_short', quotes)

        # 특정 날짜에 ohlc 값은 없는데(NaN) 지표와 시그널은 Averaging 되므로 있을 수 있음
        # 이를 NaN값으로 변경해 줌

        self.set_nans(self.metrics, quotes)
        self.set_nans(self.signals, quotes)

        self.signals.index = self.signals.index.astype('M8[ns]')



    def __repr__(self):
        return f"<<시스템: {self.name}>>"

    
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
          이 경우 해당 상품의 당일 시그널이 전일 시그널이 같도록 변경하고 거래 상품 목록에서 제외
        """
        symbols = []
        for symbol in self.symbols:
            ohlc = quote[symbol][['open','high','low','close']]

            if ohlc.isna().any():
                self.signals.loc[today, symbol] = self.signals.loc[yesterday, symbol].values
            else:
                symbols.append(symbol)

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
                if self.equity.update(today, self.trades, self.heat):
                  return True #시스템 가동 중단

            elif fire.position == SHORT and signals[symbol]['exit_short']:
                self.exit(fire, quote[symbol], type='exit')
                #자산 상태 업데이트
                if self.equity.update(today, self.trades, self.heat):
                    return True #시스템 가동 중단
        
        
        """
        3. 매매 진입

          전일 진입 신호 발생시, 시초가(+skid) 진입
        """
        for symbol in symbols:
            if signals[symbol]['enter_long'] == True:
                #stopprice = signals_today[symbol]['stop_long']
                self.enter(symbol, quote[symbol], LONG)
                if self.equity.update(today, self.trades, self.heat):
                    return True #시스템 가동 중단
                
            elif signals[symbol]['enter_short'] == True:
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
                    stopprice = self.signals.loc[today][symbol]['stop_long']
                elif fire.position == SHORT:
                    stopprice = self.signals.loc[today][symbol]['stop_short']

                fire.stopprice == stopprice

        if self.equity.update(today, self.trades, self.heat):
            return True #시스템 가동 중단

    
                
    def enter(self, symbol, quote, position):
        """
        매매 진입
        """
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
            symbol, position, order['entryprice'], order['stopprice'], 1)
        
        #order['entryrisk'] = risk
        order['entryrisk_ticks'] = risk_ticks
        
        #######################################################
        #  리스크가 음수인 것은 시스템 전략의 문제임. 나중에 수정필요 #
        #####################################################
        if risk_ticks <= 0 :
            self.trades.reject(symbol, today, sector, position, order['entryprice'])
            return
            #raise ValueError(f"리스크가 음수 또는 0일 수 없음: {order}")

        #계약수 결정
        lots = self.heat.calc_lots(symbol, sector, risk_ticks, self.equity)
        if lots == 0:
            self.trades.reject(symbol, today, sector, position, order['entryprice'])
            return
        
        order['entrylots'] = lots
        order['entryrisk'] = risk_trade * lots
        order['commission'] = self.commission * lots
        
        self.trades.add_entry(**order)
        #self.update_status()
    
    def lots_calculator(self):
        """ 
        진입 계약수 결정:
        1) 가능한 heat 이 있는지 확인
        2) 여분의 heat 범위에서 max_lots 이내로 결정 

        %섹터 heat 계산은 나중에 업데이트
        """
        heat  =  self.heat()

    def price_to_value(self, symbol, position, initial_price, final_price, lots):
        """
        상품 가격 차이로부터 가치를 계산
        """
        tickunit = instruments[symbol].tickunit
        tickvalue = instruments[symbol].tickvalue
        value_ticks = round(position*(initial_price-final_price)/tickunit)
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

    def set_nans(self, df, quotes):
        """
        ohlc 데이타가 nan 인데도, averaging 과정에서 시그널이 있을 수 있음.
        이를 nan으로 바꾸는 함수
        df: self.signals 또는 self.metrics
        """
        for symbol in self.symbols:
            flag = quotes[symbol][['open','high','low','close']].isna().any(axis=1)
            df.loc[flag==True, symbol] = np.nan

    
    def equity_plot(self):
        equitylog = pd.DataFrame(self.equity.log()).set_index('date')
        equity = equitylog.groupby(by='date').last()
        x = equity.index.values
        capital = equity.capital.values
        fixed_capital = equity.fixed_capital.values
        principal = (capital - equity.profit).values #매매직전원금
        max_capital = equity.max_capital.values
        p = self.principal #투자원금

        fig, ax = plt.subplots(1,1, figsize=(15, 8))
        ax.fill_between(x, p, fixed_capital, where=fixed_capital>=p, facecolor='green', alpha=0.4, interpolate=True, label='fixed_capital')
        ax.fill_between(x, p, fixed_capital, where=fixed_capital<p, facecolor='red', alpha=0.6, interpolate=True)
        ax.fill_between(x, capital, max_capital, color='grey', alpha=0.2)

        ax.plot(x, principal, color='orange',alpha=0.7, linewidth=1, label='capital')
        ax.plot(x, capital, color='black',alpha=0.7, linewidth=1)

        ax.set_xlim([x.min(), x.max()])

        #reference curve
        rate = 0.1 #annual interest rate
        refx = (x-x[0])/np.timedelta64(365,'D')
        refy = p*np.exp(rate*refx)
        ax.plot(x, refy, color='magenta', linestyle='--', label='reference')

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
                
        plt.show()

    def summary(self, level=0):
        if level == 0:
            self.equity_plot()
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
                '총손익': table.profit.sum(),
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
                    '평균손익': "{:.2f}",
                    '표준편차': "{:.2f}",
                    '위험대비손익': "{:.2%}",
                    '승률': "{:.2%}",
                    #'# trades': "{:.f}"
                })
        return df

    

       
   



        
