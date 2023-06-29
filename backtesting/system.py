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
from tools.quotes import Quotes

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
        
        #매매환경
        self.commission = abstract['commission']
        self.skid = abstract['skid']
        self.allow_pyramiding = abstract['allow_pyramiding']


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

        #주문 바구니
        self.orderbag = {'enter':[], 'exit':[]}
        
        #재무 상태 내역서 
        self.equity = EquityBook(self.principal, self.from_date)

        # 매매 내역서 
        self.trades = TradesBook(self.id)
        
        # 시장 데이터 
        quotes = Quotes(quotes[self.symbols])


        #지표 생성
        #self.metrics = pd.DataFrame() #지표 생성
        #self.metrics.attrs['name'] = self.name
        self.metrics = self.create_metrics(abstract['metrics'], quotes)
        
        #생성된 지표 일봉데이터 결합
        df = pd.concat([quotes, self.metrics], axis=1) 

        #매매 시그널 생성
        self.signals = pd.DataFrame() #시그널 생성
        self.signals.attrs['name'] = self.name

        self.entry_rule = abstract['entry_rule']
        if self.entry_rule['long']:
            self.create_signal(self.entry_rule['long'], 'enter_long', df)
        
        if self.entry_rule['short']:
            self.create_signal(self.entry_rule['short'], 'enter_short', df)
        
        self.exit_rule = abstract['exit_rule']
        if self.exit_rule['long']:
            self.create_signal(self.exit_rule['long'], 'exit_long', df)
        if self.exit_rule['short']:
            self.create_signal(self.exit_rule['short'], 'exit_short', df)


        self.stop_rule = abstract['stop_rule']
        if self.stop_rule['long']:
            self.create_stops(self.stop_rule['long'], 'stop_long', df)
        if self.stop_rule['short']:
            self.create_stops(self.stop_rule['short'], 'stop_short', df)

        self.signals.index = self.signals.index.astype('M8[ns]')
        

        self.quotes = quotes.loc[self.from_date:self.to_date]
        self.metrics = self.metrics.loc[self.from_date:self.to_date]
        self.signals = self.signals.loc[self.from_date:self.to_date]
        # OHLC 데이터 없는날, index type등 처리
        #self.compensate_signals(quotes)

        #self.set_nans(self.metrics, quotes)
        #self.set_nans(self.signals, quotes)


    def __repr__(self):
        return f"<<시스템: {self.name}>>"



    def create_metrics(self, metrics, quotes):
        """
        configurator 파일에 등록된 지표들을 pd Dataframe으로 생성
        - metrics: 지표 데이터
        - qutoes: 시장 데이터

        차트 생성시 OHLC 차트위에 지표를 그릴지, 추가 axis를 만들어서 생성할지 구분하기 위해
        각 metric은 2가지 속성값 중에 하나를 갖음
        1. price: 가격 기반 지표. ohlc위에 그림
        2. index: 별도의 axis 에 그림
        지표별 속성은 qutoes의 attrs['metric_types'] 에서 확인가능
        """

        # 이 시스템에서 사용할 상품목록에 한해서 지표 생성
        #quotes = Quotes(quotes[self.symbols]) 

        # 설정 파일에 [필드이름, 함수이름, 파라메터] 형식으로 지표가 저장되어 있음
        # ex) [ema200, EMA, window=200]
        # EMA는 quote 모듈에 등록된 함수로 지수이평을 파라메터에 맞게 생성해줌 
        lists = []
        for indicator in metrics:
            fieldname = indicator[0]
            ind = indicator[1]
            params = ','.join(indicator[2:])
            params = eval(f'dict({params})')
            series = getattr(quotes, ind)(fieldname=fieldname, **params)
            lists.append(series)

        df = pd.concat(lists, axis=1)
        df = df.sort_index(axis=1, level=0, sort_remaining=False)

        # plotting시 추가 axes 생성여부를 판단하기 위해 각 metric의 형식을 attribute로 저장
        metric_types = quotes.attrs['metric_types']
        numeric_type = {}
        for metric in self.abstract['metrics']:
            numeric_type[metric[0]] = [k for k,v in metric_types.items() if metric[1] in v][0]
        df.attrs['type'] = numeric_type

        # OHLC 데이터가 없는 거래일에 metric 값은 Averaging 결과로써 존재할 수 있음. 
        # OHLC 데이터 Nan 인 날은 거래를 하지 않기 때문에 시그널도 NaN 값으로 변경 해줌
        for symbol in self.symbols:
            flag = quotes[symbol][['open','high','low','close']].isna().any(axis=1)
            df.loc[flag, symbol] = np.nan

        df.attrs['name'] = self.name
        return df



    def create_signal(self, rules, name, quotes):
        """
        사용자 정의된 매수, 매매 규칙에 해당되는 날에 시그널(True or False) 생성
        - rules: 매매규칙
        - name: 시그널 이름
        - quotes: 시장 일봉데이터
        """
        signals = []
        for symbol in self.symbols:
            condition = rules
            df = quotes[symbol]
            #fields = df.columns.to_list()
            
            # ex) 조건이 ma5>ma20 인경우
            #     df['ma5']>df['ma20'] 의 pandas 조건식으로 변환
            #for field in fields:
            #    condition = re.sub(f"(?<![A-Za-z0-9])({field})(?![A-Za-z0-9])", f"df['{field}']", condition)

            #self.quotes[symbol,'buy_signal'] = eval(rule)
            #signal = eval(condition)
            signal = pd.Series(data=False, index=df.index)
            signal.loc[df.query(rules).index] = True
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

    def trade(self, today):

        quote = self.quotes.loc[today]
        signals = self.signals.loc[today] #오늘자 시그널
        mask = quote.isna().groupby('symbol').all()
        tradables = list(mask[~mask].index) #오늘 거래가능 상품 목록


        orderbag = self.orderbag.copy()
        
        # 1. 기존 매매(fire) 청산
        if orderbag['exit']:
            for fire in orderbag['exit']:
                if fire.symbol in tradables:
                    self.exit(fire, quote[fire.symbol], type='EXIT')
                    self.orderbag['exit'].remove(fire)

            #자산 상태 업데이트
            self.equity.update(today, self.trades, self.heat)

        
        # 2. 신규 매매 진입
        if orderbag['enter']:
            for item in orderbag['enter']:
                if item['symbol'] in tradables: 
                    self.enter(item['symbol'], quote[item['symbol']], item['position'])
                    self.orderbag['enter'].remove(item)

            self.equity.update(today, self.trades, self.heat)


        # 3. 장중 스탑 청산 및 장 종료 후 스탑 가격 및 상태 업데이트
        fires = self.trades.get_on_fires()
        fires = [fire for fire in fires if fire.symbol in tradables]
        for fire in fires:
            today_low = quote[fire.symbol]['low']
            today_high = quote[fire.symbol]['high']
            today_close = quote[fire.symbol]['close']

            if fire.position == LONG and fire.stopprice >= today_low:
                self.trades.add_exit(fire, today, fire.stopprice, fire.lots, 'STOP')

            elif fire.position == SHORT and fire.stopprice <= today_high:
                self.trades.add_exit(fire, today, fire.stopprice, fire.lots, 'STOP')

            else:
                if fire.position == LONG:
                    stopprice = signals[fire.symbol]['stop_long']
                elif fire.position == SHORT:
                    stopprice = signals[fire.symbol]['stop_short']

                fire.stopprice == stopprice
                fire.update_status(today_close, stopprice)
        
        if fires:
            self.equity.update(today, self.trades, self.heat)

        # 4. 장 종료 후 내일 매매 목록 바구니에 담기
        for symbol in tradables:
            fire = self.trades.get_on_fires(symbol=symbol)
            signal = signals[symbol]
            
            if signal.get('enter_long'):
                # 해당 상품이 이미 매매중이고 피라미딩을 허용안하는 시스템이면 건너뜀
                if fire and not self.allow_pyramiding:
                    pass
                else: 
                    self.orderbag['enter'].append({
                        'symbol': symbol,
                        'position': LONG
                    })
  
            if signal.get('enter_short'):
                # 해당 상품이 이미 매매중이고 피라미딩을 허용안하는 시스템이면 건너뜀
                if fire and not self.allow_pyramiding:
                    pass
                else: 
                    self.orderbag['enter'].append({
                        'symbol': symbol,
                        'position': SHORT
                    })

            if fire and (signal.get('exit_long') or signal.get('exit_long')):
                self.orderbag['exit'].append(fire[0])

        
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
        
        if risk_ticks <= 0 :
            raise ValueError(f"리스크가 음수 또는 0일 수 없음: {order}")
        
        #계약수 결정
        lots = self.heat.calc_lots(symbol, sector, risk_ticks, self.equity)
        if lots == 0:
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

    
    
    def exit(self, fire, quote, type='EXIT'):
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
            1)'buy' : open + skid
            2)'sell': open - skid 
        """
        skid = self.skid * instruments[symbol].tickunit

        if (position==LONG and type == 'enter') or\
            (position==SHORT and type == 'exit'):
            return quote['open'] + skid

        elif (position==LONG and type == 'exit') or\
            (position==SHORT and type == 'enter'):
            return quote['open'] - skid
        
        
    def performance(self):
        """
        시스템 종합 성능 보고 
        """

        #1. 차트 데이터 
        equity  = pd.DataFrame(self.equity.log())\
                    .set_index('date')\
                    .groupby(by='date').last()
        rate = 0.1 #interest rate
        equity['date'] = equity.index.values.astype('int64')/1000000
        equity['cash'] = equity.capital - equity.flame
        equity['reference'] = equity.principal*np.exp(rate*((equity.index - equity.index[0])/np.timedelta64(365,'D')))
        
        total_value = equity[['date','capital']].round().values.tolist()
        defined_value_p = equity[equity['fixed_capital']>=equity.principal][['date','fixed_capital']].round().values.tolist()
        defined_value_n = equity[equity['fixed_capital']<equity.principal][['date','fixed_capital']].round().values.tolist()
        max_value = equity[['date','max_capital']].round().values.tolist()
        cash = equity[['date','cash']].round().values.tolist()
        reference = equity[['date','reference']].values.tolist()

        chart_data = {
            'total_value' :  total_value,
            'defined_value_p': defined_value_p,
            'defined_value_n': defined_value_n,
            'max_value': max_value,
            'cash': cash,
            'reference': reference,
        }

        #2. 성능 데이터 
        win_trades = self.trades.get(result='WIN')
        lose_trades = self.trades.get(result='LOSE')
        trades = win_trades + lose_trades
        cnt = len(trades)

        performance = {
            'capital': self.equity.capital,
            'profit': self.equity.profit,
            'profit_rate': (self.equity.capital / self.principal),
            'bliss': self.equity.cagr/self.equity.mdd if self.equity.mdd > 0 else '',
            'cagr': self.equity.cagr,
            'mdd': self.equity.mdd,
            #손익비
            'avg_ptl': sum([t.profit for t in win_trades])/sum([t.profit for t in lose_trades]) if lose_trades else 'inf', 
            'profit_to_risk': sum([t.profit/t.entryrisk for t in trades])/cnt if cnt else '',
            'winrate': len(win_trades)/cnt if cnt else 0,
            'avg_profit': sum([t.profit for t in trades])/cnt if cnt else '',
            'avg_win': sum([t.profit for t in win_trades])/len(win_trades) if win_trades else 0,
            'avg_lose': sum([t.profit for t in lose_trades])/len(lose_trades) if lose_trades else 0,
            'duration': sum(t.duration for t in trades)/cnt if cnt else '',
            'num_trades': cnt,
            'chartdata': chart_data       
        }
        return performance
    
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
            '손익비': sum([t.profit for t in win_trades])/sum([t.profit for t in lose_trades]) if lose_trades else 'inf',
            '승률': len(win_trades) / cnt if cnt else '',
            '위험대비손익': sum([t.profit/t.entryrisk for t in trades])/cnt if cnt else '',
            '평균손익': sum([t.profit for t in trades])/cnt if cnt else '',
            '평균수익': sum([t.profit for t in win_trades])/len(win_trades) if win_trades else 0,
            '평균손실': sum([t.profit for t in lose_trades])/len(lose_trades) if lose_trades else 0,
            #'손익표준편차': trade.profit.std(),
            '보유기간': sum(t.duration for t in trades)/cnt if cnt else '',
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
                    '평균손실': "{:,.0f}",
                    '보유기간': "{:,.0f} 일",
                    #'# trades': "{:.1f}"
                })
        return report

    def trade_result(self, level='name'):
        result = []
        tradelog = pd.DataFrame(self.trades.log())
        for lev, table in tradelog.groupby(level):
            
            if level=='name':
                symbol = table['symbol'].iloc[0]
            elif level=='sector':
                symbol = table['sector'].iloc[0] 
            trade = {
                #'symbol': symbol,
                'name': lev,#self.pinfo[symbol]['name'],
                'symbol': symbol,
                'profit': table.profit.sum()+table.flame.sum(),
                'avg_profit': table.profit.mean(),
                'profit_to_risk': 100*(table.profit/table.entryrisk).mean(),
                'winrate': 100*len(table[table.profit > 0])/len(table),
                'duration': table.duration.mean(),
                'num_trades': len(table)
            }
            result.append(trade)
            result.sort(
                key = lambda x: x['profit'],
                reverse = True
            )
        return result
    
    def sector_data(self, sector):
        """
        섹터별 누적수익을 highchart 로 나타내기 위한 함수
        """
        data = []
        df = pd.DataFrame(self.trades.log(sector=sector))
        df['cum_profit'] = df['profit'].cumsum()
        grouped = df.groupby('exitdate').last()
        grouped['date'] = grouped.index.values.astype('int64')/1000000
        
        return grouped[['date','cum_profit']].values.tolist()


    def product_data(self, symbol):
        """
        상품별 매매 결과를 Highchart 그래프로 그리는 
        html string을 리턴함
        """
        from functools import reduce
        
        name = instruments[symbol].name
        quote = self.quotes[symbol][['open','high','low','close']].dropna()
        quote.insert(0, 'date', quote.index.astype('int64')/1000000)
        trades = pd.DataFrame(self.trades.log(symbol=symbol))
        if len(trades) == 0:
            return {}
        
        # 차트 데이터 생성
        # 1. ohlc 데이터 수익 & 손실 매매 구분
        win_masks = []
        lose_masks =[]
        for idx, row in trades.iterrows():
            mask = (quote.index >= row['entrydate']) & (quote.index <= row['exitdate'])
            if row['result'] == 'WIN':
                win_masks.append(mask)
            elif row['result'] == 'LOSE':
                lose_masks.append(mask)

        win_mask = reduce(np.logical_or, win_masks) if win_masks else quote['date'] == -1
        lose_mask = reduce(np.logical_or, lose_masks) if lose_masks else quote['date'] == -1

        win_dates = quote[win_mask].index
        lose_dates = quote[lose_mask].index
        neutral_dates = quote[~win_mask & ~lose_mask].index
        wins = quote.loc[win_dates].values.tolist()
        loses = quote.loc[lose_dates].values.tolist()
        neutrals = quote.loc[neutral_dates].values.tolist()

        #2. 지표 데이터
        metrics = self.metrics[symbol].dropna()
        metrics.insert(0, 'date', metrics.index.astype('int64')/1000000)

        metric_type = {
            'price':[k for k, v in self.metrics.attrs['type'].items() if v=='price' ],
            'index':[k for k, v in self.metrics.attrs['type'].items() if v=='index' ]
        }

        #2-1. ohlc 차트와 함께 나타낼 지표
        metrics_with_ohlc = []
        for metric in metric_type['price']:
            metrics_with_ohlc.append({
                'name': metric,
                'data': metrics[['date',metric]].values.tolist()
            })
        
        #2-2 별도 axis에 나타낼 지표
        metrics_separated = []
        for metric in metric_type['index']:
            metrics_separated.append({
                'name': metric,
                'data': metrics[['date',metric]].values.tolist()
            })

        #3. 누적수익 곡선
        trades['date'] = trades['exitdate'].values.astype('int64')/1000000
        trades['cum_profit'] = trades['profit'].cumsum()
        trades.dropna(inplace=True)
        cum_profit = trades[['date','cum_profit']].values.tolist()
        win_profit = trades[trades['profit']>=0][['date','profit']].values.tolist()
        lose_profit =  trades[trades['profit']<0][['date','profit']].values.tolist()

        return {
            'wins': wins,
            'loses': loses,
            'neutrals': neutrals,
            'metrics_with_ohlc': metrics_with_ohlc,
            'metrics_separated': metrics_separated,
            'cum_profit': cum_profit,
            'win_profit': win_profit,
            'lose_profit':lose_profit
        }

    def create_report(self):
        """
        시스템 성능을 html파일 형식으로 작성 
        필요한 그래프는 highchart 모듈 이용
        """
        import shutil

        #폴더 생성
        foldername = self.name + '_' + datetime.today().strftime('%Y%m%d%H%M')
        savedir = os.path.join('report',foldername) 
        os.mkdir(savedir)

        data = {}

        #1. 시스템 명세
        system_info = {
            'today': datetime.today().strftime('%Y년 %m월 %d일'),
            'name': self.name,
            'description': self.description or '',
            'sector': self.abstract['sectors'] or '',
            'instruments': ', '.join(self.symbols) or '',
            'start': self.from_date.strftime('%Y-%m-%d'),
            'end': self.to_date.strftime('%Y-%m-%d') ,
            'principal': self.principal,
            'heat_system':self.abstract['heat_system'],
            'max_system_heat': self.abstract['max_system_heat'],
            'max_sector_heat':self.abstract['max_sector_heat'],
            'max_trade_heat': self.abstract['max_trade_heat'],
            'max_lots': self.abstract['max_lots'],
            'commission': self.abstract['commission'],
            'skid': self.abstract['skid'],
            'metrics': '\n'.join([str(metric) for metric in self.abstract['metrics']]),
            'entry_rule':{
                'long': self.abstract['entry_rule']['long'] or '',
                'short': self.abstract['entry_rule']['short'] or '',
            },
            'exit_rule': {
                'long': self.abstract['exit_rule']['long'] or '',
                'short': self.abstract['exit_rule']['short'] or '',
            },
            'stop_rule': {
                'long': self.abstract['stop_rule']['long'] or '',
                'short': self.abstract['stop_rule']['short'] or '',
            }
        }

        #2. 성능 데이터 
        performance = self.performance()


        #3. 섹터 결과 
        sector_result = self.trade_result(level='sector')

        #섹터별 상세기록
        sector_detail = []
        for sector in self.sectors:
            sector_detail.append({
                'sector': sector,
                'chartdata': self.sector_data(sector)
            })
 
        #4. 상품별 결과
        product_result = self.trade_result(level='name')

        #상품별 상세 기록
        for symbol in self.symbols:

            product_detail ={
                'symbol': symbol,
                'name': instruments[symbol].name,
                'chartdata': self.product_data(symbol)
            }
            #각각을 별도 파일로 저장
            with open(os.path.join(savedir, f'product_data({symbol}).txt'), 'w', encoding='utf8') as f:
                f.write(f"var product_data={product_detail}")
            
            #종목별 거래 기록 저장
            pd.DataFrame(self.trades.log(symbol=symbol)).to_csv(os.path.join(savedir,f'trade_history_({symbol}).csv'))

        
        
        #detail_result = self.trade_result(level='symbol')
        data ={
            'system_info': system_info,
            'performance': performance,
            'sector_result': sector_result,
            'product_result': product_result,
            'sector_detail': sector_detail
        }

        #파일 생성 및 저장
        with open(os.path.join(savedir, 'data.txt'), 'w', encoding='utf8') as f:
            f.write(f"var data={data}")

        #템플릿 파일 복사
        shutil.copy2('report_template.html', os.path.join(savedir,'0_report.html'))
        
        # 전체 거래 내역 생성 및 저장
        pd.DataFrame(self.trades.log()).to_csv(os.path.join(savedir,f'trade_history.csv'))

        # 자산 내역 생성 및 저장
        pd.DataFrame(self.equity.log()).to_csv(os.path.join(savedir,f'equity_history.csv'))