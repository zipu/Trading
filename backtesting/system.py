"""
사용자가 등록한 시스템을 오브젝트화 함
"""
from multiprocessing import log_to_stderr
from os import stat

from collections import defaultdict
import re
import pandas as pd

from tools.instruments import instruments
from .book import Trades, Status, DefaultHeat

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
        self.name = abstract['name']
        self.description = abstract['description']
        self.symbols = abstract['instruments'] #매매상품 코드 목록
        #if not self.symbols: #코드 목록이 없으면 srf 전체 목록으로 매매 진행
        #    self.symbols = instruments.get_symbols('srf')
        
       #self.instruments = [instruments[symbol] for symbol in self.symbols]
        
        #섹터정보
        self.sectors = defaultdict(list)
        if abstract['sectors'] == 'pre-defined':
            for symbol in self.symbols:
                self.sectors[instruments[symbol].sector].append(symbol)
        
        
        #매매환경
        self.commission = abstract['commission']
        self.skid = abstract['skid']

        #자본금
        self.principal = abstract['principal'] 
        
        #한도 설정
        # 시스템 허용 위험한도 : 전체 자산 대비 
        self.heat = eval(abstract['heat'])( 
            abstract['max_system_heat'],
            abstract['max_sector_heat'],
            abstract['max_trade_heat'],
            abstract['max_lots']
        )
        
        #재무 정보 
        self.status = Status(self.principal)

        # 매매 내역서 
        self.trades = Trades(self.id)
        
        #self.capital = self.principal #현재 총 자산 = 현금 + 평가자산
        #self.cash = self.principal #현재 현금자산
        #self.security = 0 #현재 증권 자산 = 증거금 + 평가손익
        #self.fixed_capital = self.principal #총 자산 - 리스크 (손절 기준 총 자산)
        
        #과열 상태
        #self.system_heat = 0 #시스템 과열 지표 (risk/capital)
        #self.sector_heat = {sector: 0 for sector in self.sectors.keys()} #섹터당 리스크
        #self.risk = 0 #전체 리스크

        #지표 설정
        self.metrics = pd.DataFrame() #지표 생성
        self.metrics.attrs['name'] = self.name
        self.create_metrics(abstract['metrics'], quotes)

        #매매 시그널 생성
        self.signals = pd.DataFrame() #시그널 생성
        self.signals.attrs['name'] = self.name

        #생성된 인디케이터와 일봉데이터 결합
        quotes = pd.concat([quotes, self.metrics], axis=1) 

        self.entry_rule = abstract['entry_rule']
        self.create_signal(self.entry_rule['long'], 'enter_long', quotes)
        self.create_signal(self.entry_rule['short'], 'enter_short', quotes)
        
        self.exit_rule = abstract['exit_rule']
        self.create_signal(self.exit_rule['long'], 'exit_long', quotes)
        self.create_signal(self.exit_rule['short'], 'exit_short', quotes)


        self.stop_rule = abstract['stop_rule']
        self.create_stops(self.stop_rule['long'], 'stop_long', quotes)
        self.create_stops(self.stop_rule['short'], 'stop_short', quotes)

        # plotting 용도로 지표들의 종류를 세팅 (별도 plotting이 필요한지 용도)
        self.set_indicator_types()


        
    def __repr__(self):
        return f"<<시스템: {self.name}>>"

    
    def trade(self, yesterday, quote):
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
        fires = self.trades.get_on_fires() #진행중인 매매
        signals = self.signals.loc[yesterday]
        

        #1. 진행중인 매매 청산
        for fire in fires:
            if fire.position == LONG and signals['exit_long']:
                self.exit(fire, quote[fire.symbol], 'exit')

            elif fire.position == SHORT and signals['stop_short']:
                self.exit(fire, quote[fire.symbol], 'exit')

            self.status.update(today, self.trades)


        #2. 진입
        for symbol in self.symbols:
            if signals[symbol]['enter_long'] == True:
                self.enter(symbol, quote[symbol], LONG)
            
            if signals[symbol]['enter_short'] == True:
                self.enter(symbol, quote[symbol], SHORT)


        #3. STOP 가격 및 자산 업데이트
                
    def enter(self, symbol, quote, position):
        """
        매매 진입
        """
        today = quote.name
        order = {
            'entrydate': today,
            'symbol': symbol,
            'position': position,
            'entryprice': self.price(quote, position, 'enter')
        }

        #계약수 결정
        lots = self.lots_calculator(order)
        if lots == 0:
            self.book.reject(order)
            return
        
        order['entrylots'] = lots
        if position == LONG:
            order['stopprice'] = self.signals.loc[today][symbol]['stop_long']
        elif position == SHORT:
            order['stopprice'] = self.signals.loc[today][symbol]['stop_short']

        risk, risk_ticks = self.risk_calculator(symbol, order['entryprice'], order['stopprice'], lots)
        order['risk'] = risk
        order['risk_ticks'] = risk_ticks


        self.book.add_entry(order)
        self.update_status()
    
    def risk_calculator(self, symbol, entryprice, stopprice, lots):
        tickunit = instruments[symbol].tickunit
        tickvalue = instruments[symbol].tickvalue
        risk_ticks = abs(entryprice-stopprice)/tickunit
        risk = risk_ticks * tickvalue * lots

        return risk, risk_ticks

    def lots_calculator(self):
        """ 
        진입 계약수 결정:
        1) 가능한 heat 이 있는지 확인
        2) 여분의 heat 범위에서 max_lots 이내로 결정 

        %섹터 heat 계산은 나중에 업데이트
        """
        heat  =  self.heat()

    def calc_profit(self, fire, exitprice, lots):
        instrument = instruments[fire.symbol]
        tickunit = instrument.tickunit
        tickvalue = instrument.tickvalue
        profit_tick = (fire.entryprice - exitprice)*fire.position/tickunit
        profit = profit_tick * tickvalue * lots
        profit = round(profit, 2)
        profit_ticks = round(profit_ticks)
        return profit, profit_ticks

        
    def exit(self, fire, quote):
        #분할 매도 코드 추후 필요시 삽입
        #lots = self.lots_calculatr 
        
        exitdate = quote.name
        exitprice = self.price(quote, fire['position'], 'exit')
        #####################################
        #추후 분할 청산 시스템 사용시 변경 필요#
        #####################################
        exitlots = fire['lots']

        profit, profit_ticks = self.calc_profit(fire, exitprice, exitlots)
        if profit > 0:
            result = 'WIN'
        else:
            result = 'LOSE'

        self.trades.add_exit(fire, exitdate, exitprice, exitlots, profit, profit_ticks, result)

                
    def update_stopprice(self, fire):
        """
        거래중인 종목의 스탑 가격 갱신
        """
        

    
    def price(self, quote, position, type):
        """ 
        매매가격결정 
         position: LONG or SHORT
         type: enter or exit
            1)'buy' : open + (high - open) * skid
            2)'sell': open - (open - low) * skid 
        """
        if position==LONG and type == 'enter':
            return quote['open'] + (quote['high'] - quote['open'])*self.skid

        elif position == LONG and type == 'exit':
            return quote['open'] - (quote['open'] - quote['low'])*self.skid

        elif position == SHORT and type == 'enter':
            return quote['open'] - (quote['open'] - quote['low'])*self.skid

        elif position == SHORT and type == 'exit':
            return quote['open'] + (quote['high'] - quote['open'])*self.skid
        
        
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
        사용자 정의된 규칙에 따른 스탑 가격 결정
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


    def set_indicator_types(self):
        """
        quotes의 column 별 유형을 dataframe 의 attr에 저장
        plotting 용도로 사용
        """
        indicator_types={
            'price': ['EMA','MA','MAX','MIN'],
            'index': ['ATR'],
            'signal': ['enter_long','exit_long','enter_short','exit_short','stop_long','stop_short']
        }
        numeric_type = {}

        for indicator in self.abstract['metrics']:
            numeric_type[indicator[0]] = [k for k,v in indicator_types.items() if indicator[1] in v][0]

        self.metrics.attrs['type'] = numeric_type



        
