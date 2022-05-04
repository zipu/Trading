"""
사용자가 등록한 시스템을 오브젝트화 함
"""
import re
import pandas as pd

from tools.instruments import instruments

class System:

    def __init__(self, abstract, quotes):
        """
        abstract: 시스템 설정 내용
        qutoes: indicator 생성용
        """

        self.abstract = abstract
        self.name = abstract['name']
        self.description = abstract['description']
        self.symbols = abstract['instruments'] #매매상품 코드 목록
        #if not self.symbols: #코드 목록이 없으면 srf 전체 목록으로 매매 진행
        #    self.symbols = instruments.get_symbols('srf')
        
        self.instruments = [instruments[symbol] for symbol in self.symbols]
        #매매환경
        self.commission = abstract['commission']
        self.skid = abstract['skid']

        self.principal = abstract['principal'] #자본금
        
        #한도 설정
        self.heat_system = abstract['heat_system'] #시스템 허용 위험한도
        self.heat_sector = abstract['heat_sector'] #섹터 허용 위험한도
        self.heat_trade = abstract['heat_trade'] #매매당 허용 위험한도
        self.max_lots = abstract['max_lots'] #매매당 최대 계약수

        #지표 설정
        self.quotes = pd.DataFrame() #지표 및 시그널 저장
        self.quotes.attrs['name'] = self.name
        self.create_indicators(abstract['indicators'], quotes)

        #매매 시그널 생성
        self.entry_rule = abstract['entry_rule']
        self.create_signal(self.entry_rule['long'], 'enter_long', quotes)
        self.create_signal(self.entry_rule['short'], 'enter_short', quotes)
        
        self.exit_rule = abstract['exit_rule']
        self.create_signal(self.exit_rule['long'], 'exit_long', quotes)
        self.create_signal(self.exit_rule['short'], 'exit_short', quotes)


        self.stop_rule = abstract['stop_rule']
        self.create_signal(self.stop_rule['long'], 'stop_long', quotes)
        self.create_signal(self.stop_rule['short'], 'stop_short', quotes)
        

    def __repr__(self):
        return f"<<시스템: {self.name}>>"

    def create_indicators(self, indicators, quotes):
        """
        configurator 파일에 등록된 사용 인디케이터를 pd Dataframe으로 생성
        - indicators: 등록할 지표 저보
        - qutoes: 시장 일봉데이터
        """
        lists = []
        for indicator in indicators:
            #window = [x.split('=')[1] for x in indicator if 'window' in x][0]
            name = indicator[0]
            #fieldname = f"{indicator[0]}{window}_{system['name']}"
            
            kwargs = ','.join(indicator[1:])
            kwargs = eval(f'dict({kwargs})')
            series = getattr(quotes, name)(**kwargs)
            lists.append(series)

        df = pd.concat([self.quotes]+lists, axis=1)
        self.quotes = df.sort_index(axis=1, level=0, sort_remaining=False)

    def create_signal(self, rules, name, quotes):
        """
        사용자 정의된 매수, 매매 규칙에 해당되는 날에 시그널(True or False) 생성
        - rules: 매매규칙
        - name: 매매이름
        - quotes: 시장 일봉데이터
        """
        if not rules:
            return
        
        quotes = pd.concat([quotes, self.quotes], axis=1)
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
        self.quotes = pd.concat([self.quotes, signals], axis=1)
        #self.quotes[signals.columns] = signals
        self.quotes.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)


    def create_stop_signal(self, rules, name):
        """
        사용자 정의된 스탑 규칙에 해당되는 날에 시그널 생성
        """
        if not rules:
            return
        
