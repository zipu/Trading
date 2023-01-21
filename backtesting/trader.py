"""
메인 프로그램
"""
from multiprocessing.sharedctypes import Value
import numpy as np

from .system import System
from tools.instruments import instruments
from tools.quotes import Quotes

long = LONG = L = 1
short = SHORT = S = -1

class Trader:
    """
     여러 시스템을 테스트하고 분석하는 트레이더 오브젝트
    """
    
    #commission = 3.5 #편도 수수료
    #sectors = ['Currency','Grain','Meat','Tropical','Petroleum','Equity','Rate']
    
    def __init__(self, quotes_style, systems=[]):
        """
        symbols: 거래에 사용될 상품 목록
        systems: 시스템 목록
        """
        #srf 데이터가 있는 상품목록 호출
        self.instruments = [ins for ins in instruments.values() if ins.srf ]
        self.quotes = instruments.quotes(
                            symbols=[ins.symbol for ins in self.instruments],
                            method=quotes_style
                        )
        self.quotes_style = quotes_style
        #시스템 등록
        self.systems = []
        self.add_systems(systems)

    
    def run(self):
        """
        매매 진행
        """
        from time import time
        
        print("매매시작")
        dates = self.quotes.index
        systems = self.systems.copy()
        
        time0=time()
        for date in dates[1:]:
            quote = self.quotes.loc[date]

            for system in systems:
                if date < system.from_date or date > system.to_date:
                    #시스템 설정 날짜범위 밖이면 패스
                    continue
                else:
                    print(f"거래일: {date}, 시스템: {system.name} {time()-time0}sec" )
                    time0 = time()
                    # 자산이 음수가 되면 system.trade 함수가 True 값을 리턴하고 거래를 종료함
                    if system.trade(quote[system.symbols]):
                        print(f"###### 시스템 가동 종료: {system.name} #######")
                        systems.pop(systems.index(system))

                    
                 
                    

        return


    def add_systems(self, systems):
        #systems 값이 리스트가 아니라 한 시스템인경우
        if not isinstance(systems, list):
            systems = [systems]
            
        # 사용자 제공 정보 검사
        for system in systems:
            # 연결 정보 다르면 에러
            if system['quotes_style'] != self.quotes_style:
                msg = "\nQuotes_style does NOT match.\n"\
                      f"trader's quotes_style: {self.quotes_style}\n"\
                      f"system <<{system['name']}>>'s quotes_style: {system['quotes_style']}"
                raise ValueError(msg)

            #상품 목록 없으면 srf and ebest 목록으로 테스트 진행
            if not system['instruments']:
                system['instruments'] = instruments.get_symbols('srf', 'ebest')
        
        
        for id, system in enumerate(systems):
            quotes = self.quotes[system['instruments']]
            #print(quotes)
            self.systems.append(
                    System(system, Quotes(quotes, type='multiple'), id)
                )

    