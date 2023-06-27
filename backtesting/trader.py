"""
메인 프로그램
"""
from multiprocessing.sharedctypes import Value
import numpy as np

from .system import System
from tools.instruments import instruments
from tools.quotes import Quotes

from tools.constants import SRF_CONTINUOUS_BO_DB_PATH

long = LONG = L = 1
short = SHORT = S = -1

class Trader:
    """
     여러 시스템을 운용하고 그 결과를 종합하여 분석하는 총괄 오브젝트 
    """
    
    def __init__(self, systems=[], db=SRF_CONTINUOUS_BO_DB_PATH):
        """
        symbols: 거래에 사용될 상품 목록
        systems: 시스템 목록
        """
        #데이터베이스에 있는 종목 리스트 호출
        self.instruments = []
        for symbol in instruments.db_symbols(db=db):
            if instruments[symbol].tradable:
                self.instruments.append(symbol)

        
        self.quotes = instruments.quotes(
                            symbols=self.instruments,
                            db=db
                        )
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

            for system in systems.copy():
                if date < system.from_date or date > system.to_date:
                    #시스템 설정 날짜범위 밖이면 패스
                    continue
                else:
                    print(f"거래일: {date}, 시스템: {system.name}") #{time()-time0}sec" )
                    #time0 = time()
                    system.trade(quote[system.symbols])
                    if system.equity.capital < 0:
                        print(f"###### 시스템 가동 종료: {system.name} #######")
                        systems.remove(systems)
        print("매매종료")
        return


    def add_systems(self, systems):
        #systems 값이 리스트가 아니라 한 시스템인경우
        if not isinstance(systems, list):
            systems = [systems]
            
        for id, system in enumerate(systems):
            #quotes = Quotes(self.quotes[system['instruments']])
            #print(quotes)
            self.systems.append(
                    System(system, self.quotes, id)
                )