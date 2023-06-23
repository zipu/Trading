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
        self.instruments = instruments.db_symbols(db=db)
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
            #상품 목록 없으면 전체 상품목록으로 테스트 진행
            if not system['instruments']:
                system['instruments'] = self.instruments
        
        
        for id, system in enumerate(systems):
            quotes = Quotes(self.quotes[system['instruments']], type='multiple')
            #print(quotes)
            self.systems.append(
                    System(system, quotes, id)
                )

    
    def create_report(self):
        """ 결과 보고서 작성 """
        #폴더 생성
        foldername = self.name + '_' + datetime.today().strftime('%Y%m%d%H%M')
        os.makedirs(foldername)

        #1. 종합 결과
        #equity chart 이미지 파일 생성 및 저장 
        fig = self.equity_plot()
        fig.tight_layout()
        fig.savefig(os.path.join(foldername,'equity_chart.svg')) #equity_chart


        result = self.summary().data.to_dict()
        for k,v in result.items():
            result[k] = v['Result']

        
