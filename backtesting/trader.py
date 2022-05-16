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
        print("매매시작")
        # 매매판단은 전날 데이터를 기준으로 하기때문에, 직전 거래일 날짜인덱스를 인수로 함
        dates = self.quotes.index
        for date in dates[1:]:
            
            yesterday = dates[dates.get_loc(date) - 1]
            quote = self.quotes.loc[date]

            for system in self.systems:
                print(f"거래일: {date}, 시스템: {system.name}")
                system.trade(yesterday, quote)
                
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

            #상품 목록 없으면 srf 목록으로 테스트 진행
            if not system['instruments']:
                system['instruments'] = instruments.get_symbols('srf')
        
        
        for id, system in enumerate(systems):
            quotes = self.quotes[system['instruments']]
            #print(quotes)
            self.systems.append(
                    System(system, Quotes(quotes, type='multiple'), id)
                )
        
            #인디케이터 생성
            #for indicator in system['indicators']:
            #    window = [x.split('=')[1] for x in indicator if 'window' in x][0]
            #    name = indicator[0]
            #    fieldname = f"{indicator[0]}{window}_{system['name']}"
            #    
            #    kwargs = ','.join(indicator[1:])
            #    kwargs = eval(f'dict({kwargs})')
        
            #    getattr(self.quotes, name)(**kwargs, inplace=True, fieldname=fieldname)
        
        
        #self.quotes = self.quotes.iloc[30:] #처음 한달은 데이터 성숙기간

    @classmethod
    def price_to_value(cls, inst, price):
        """
        상품가격(차이)를 그에 해당하는 화폐단위로 변화
        """
        return price * inst['tick_value'] / inst['tick_unit']
    
    @classmethod
    def get_profit(self, inst, position, entryprice, exitprice, lot=1):
        """
        틱: (청산가격 - 진입가격)/틱단위
        손익계산: 랏수 * 틱가치 * 틱      
        """
        if np.isnan(entryprice) or np.isnan(exitprice):
            raise ValueError('Nan value can not be calculated')
        
        tick = round(position * (exitprice - entryprice)/inst['tick_unit'])
        profit = lot * inst['tick_value']* tick
        
        return profit, tick
    
    @classmethod
    def get_price(cls, pinfo, price1, price2, skid):
        """
        진입 주문시 슬리피지 계산
        """
        bound = (price2 - price1)*skid
        #price = np.random.uniform(price1, price1 + bound)
        
        price = round(price1+bound, pinfo['decimal_places'])
        
        return price
    
    
    @classmethod
    def get_lot(cls, risk, heat):
        lot = int(heat / risk)
        return lot
    
    
   