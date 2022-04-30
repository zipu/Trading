"""
메인 프로그램
"""
import numpy as np

from .system import System
from tools.instruments import instruments

long = LONG = L = 1
short = SHORT = S = -1

class Market:
    """
    시장 정보 제공
    """
    
    #commission = 3.5 #편도 수수료
    #sectors = ['Currency','Grain','Meat','Tropical','Petroleum','Equity','Rate']
    
    def __init__(self, quotes_style, systems):
        """
        symbols: 거래에 사용될 상품 목록
        systems: 시스템 목록
        """
        self.instruments = [ins for ins in instruments.values() if ins.srf ]
        self.quotes = instruments.quotes(
                            symbols=[ins.symbol for ins in self.instruments],
                            method=quotes_style
                        )
        #시스템 등록
        self.systems = []
        self.add_system(systems)
        
        #매매환경
        self.commission = system['commission']
        self.skid = system['skid']

        #인디케이터 생성
        for indicator in system['indicators']:
            name = indicator[0]
            kwargs = ','.join(indicator[1:])
            kwargs = eval(f'dict({kwargs})')
            getattr(self.quotes, name)(**kwargs, inplace=True)
        self.quotes = self.quotes.iloc[30:] #처음 한달은 데이터 성숙기간 

    
    def run(self):
        """
        프로그램 구동 함수
        """
        pass

    def add_system(self, systems):
        for system in systems:
            self.systems.append(System(system))

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
    
    
   