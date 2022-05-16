"""
리스크 매니지 먼트
백태스팅 진행 시 다양한 종류의 리스크 매니지먼트 형태를 비교해 보기위해 모듈화 함
시장은 예측 불가능하기 때문에 수학적으로 최적화 할 수 없으나
자금 관리는 최적화가 (어느정도) 가능하기 때문에 트레이딩에 있어서 가장 중요한 팩터중 하나임 
"""
from .book import Book


class DefaultHeat:
    """ 
    디폴트 리스크 매니지먼트
    1. 현재 과열 상태 출력
    2. 매매 계약수 결정
    system heat = total risk / capital
    sector heat = total sector risk / capital
    trader heat = individual risk / capital
    """

    def __init__(self, max_system_heat, max_sector_heat, max_trade_heat, max_lots):
        self.max_system_heat = max_system_heat
        self.max_sector_heat = max_sector_heat
        self.max_trade_heat = max_trade_heat
        self.max_lots = max_lots
        
        self.book = Book(name='DafaultHeat')
        self.system_heat = 0
        self.sector_heat = {}


    def system_heat(self, asset, fires):
        return sum([fire['risk'] for fire in fires]) / asset.capital

    def sector_heat(self, asset, fires):
        sector_risk = {fire['sector']:0 for fire in fires}
        for fire in fires:
            sector_risk[fire['sector']] += fire['risk']
        
        sector_heat = {k:v/asset.capital for k,v in sector_risk.items()}
        return sector_risk, sector_heat

    def risk_calculator(self, asset, order):
        
        
        
            
            


