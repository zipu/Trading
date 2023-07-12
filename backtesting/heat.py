"""
리스크 매니지 먼트
백태스팅 진행 시 다양한 종류의 리스크 매니지먼트 형태를 비교해 보기위해 모듈화 함
시장은 예측 불가능하기 때문에 수학적으로 최적화 할 수 없으나
자금 관리는 최적화가 (어느정도) 가능하기 때문에 트레이딩에 있어서 가장 중요한 팩터중 하나임 
"""
from tools.instruments import instruments

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
        self.max_lots = max_lots #상품당 최대 계약수
        
    def system_heat(self, status):
        return  status.risk/status.capital

    def sector_heat(self, status, fires):
        sector_risk = {fire.sector:0 for fire in fires}
        for fire in fires:
            sector_risk[fire.sector] += fire.risk
        
        sector_heat = {k:v/status.capital for k,v in sector_risk.items()}
        return sector_heat, sector_risk

    def calc_lots(self, symbol, sector, risk_ticks, equity):
        # 1계약 기준 리스크
        risk = risk_ticks*instruments[symbol].tickvalue
        
        # 바운더리
        capital = equity.capital
        
        max_system_risk = capital * self.max_system_heat
        max_sector_risk = capital * self.max_sector_heat
        max_trade_risk = capital * self.max_trade_heat
        
        #현재 섹터 리스크
        if equity.sector_risk.get(sector):
            sector_risk = equity.sector_risk[sector]
        else:
            sector_risk = 0
        
        if risk + equity.risk > max_system_risk:
            return 0
        else:
            lots_system = int((max_system_risk - equity.risk)/risk) 

        if risk + sector_risk > max_sector_risk:
            return 0
        else:
            lots_sector = int((max_sector_risk - sector_risk)/risk)

        if risk > max_trade_risk:
            return 0
        else:
            lots_trade = int(max_trade_risk/risk)

        if self.max_lots:
            return min([lots_system, lots_sector, lots_trade, self.max_lots])
        else: 
            return min([lots_system, lots_sector, lots_trade])

        

        

        

        
        
        
        
        
        
            
            


