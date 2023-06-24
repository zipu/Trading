import math

from tools.instruments import instruments


class Equity:
    """
    현재 자산 상태 정보
    """
    def __init__(self, id, date):
        self.id = id
        self.date=date 
        #self.principal=principal #초기투자금
        self.capital=None #총자산 = 현금 + 증권평가액
        self.max_capital=None #총자산의 최고치
        self.cash=None #현금
        self.security=None #증권평가액 + 증거금 (증거금은 묶여서 사용못하기 때문에)
        self.fixed_capital=None #확정 자산 = 총자산 - 리스크
        self.profit=None #누적확정손익
        self.flame=None #평가손익
        self.commission = None #수수료

        self.system_heat = None 
        self.risk = None
        self.sector_heat = None
        self.sector_risk = None

        self.dd = None
        self.mdd=None
        self.cagr=None
        

class EquityBook:
    """ 자산 내ㅕㄱ """
    def __init__(self, principal, from_date):

        self.from_date = from_date
        self.principal = principal
        self.book = []

        self.date=from_date
        self.principal=principal
        self.capital=principal
        self.cash=principal
        self.security=0
        self.fixed_capital=0 #총자산 - 리스크
        self.profit=0 #누적손익
        self.flame = 0 #평가손익
        self.commission = 0

        self.max_capital = principal

        self.system_heat = 0
        self.risk = 0
        self.sector_heat = {}
        self.sector_risk = {}

        self.dd = 0
        self.mdd= 0
        self.cagr = 0

    def get(self, **kwargs):
        items = []

        for equity in self.book:
            if all(getattr(equity, k) == v for k,v in kwargs.items()):
                items.append(equity)
        return items
      
   
    def log(self, **kwargs):
        items = []
        for equity in self.book:
            item = {
            'id':equity.id,
            'date':equity.date,
            'principal': self.principal,
            'capital':equity.capital,
            'security':equity.security,
            'cash':equity.cash,
            'fixed_capital':equity.fixed_capital,
            'profit':equity.profit,
            'flame': equity.flame,
            'risk':equity.risk,
            'system_heat':equity.system_heat,
            'sector_heat':equity.sector_heat,
            'sector_risk':equity.sector_risk,
            'max_capital': equity.max_capital,
            'dd':equity.dd,
            'mdd':equity.mdd,
            'cagr':equity.cagr,
            'commission':equity.commission
            }

            if all([item[k]==v for k,v in kwargs.items()]):
                items.append(item)
        return items

    def update(self, date, trades, heat):
        equity = Equity(len(self.book)+1, date)
        self.book.append(equity)
        
        #자산 정보
        equity.capital = self.principal + trades.profit + trades.flame - trades.commission
        equity.security = trades.margin + trades.flame
        equity.cash = equity.capital - equity.security
        equity.fixed_capital = equity.capital + trades.risk
        
        if equity.cash <= 0:
            """ 시스템 작동 중단"""
            return True 
        
        
        

        #매매 성능
        equity.profit = trades.profit
        equity.flame = trades.flame
        equity.commission = trades.commission
        equity.risk = trades.risk
        equity.system_heat = heat.system_heat(equity)
        equity.sector_heat, equity.sector_risk = heat.sector_heat(equity, trades.get_on_fires())
        
        
        """
        max_capital = max([status.capital for status in self.book])
        status.dd = 100* (max_capital - status.capital)/max_capital
        status.mdd = max([status.dd for status in self.book])
        """
        
        equity.max_capital = equity.capital if equity.capital > self.max_capital else self.max_capital
        equity.dd = 100* (equity.max_capital - equity.capital)/equity.max_capital
        equity.mdd = equity.dd if equity.dd > self.mdd else self.mdd


        years = (date - self.from_date).days/365

        if years > 0:
            equity.cagr =  math.pow(equity.capital/self.principal, 1/years) - 1
        else: 
            equity.cagr = 0

        #최신 정보 갱신
        self.date=date
        self.capital=equity.capital
        self.cash=equity.cash
        self.security=equity.security
        self.fixed_capital=equity.fixed_capital
        self.profit=equity.profit
        self.flame=equity.flame
        self.commission = equity.commission

        self.system_heat = equity.system_heat
        self.risk = equity.risk
        self.sector_heat = equity.sector_heat
        self.sector_risk = equity.sector_risk

        self.max_capital = equity.max_capital
        self.dd = equity.dd
        self.mdd= equity.mdd
        self.cagr = equity.cagr


    def last(self):
        return self.book[-1]
