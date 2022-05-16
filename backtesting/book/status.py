from .book import Book
from tools.instruments import instruments

class Status:
    """ 시스템 재무 상태 기록 """
    def __init__(self, principal):
        self.principal = principal
        self.book = Book(name='Asset Book')
        
        self.capital = self.principal #현재 총 자산 = 현금 + 평가자산
        self.cash = self.principal #현재 현금자산
        self.security = 0 #현재 증권 자산 = 증거금 + 평가손익
        self.fixed_capital = self.principal #총 자산 - 리스크 (손절 기준 총 자산)
        

    def write(self, date, capital, cash, security, fixed_capital):
        statement = {
            'date': date,
            'capital': capital,
            'cash':cash,
            'security': security,
            'fixed_capital': fixed_capital
        }

        self.capital = capital
        self.cash = cash
        self.security = security
        self.fixed_capital = fixed_capital

        self.book.write(statement)

    def update(self, date, trades):
        fires = trades.get_on_fires()
        


        



    
    def status(self):
        return {
            'capital': self.capital,
            'cash':self.cash,
            'security': self.security,
            'fixed_capital': self.fixed_capital
        }
