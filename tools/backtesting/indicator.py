"""
EMA, Volatility Index 등 시장 지표를 계산하는 클래스
모든 시계열 지표들은 지표계산에 사용되는 기간이 필요하다
ex) 5일선, 20일선
이 시간간격을 Time Period 또는 줄여서 period로 부르기로 한다. 
"""

class Indicator:
    pass



class MA(Indicator):
    """
    이동평균 (Moving Average)
    """
    def __init__(self, quotes, period=5):
        self.period = period
        self.name = f"Moving Average of {period}-day period"
        self._calculate(quotes)
    
    def _calculate(self, quotes):
        




