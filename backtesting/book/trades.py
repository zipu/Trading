from .book import Book

LONG = 1
SHORT = -1

class Trades:
    """
    개별 매매기록
    """
    def __init__(self, system_id):
        self.system_id = system_id
        self.book = Book(name='trades')

        # 진입정보
        #self.entrydate = statement['entrydate']
        #self.symbol = statement['symbol']
        #self.position = statement['position']
        #self.entryprice = statement['entryprice']
        #self.entrylots = statement['entrylots']
        #self.stopprice = statement['stopprice']
        #self.risk = statement['risk']
        #self.risk_tick = statement['risk_ticks']
#
        ##청산정보
        #self.exitdates  = []
        #self.exitlots = []
        #self.exitprices =[]
        #
        ##상태 및 결과
        #self.current_lots = self.entrylots
        #self.profits = []
        #self.profits_tick = []
        #self.duration = 0
        #self.is_open = True
        #self.description = ''

    def get(self, **kwargs):
        return self.book.get(kwargs)


    def get_on_fires(self):
        return self.book.get(on_fire=True)

    def add_entry(self, entrydate, symbol, sector, position, entryprice, entrylots, entryrisk):
        statement = {
            #진입 정보
            'entrydate': entrydate,
            'symbol': symbol,
            'sector': sector,
            'position': position,
            'entryprice': entryprice,
            'entrylots': entrylots,
            'entryrisk': entryrisk,
            #청산정보
            'exitdates': [],
            'exitlots': [],
            'profit': [],
            'profit_ticks': [],
            'duration': 0,
            'on_fire': True,
            'result': [],

            #상태정보
            'risk':entryrisk,
            'lots': 0,
        }

        self.book.write(statement)
        

    def add_exit(self, fire, exitdate, exitprice, exitlots, profit, profit_ticks, result=None):
        fire['exitdates'].append(exitdate)
        fire['exitprice'].append(exitprice)
        fire['exitlots'].append(exitlots)
        fire['profit'].append(profit)
        fire['profit_ticks'].append(profit_ticks)
        fire['lots'] = fire['entrylots'] - sum(fire['exitlots'])

        if result:
            fire['result'].append(result)


        if fire['lots'] < 0:
            raise ValueError(f"청산 계약수가 진입 계약수보다 많습니다: {fire}")

        elif fire['lots'] == 0:
            fire['on_fire'] = False #매매종료
        
        
