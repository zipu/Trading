from signal import raise_signal
from site import setcopyright
from unicodedata import name
from tools.instruments import instruments
#from .book import Book

LONG = 1
SHORT = -1

class Trade:
    """
    개별 매매 기록
    """
    def __init__(self, id, entrydate, name, symbol, sector, position, entryprice\
        ,entrylots, stopprice, entryrisk, entryrisk_ticks, commission):
        
        # 진입 정보
        self.id = id
        self.entrydate = entrydate
        self.name = name
        self.symbol = symbol
        self.sector = sector
        self.position = position
        self.entryprice = entryprice
        self.entrylots = entrylots
        self.entryrisk = entryrisk
        self.entryrisk_ticks = entryrisk_ticks
        self.commission = commission

        #청산 정보
        self.exits = []
        """
        exit = {
            'exittype': exit or stop, 
            'exitdate': datetime,
            'exitprice': price,
            'exitlot': integer,
            'profit': value,
            'profit_tick': integer,
            'duration': days,
            'result': WIN or LOSE
        }
        """

        #상태 정보
        self.currentprice = None
        self.stopprice = stopprice
        self.risk = None
        self.risk_ticks = None
        self.lots = entrylots
        self.flame = 0 #평가손익
        self.profit = 0 #확정손익
        self.duration = 0 #보유기간
        self.exittype = '' #청산 종류
        self.result = ''
        self.on_fire = True

        
        self.update_status(entryprice, stopprice)

    def __str__(self):
        return f"""
        id: {self.id}
        name: {self.name}
        symbol: {self.symbol}
        sector: {self.sector}
        position: {self.position}
        entry price: {self.entryprice}
        entrylots: {self.entrylots}
        entryrisk: {self.entryrisk}
        result: {self.result}
        on_fire: {self.on_fire}
        """

    
    def add_exit(self, exitdate, exitprice, exitlots, exittype):

        profit, profit_ticks = self.price_to_value(
            self.symbol, self.position, self.entryprice, exitprice, exitlots)

        exit = {
            'exittype': exittype, 
            'exitdate': exitdate,
            'exitprice': exitprice,
            'exitlots': exitlots,
            'profit': profit,
            'profit_ticks': profit_ticks,
            'duration': (exitdate - self.entrydate).days,
            'result': 'WIN' if profit>0 else 'LOSE'
        }
        self.exits.append(exit)
        
        self.update_status(exitprice, self.stopprice)

        return profit
    
    def update_status(self, currentprice, stopprice):
        """ 매매 상태 업데이트 """
        self.currentprice = currentprice
        self.stopprice = stopprice

        self.lots = self.entrylots - sum([exit['exitlots'] for exit in self.exits])

        self.risk, self.risk_ticks = self.price_to_value(
            self.symbol, self.position, currentprice, stopprice, self.lots)
        
        self.flame, _ = self.price_to_value(
            self.symbol, self.position, self.entryprice, currentprice, self.lots)

        self.profit = sum([exit['profit'] for exit in self.exits])
        self.profit_ticks = sum([exit['profit_ticks'] for exit in self.exits])
        

        if self.lots < 0:
            raise ValueError(f"청산 계약수가 진입 계약수보다 많습니다: {self.id}")
        
        if self.lots == 0:
            self.on_fire = False
            self.duration = self.exits[-1]['duration'] if self.exits else 0
            self.result = 'WIN' if self.profit > 0 else 'LOSE'
            self.exittype = self.exits[-1]['exittype'] if self.exits else ''
    
    
    def price_to_value(self, symbol, position, initial_price, final_price, lots):
        """
        상품 가격 차이로부터 가치를 계산
        """
        tickunit = instruments[symbol].tickunit
        tickvalue = instruments[symbol].tickvalue
        value_ticks = round(position*(final_price-initial_price)/tickunit)
        value = value_ticks * tickvalue * lots

        return value, value_ticks


class TradesBook:
    """
    개별 매매기록
    """
    def __init__(self, system_id):
        self.system_id = system_id
        self.book = [] #전체 매매 리스트
        self.fires = [] #진행중인 매매 리스트

        # 누적 상태 정보
        #self.profit = 0
        #self.margin = 0

        # 매매 문서 형식
        
        self.profit = 0 #누적 수익
        self.commission = 0 #누적 수수료
        self._flame = 0 #평가 손익
        #self.margin = 0
        #self.risk = 0
        #self.commission = 0


    @property
    def flame(self):
        """ 평가손익(flame) """
        fires = self.get_on_fires()
        return sum([fire.flame for fire in fires])

    @property
    def margin(self):
        """ 증거금 합산 """
        fires = self.get_on_fires()
        return sum([instruments[fire.symbol].margin * fire.lots for fire in fires])

    @property
    def risk(self):
        """ 총 리스크 """
        fires = self.get_on_fires()
        return sum([fire.risk for fire in fires])


    def log(self, **kwargs):
        items = []
        for trade in self.book:
            item = {
            'id':trade.id,
            'entrydate': trade.entrydate,
            #'exitdate': trade.exits[-1]['exitdate'] if not trade.on_fire else None,
            'name':trade.name,
            'symbol':trade.symbol,
            'sector':trade.sector,
            'position':'Long' if trade.position==1 else 'Short',
            'entryprice':trade.entryprice,
            'entrylots':trade.entrylots,
            'entryrisk':trade.entryrisk,
            'entryrisk_ticks':trade.entryrisk_ticks,
            'commission':trade.commission,
            #'exits':trade.exits,
            '#exits':len(trade.exits),
            }

            for i, exit in enumerate(trade.exits):
                item[f'exit({i})_date'] = exit['exitdate']
                item[f'exit({i})_price'] = exit['exitprice']
                item[f'exit({i})_lots'] = exit['exitlots']
                item[f'exit({i})_profit'] = exit['profit']
                item[f'exit({i})_profit_ticks'] = exit['profit_ticks']
                item[f'exit({i})_duration'] = exit['duration']
                item[f'exit({i})_type'] = exit['exittype']
            item['exitdate']=trade.exits[-1]['exitdate'] if trade.exits else None
            item['profit']=trade.profit
            item['profit_ticks']=trade.profit_ticks
            item['duration'] = trade.exits[-1]['duration'] if trade.exits else None
            item['result']=trade.result
            item['currentprice']=trade.currentprice
            item['stopprice']=trade.stopprice
            item['risk']=trade.risk
            item['risk_ticks']=trade.risk_ticks
            item['lots']=trade.lots
            item['flame']=trade.flame
            item['on_fire']=trade.on_fire


            if all([item[k]==v for k,v in kwargs.items()]):
                items.append(item)
        
        return items


    def get(self, **kwargs):
        items = []

        for trade in self.book:
            if all(getattr(trade, k) == v for k,v in kwargs.items()):
                items.append(trade)
        return items
      

    def get_on_fires(self, **kwargs):
        if kwargs:
            fires = []
            for fire in self.fires:
                if all(getattr(fire, k) == v for k,v in kwargs.items()):
                    fires.append(fire)

            return fires
        else:
            return self.fires
          
    
    def add_entry(self, entrydate, symbol, sector, position, entryprice, entrylots, stopprice, commission,\
                  entryrisk, entryrisk_ticks):
        
        #risk, risk_ticks = self.price_to_value(symbol, position, entryprice, stopprice, entrylots)
        if entryrisk < 0:
            raise ValueError(f"Risk must be positive: {entrydate},{symbol},{position},{entryprice},{stopprice}")
        
        trade = Trade(
            id = len(self.book)+1,
            entrydate = entrydate,
            name = instruments[symbol].name,
            symbol = symbol,
            sector = sector,
            position = position,
            entryprice = entryprice,
            entrylots = entrylots,
            entryrisk = entryrisk,
            entryrisk_ticks = entryrisk_ticks,
            commission = commission,
            stopprice = stopprice,
        )

        self.book.append(trade)
        self.fires.append(trade)
        self.commission += commission
       

    def add_exit(self, fire, exitdate, exitprice, exitlots, exittype):
        """
        청산 기록 
        """
        profit = fire.add_exit(exitdate, exitprice, exitlots, exittype)
        self.profit += profit
        if not fire.on_fire:
            self.fires.pop(self.fires.index(fire))
       

    def reject(self, symbol, entrydate, sector, position, entryprice):
        trade = Trade(
            id = len(self.book)+1,
            entrydate = entrydate,
            name = instruments[symbol].name,
            symbol = symbol,
            sector = sector,
            position = position,
            entryprice = entryprice,
            entrylots = 0,
            entryrisk = 0,
            entryrisk_ticks = 0,
            commission = 0,
            stopprice = 0,
        )
        trade.on_fire = False
        trade.result = 'REJECT'

        self.book.append(trade)


    def update_status(self, fire, currentprice, stopprice):
        fire.update_status(currentprice, stopprice)
    
    
