"""
백테스트 과정에서 시장(Market)의 역할을 함
일봉 데이터 등의 Raw Data 또는 가공된(pre-processed)된 지표값을 제공함 ex)EMA

추후, 물가지수 또는 금리 등의 기초(fundamental) 정보도 제공 할 수도 있음
"""

from ..instruments import instruments

long = LONG = L = 1
short = SHORT = S = -1

class Market:
    """
    시장 정보 제공
    지표 생성기능
    """
    
    #commission = 3.5 #편도 수수료
    #sectors = ['Currency','Grain','Meat','Tropical','Petroleum','Equity','Rate']
    
    def __init__(self, feed, signal=None):
        """
        feed: pandas dataframe 형식의 시장 기초데이터
        signal: signal 생성 함수
        """
        if not signal:
            signal = self.default_signal
        
        self.pinfo = {}
        self._preprocessing(feed, signal)
        
    def _preprocessing(self, feed, signal):
        """
        종목별로 시그널을 생성하여 feed에 merge하고
        종목별 데이터를 날짜순으로 모두 합침
        """
        header = feed.attrs['columns'].split(';') #데이터 명
        
        container = []
        pinfo = product_info()
        for (cnt,inst) in enumerate(feed.values()):
            symbol =  inst.attrs['symbol']
            if symbol == 'None' or not symbol:
                continue
            
            else:
                self.pinfo[symbol] = pinfo[symbol]
            
            datatable = pd.DataFrame(inst.value[:,1:], index=inst.value[:,0].astype('M8[s]'), columns=header[1:])
            datatable.sort_index(inplace=True)
            
            if signal:
                print(f"\r preprocessing data...({cnt})          ", end='', flush=True)
                signal(datatable)
            columns = datatable.columns.tolist()
            new_column = [[symbol for i in range(len(columns))], columns]
            datatable.columns = new_column
            container.append(datatable)
        print('\nDone')
        # warm period = 60days
        self.feed = pd.concat(container, axis=1).sort_index(axis=1).iloc[60:]
    
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
    
    @classmethod
    def set_ATR(cls, metrics, span=20):
            df = pd.DataFrame()
            df['hl'] = metrics['high'] - metrics['low']
            df['hc'] = np.abs(metrics['high'] - metrics['close'].shift(1))
            df['lc'] = np.abs(metrics['low'] - metrics['close'].shift(1))
            df['TR'] = df.max(axis=1)
            metrics['ATR'] = df['TR'].ewm(span).mean()
    
    @staticmethod
    def default_signal(datatable):
        """
        시장 기초데이터로부터 MA, trend index등을 생성하는 
        data preprocessing method이다.
        
        datatable: 종목별 ohlc 데이터를 저장한 pandas dataframe
        gc: 20일 지수이평과 60일 지수이평의 교차신호
        atr: 20일 atr
        
        """
        Market.set_ATR(datatable, span=20)
        
        ema20 = datatable['close'].ewm(20).mean()
        ema60 = datatable['close'].ewm(60).mean()
        
        datatable['golden'] = (ema20>ema60).astype('int').diff().shift(1)
        datatable.dropna(inplace=True)