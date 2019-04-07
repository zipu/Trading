"""
이베스트 XingAPI 로부터 해외선물의 종목별 데이터를 다운 받아
정리후 저장하는 스크립트
"""
import os
import logging, time, json, pickle, traceback
from datetime import datetime, timedelta
from shutil import copyfile
import numpy as np

from eBest.xingAPI import Session, XAEvents, Query, Real
from eBest.meta import Helper
from models import Products, Product, Contract

#dev
import pythoncom

#log formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)-.19s [%(name)s][%(levelname)s]  %(message)s",
    handlers=[
        logging.FileHandler(f"log/{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

#각 파일경로
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data')
PRODUCTSFILE = os.path.join(BASE_DIR, 'products.pkl')
#RAWDBFILE = os.path.join(BASE_DIR, "rawohlc.db")


class CollectData:
    """
    XingAPI 로그인 -> 데이터 수집 -> 로컬DB에 저장
    일련의 데이터수집 프로세스를 기능하는 클래스
    """
    def __init__(self):
        self.logger = logging.getLogger('CollectData')
        try:
            self.products = pickle.load(open(PRODUCTSFILE, 'rb'))
        except FileNotFoundError:
            self.products = Products()

        XAEvents.instance = self #이벤트 처리 클래스에 현재 인스턴스를 저장
        self.session = None #세션 오브젝트 
        self.query = None #쿼리 오브젝트
        self.timer = 0 #타이머
        #self.yesterday = (datetime.today() - timedelta(1)).strftime('%Y%m%d')
        
        #로그인
        self.next_step(0)

        #dev
        self.flag = False

    def next_step(self, stepno):

        if stepno == 0: #login
            self.login()

        elif stepno == 1: #products info
            self.logger.info("STEP 1: Update products information")
            self.request_productsinfo()

        elif stepno == 2: # 종목별 정보 요청
            self.logger.info("STEP 2: Update contracts information")
            self.contracts = self.products.symbols('all')
            self.request_contractsinfo()
            
        elif stepno == 3: # raw ohlc ---> minute data 로 대체함 (Deprecated)
            self.logger.info("STEP 3: Update Daily OHLC")
            self.contracts = self.products.symbols('db')
            self.request_ohlc()

        elif stepno == 4: # minute data
            self.logger.info("STEP 4: Update Minute Data")
            self.contracts = self.products.symbols('db')
            self.request_minute()
        
        elif stepno == 5: # save
            self.logger.info("STEP 5: Save the objects")
            self.save()

    # 10분당 총 tr 200회 제한 및 tr 당 조회제한 확인
    def check_req_limit(self, trcode):
        count = self.query.get_tr_count_request(trcode)
        limit = self.query.get_tr_count_limit(trcode)
        base = self.query.get_tr_count_base_sec(trcode)
        if count <= 1: #타이머 초기화
            self.timer = time.time()

        proctime = time.time() - self.timer
        time.sleep(base+.1)
        return count, limit, int(proctime)

    def parse_err_code(self, trcode, errcode):
        msg = self.session.get_error_message(errcode)
        self.logger.warning(f"(Error: {errcode}, TR: {trcode}) {msg}")

    #Step 0: 로그인
    def login(self):
        self.session = Session(demo=True)
        
        with open("secret.json") as fp:
            user = json.load(fp)
            if self.session.connect_server():
                self.logger.info("서버 접속 완료")
                if self.session.login(user['id'],user['password']):
                    self.logger.info("로그인 시도중..")
                else:
                    self.logger.warning("로그인 실패")
            else:
                self.logger.warning("서버 접속 실패")
                err = self.session.get_last_error()
                errmsg = self.session.get_error_message(err)
                self.logger.info('Error message: %s', errmsg)

    @XAEvents.on('OnLogin')
    def on_login(self, errcode, msg):
        self.logger.info("(%s): %s", errcode, msg)
        if errcode=='0000':
            self.next_step(1)
            
    #Step 1: 전체상품정보 요청
    def request_productsinfo(self):
        self.query = Query('o3101')
        fields = dict(gubun='')
        errcode = self.query.request(self.query.tr['inblock'], fields)
        if int(errcode) < 0:
            self.parse_err_code(self.query.tr['code'], errcode)

    @XAEvents.on('OnReceiveData', code='o3101')
    def on_request_productsinfo(self, code):
        
        outblock = self.query.tr['outblock']
        count = self.query.get_block_count(outblock)
        # 매매가능 월물리스트 초기화 (!중요)
        for product in self.products.values(): 
            product.tradables = []
        symbols = set()
        
        for i in range(count):
            p = {
                'psymbol': self.query.get_field_data(outblock, 'BscGdsCd', i), #상품코드
                'pname': self.query.get_field_data(outblock, 'BscGdsNm', i), #기초 상품명
                'csymbol': self.query.get_field_data(outblock, 'Symbol', i), #종목코드
                'cname': self.query.get_field_data(outblock, 'SymbolNm', i), #종목명
                'currency': self.query.get_field_data(outblock, 'CrncyCd', i), #통화구분
                'excsymbol': self.query.get_field_data(outblock, 'ExchCd', i), #거래소코드
                'exchange': self.query.get_field_data(outblock, 'ExchNm', i), #거래소명
                'market': Helper.market_symbol(self.query.get_field_data(outblock, 'GdsCd', i)), #시장구분
                'notation': self.query.get_field_data(outblock, 'NotaCd', i), #진법구분
                'unit': self.query.get_field_data(outblock, 'UntPrc', i), #호가단위
                'price_per_unit': self.query.get_field_data(outblock, 'MnChgAmt', i), #호가당가격
                'rgltfactor': self.query.get_field_data(outblock, 'RgltFctr', i), #가격조정계수
                'opentime': datetime.strptime(self.query.get_field_data(outblock, 'DlStrtTm', i), '%H%M%S').time(), #거래시작시간(한국)
                'closetime': datetime.strptime(self.query.get_field_data(outblock, 'DlEndTm', i), '%H%M%S').time(),#거래종료시간(한국)
                'decimal_len': self.query.get_field_data(outblock, 'DotGb', i), #유효소숫점
                'initial_margin': self.query.get_field_data(outblock, 'OpngMgn', i), #개시증거금
                'mntnc_margin': self.query.get_field_data(outblock, 'MntncMgn', i), #유지증거금
                'is_tradable': True if self.query.get_field_data(outblock, 'DlPsblCd', i)=='1' else False, #거래가능구분
            }

            if p['psymbol'] not in self.products:
                product = Product()
                product.updateinfo(p)
                self.products[p['psymbol']] = product
                
                self.logger.info(f'New product {product.name}[{product.symbol}] has created.')
                
            elif p['psymbol'] not in symbols:
                self.products.get(p['psymbol']).updateinfo(p)
            
            if len(self.products.get(p['psymbol']).tradables) > 2:
                continue
            
            self.products.get(p['psymbol']).tradables.append(p['csymbol'])
            symbols.add(p['psymbol'])

        self.logger.info("Products information updated")
        return self.next_step(2)

    #Step 2: 종목(월물)별 상품정보 요청
    def request_contractsinfo(self, symbol=None):
        symbol = symbol or self.contracts.pop(0)
        self.query = Query('o3105')
        
        # 조회 TR 횟수 확인     
        cnt, limit, proctime = self.check_req_limit('o3105')
        self.logger.info(f"({symbol}) Updating information (CNT:{len(self.contracts)}, TR: {cnt}/{limit}, TIME: {proctime} sec)")

        # 조회 요청
        fields = {'symbol': symbol}
        errcode = self.query.request(self.query.tr['inblock'], fields)
        if int(errcode) < 0:
            self.parse_err_code('o3105', errcode)
            if int(errcode) == -34:
                self.request_contractsinfo(symbol)
    
    @XAEvents.on('OnReceiveData', code='o3105')
    def on_request_contractsinfo(self, code):
        outblock = self.query.tr['outblock']
        
        timediff = timedelta(hours=int(self.query.get_field_data(outblock, 'TimeDiff', 0)))
        
        ovsstrday = self.query.get_field_data(outblock, 'OvsStrDay', 0) 
        opendate = ovsstrday + self.query.get_field_data(outblock, 'OvsStrTm', 0)
        ovsendday = self.query.get_field_data(outblock, 'OvsEndDay', 0)
        closedate = ovsendday + self.query.get_field_data(outblock, 'OvsEndTm', 0) 
        pcode = self.query.get_field_data(outblock, 'BscGdsCd', 0) #상품코드
        c = {
            'symbol': self.query.get_field_data(outblock, 'Symbol', 0), #월물코드
            'name': self.query.get_field_data(outblock, 'SymbolNm', 0), #월물명
            'ecprice': self.query.get_field_data(outblock, 'EcPrc', 0), #정산가격
            'appldate': datetime.strptime(self.query.get_field_data(outblock, 'ApplDate', 0), '%Y%m%d'), #종목배치수신일
            'eccd': self.query.get_field_data(outblock, 'EcCd', 0), #정산구분
            'eminicode': self.query.get_field_data(outblock, 'EminiCd', 0), #Emini구분
            'year': self.query.get_field_data(outblock, 'LstngYr', 0), #상장년
            'month': self.query.get_field_data(outblock, 'LstngM', 0), #상장월
            'seqno': self.query.get_field_data(outblock, 'SeqNo', 0), #월물순서
            'expiration': datetime.strptime(self.query.get_field_data(outblock, 'MtrtDt', 0), '%Y%m%d'), #만기일자
            'final_tradeday': datetime.strptime(self.query.get_field_data(outblock, 'FnlDlDt', 0), '%Y%m%d'), #최종거래일
            'opendate': datetime.strptime(opendate, '%Y%m%d%H%M%S')-timediff, #거래시작일자(한국)
            'closedate': datetime.strptime(closedate, '%Y%m%d%H%M%S')-timediff, #거래종료일자(한국)
            'ovsstrday': datetime.strptime(ovsstrday, '%Y%m%d'), #거래시작일 (현지)
            'ovsendday': datetime.strptime(ovsendday, '%Y%m%d'), #거래종료일 (현지)
            'is_tradable': True if self.query.get_field_data(outblock, 'DlPsblCd', 0)=='1' else False #거래가능구분코드
        }

        # 신규월물 추가 및 업데이트  
        if not self.products[pcode].get(c['symbol']):
            self.products[pcode][c['symbol']] = Contract(c['symbol'], c['name'])
            self.logger.info(f"New contract {c['name']} has created")
        self.products[pcode][c['symbol']].updateinfo(c)

        if self.contracts:
            self.request_contractsinfo()
        else:
            self.logger.info("Contract information updated")
            self.next_step(4)        

    #STEP 3: 종목(월물)별 OHLC 데이터 업데이트 (deprecated!!!)
    def request_ohlc(self, symbol=None):
        symbol = symbol or self.contracts.pop(0)
        contract = self.products.get_contract(symbol)
        name = contract.name #self.products.get_name(symbol)
        
        #조회 TR 횟수 확인
        cnt, limit, proctime = self.check_req_limit('o3108')
        self.logger.info(f"({name}) updating OHLC on {symbol} (Cnt:{len(self.contracts)}, TR:{cnt}/{limit}, TIME: {proctime} sec)")

        #TR 조회요청
        self.query = Query('o3108')
        startdate = contract.startday()#self.products.ohlc_startdate(symbol) #시작일
        enddate = (contract.ovsendday - timedelta(1)).strftime('%Y%m%d') # 데이터 마지막 수신일
        fields = dict(
            shcode=symbol,
            gubun=0, #일별
            sdate=startdate,
            edate=enddate,
            cts_date='',
        )
        errcode = self.query.request(self.query.tr['inblock'], fields)
        if int(errcode) < 0:
            self.parse_err_code('o3108', errcode)
            if int(errcode) == -34:
                self.request_ohlc(symbol)

    @XAEvents.on('OnReceiveData', code='o3108')
    def on_request_ohlc(self, code):
        data = []

        outblock = self.query.tr['outblock']
        outblock1 = self.query.tr['outblock1']
        symbol = self.query.get_field_data(outblock, 'shcode', 0) #종목코드
        #cts_date = self.query.get_field_data(outblock, 'cts_date', 0) #연속일자
        cnt = self.query.get_block_count(outblock1)
        for i in range(cnt):
            date = self.query.get_field_data(outblock1, 'date', i) #날짜
            open = self.query.get_field_data(outblock1, 'open', i) #시가
            high = self.query.get_field_data(outblock1, 'high', i) #고가
            low = self.query.get_field_data(outblock1, 'low', i) #저가
            close = self.query.get_field_data(outblock1, 'close', i) #종가
            volume = self.query.get_field_data(outblock1, 'volume', i) #거래량

            datum = (date, open, high, low, close, volume)
            data.append(datum)
        
        self.products.get_contract(symbol).update_ohlc(data)
        
        if self.contracts: 
            self.request_ohlc()
        else:
            self.logger.info("Daily OHLC Data updated")
            self.next_step(4)
        
    #STEP 4: 종목(월물)별 분데이터 업데이트
    def request_minute(self, symbol=None, cts_date='', cts_time='', bnext=False):
        symbol = symbol or self.contracts.pop(0)
        name = self.products.get_name(symbol)
        
        # 연속조회 아닌 신규조회의 경우
        if not bnext: self.data = [] #분데이터 저장할 리스트
        
        #조회 TR 횟수 확인
        self.query = Query('o3103')
        cnt, limit, proctime = self.check_req_limit('o3103')
        self.logger.info(f"({name}) updating Minute Data on {symbol} (Cnt:{len(self.contracts)}, TR:{cnt}/{limit}, TIME: {proctime} sec)")

        #TR 조회요청
        fields = dict(
            shcode=symbol,
            ncnt=30, #분단위
            readcnt=500,
            cts_date=cts_date,
            cts_time=cts_date,
        )

        errcode = self.query.request(self.query.tr['inblock'], fields, bnext)
        if int(errcode) < 0:
            self.parse_err_code('o3103', errcode)
            if int(errcode) == -34:
                self.request_minute(symbol)

    @XAEvents.on('OnReceiveData', code='o3103')
    def on_request_minute(self, code):
        outblock = self.query.tr['outblock']
        outblock1 = self.query.tr['outblock1']
        
        symbol = self.query.get_field_data(outblock, 'shcode', 0) #종목코드
        cts_date = self.query.get_field_data(outblock, 'cts_date', 0) #연속일자
        cts_time = self.query.get_field_data(outblock, 'cts_time', 0) #연속시간
        timediff = int(self.query.get_field_data(outblock, 'timediff', 0)) * (-1) #시차

        contract = self.products.get_contract(symbol)
        lastdate = contract.lastdate_in_db()#분데이터 최신날짜
        cnt = self.query.get_block_count(outblock1)
        for i in range(cnt):
            date = self.query.get_field_data(outblock1, 'date', i) #날짜
            time =  self.query.get_field_data(outblock1, 'time', i) #시간
            open = self.query.get_field_data(outblock1, 'open', i) #시가
            high = self.query.get_field_data(outblock1, 'high', i) #고가
            low = self.query.get_field_data(outblock1, 'low', i) #저가
            close = self.query.get_field_data(outblock1, 'close', i) #종가
            volume = self.query.get_field_data(outblock1, 'volume', i) #거래량

            ndate = (np.datetime64(datetime.strptime(date+time, '%Y%m%d%H%M%S'))\
                    + np.timedelta64(timediff, 'h')).astype('M8[s]').astype('int64')
            
            self.data.append((ndate,open, high, low, close, volume))
        
        # db에 저장된 데이터까지 도달하지 못했으면 연속조회
        # 도달했으면 db 업데이트 후 다음종목 진행
        if cts_date == '00000000' or ndate <= lastdate:
            contract.update_minute(self.data[1:]) #첫번째 데이터는 현재 진행중이라 버림
            if self.contracts:
                self.request_minute()
            else:
                self.logger.info("Minute Data update completed")
                self.next_step(5)
            
        else: 
            self.request_minute(symbol, cts_date=cts_date, cts_time=cts_time, bnext=True)
        
            
        
        
    def save(self):
        """
        종목정보를 pickle 형태로 원본+백업 2개 저장
        db 백업
        """
        #상품정보 저장
        with open(os.path.join(BASE_DIR,'products',f'products_{datetime.now().strftime("%Y%m%d%H%M")}.pkl'), "wb") as f:
            pickle.dump(self.products, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(PRODUCTSFILE, "wb") as f:
            pickle.dump(self.products, f, protocol=pickle.HIGHEST_PROTOCOL)

        #db 백업
        #dst = os.path.join(BASE_DIR, 'rawohlc', f'rawohlc_{datetime.now().strftime("%Y%m%d%H%M")}.db')
        #copyfile(Products.RAWOHLCFILE, dst)
        dst = os.path.join(BASE_DIR, 'rawminute', f'rawminute_{datetime.now().strftime("%Y%m%d%H%M")}.db')
        copyfile(Products.RAWMINUTEFILE, dst)

        #연결선물 업데이트
        self.products.create_continuous_contracts()
        self.logger.info("Continuous futures update completed")

        self.logger.info("All Product Information Properly Saved")
        self.flag = True
             
             



obj = CollectData()
while True:
    pythoncom.PumpWaitingMessages()
    if obj.flag:
        break
    time.sleep(0.1)

#session = Session(demo=True)
#print(session.connect_server())
#print(session.is_load_api())
#print(session.get_server_name())
#print(session.disconnect_server())
