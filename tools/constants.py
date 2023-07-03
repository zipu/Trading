"""
파일 경로 등 공통적으로 사용되는 Constants 들 
"""

import os

dirname = os.path.dirname(__file__)

#기본 폴더
BASEDIR = os.path.join(os.path.abspath(dirname), '..')

# 데이터가 저장된 폴더 
DATADIR = os.path.join(os.path.abspath(dirname), '..', 'data')

#상품정보 저장된 csv 파일
INSTRUMENTS_CSV_PATH = os.path.join(DATADIR, 'instruments.csv')

"""
Nasdaq-Data-Link (Stevens Analytics) 관련 자료
"""
# Reference Futures(SRF) 메타데이터 
SRF_METADATA_CSV_PATH = os.path.join(DATADIR, 'nasdaq-data-link','SRF_metadata.csv')

# SRF 에서 다운 가능상품들을 월물별로 정리한 파일
SRF_CONTRACTS_CSV_PATH = os.path.join(DATADIR, 'nasdaq-data-link','SRF_contracts.csv')
SRF_CONTRACTS_JSON_PATH = os.path.join(DATADIR, 'nasdaq-data-link','SRF_contracts.json')

# SRF 월물별 전체 일데이터 데이터 베이스
SRF_CONTRACTS_DB_PATH = os.path.join(DATADIR, 'nasdaq-data-link','SRF_contracts_db.hdf')

# SRF 수정 연결선물 데이터
#1. Backward Panama Roll Over by open interest
SRF_CONTINUOUS_BO_DB_PATH = os.path.join(DATADIR, 'nasdaq-data-link','SRF_continuous_bo_db.hdf')
#2. Backward Panama Roll Over by volume
#SRF_CONTINUOUS_BV_DB_PATH = os.path.join(DATADIR, 'nasdaq-data-link','SRF_continuous_bv_db.hdf')

# SRF 단순 연결선물 데이터
SRF_CONTINUOUS_SO_DB_PATH = os.path.join(DATADIR, 'nasdaq-data-link','SRF_continuous_so_db.hdf')

# SRF 월물데이터 롤오버 정보
#2. Roll Dates information
# roll over by open interest
SRF_ROLLOVER_BO_CSV_PATH = os.path.join(DATADIR, 'nasdaq-data-link','SRF_rollover_bo.csv')
# roll over by volume
#SRF_ROLLOVER_BV_CSV_PATH = os.path.join(DATADIR, 'nasdaq-data-link','SRF_rollover_bv.csv')


"""
Kibot 관련자료
"""
# 키봇 다운가능 상품과 그 월물들 중 거래 가능한 상품들을 시간순으로 배열
KIBOT_CONTRACTS_LIST_CSV_PATH = os.path.join(DATADIR, 'kibot','contracts-list.csv')

# 키봇에서 다운받은 각 상품의 월물 데이터를 저장한 DB 파일
KIBOT_FUTURES_CONTRACTS_DB_PATH = os.path.join(DATADIR, 'kibot','futures-contracts.hdf')

# 키봇에서 월물 데이터를 Backward ajuested panama roll on volume 로 연결한 DB 파일
KIBOT_FUTURES_CONTINUOUS_BV_DB_PATH = os.path.join(DATADIR, 'kibot','futures-continuous-BV.hdf')
