"""
파일 경로 등 공통적으로 사용되는 Constants 들 
"""

import os

dirname = os.path.dirname(__file__)

# 데이터가 저장된 폴더 
DATADIR = os.path.join(os.path.abspath(dirname), '..', 'data')

