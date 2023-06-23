1. 데이터 형식
1) 이 프로젝트에서 사용되는 모든 데이터는 HDF, JSON, CSV 형식으로 저장한다.
    - HDF: 일봉데이터 등 시계열 데이터
    - CSV: 규격화된 메타데이터 
    - JSON: 비규격화된 메타데이터

2) 날짜형식
    - String format data : 'YYYY-mm-dd'
    - numpy datetime format : datetime64[h] or M8[h]
     

2.데이터 구조
 1) OHLC 선물:
	| symbol |
	|   date    |    open    |    high     |    low  |  close   |  volume | open_interest |
    dtype   int32('i')    float('f')      float('f')   float('f')   float('f')    int32('i')  int32('i')
 


* 해외선물 월물 코드 
 1월 - F
 2월 - G
 3월 - H
 4월 - J
 5월 - K
 6월 - M
 7월 - N
 8월 - Q
 9월 - U
10월 - V
11월 - X
12월 - Z
 