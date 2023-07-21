import os
import requests
import json

BASEURL = "https://openapi.ebestsec.co.kr:8080"
SECRETFILE = os.path.join('..','..','secret.json')


class Ebest:

    def __init__(self):
        with open(SECRETFILE) as f:
            self.secret = json.load(f)['ebest']['overseas']

    def login(self):
        path = "oauth2/token"
        url = f"{BASEURL}/{path}"
        header = {"content-type":"application/x-www-form-urlencoded"}
        body = {
            "appkey": self.secret['appkey'],
            "appsecretkey": self.secret['appsecretkey'],
            "grant_type":"client_credentials",
            "scope":"oob"
        }
        res = requests.post(url, headers=header, data=body)
        if res.ok:
            self.access_token = res.json()['access_token']
            print("login succeeded")
            return True
        else: 
            print(f"login failed: {res.text}")
            return False
    
    def transactions(self):
        """ 주문체결내역상세조회"""
        path="overseas-futureoption/accno"
        url = f"{BASEURL}/{path}"
        header = {  
            "content-type":"application/json; charset=utf-8", 
            "authorization": f"Bearer {self.access_token}",
            "tr_cd":"CIDBQ02400", 
            "tr_cont":"N",
            "tr_cont_key":"",
        }

        body = {
                    "CIDBQ02400InBlock1": {
                        "RecCnt": 1,
                        "IsuCodeVal": "",
                        "QrySrtDt": "20230516",
                        "QryEndDt": "20230717",
                        "ThdayTpCode": "0",
                        "OrdStatCode": "1",
                        "BnsTpCode": "0",
                        "QryTpCode": "2",
                        "OrdPtnCode": "00",
                        "OvrsDrvtFnoTpCode": "A"
                    }
                }
        res = requests.post(url, headers=header, data=json.dumps(body))
        if res.ok:
            print("login succeeded")
            return res
        else: 
            print(f"login failed: {res.text}")
            return res
        
    

