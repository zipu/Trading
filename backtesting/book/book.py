# 매매 내용 및 결과를 기록하는 모델 클래스

from tools.instruments import instruments

class Book:
    """
    매매기록을 담는 클래스
    """
    def __init__(self, name):
        self.id = -1
        self.name = name
        self.statements = []

    def __call__(self):
        return self.statements

    def __repr__(self):
        return f"{self.name} 의 기록"
    
    
    def write(self, statement):
        """ 기록 """
        
        self.id += 1
        statement[id] = self.id
        self.statements.append(statement)

    def update(self, statement):
        """ 추가기록 or 수정 """
        statement_old = self.statements[statement['id']]
        for k,v in statement.items():
            statement_old[k] = v 

    def get(self, **kwargs):
        lists = []
        for statement in self.statements:
            if all(statement[k] == v for k,v in kwargs.items()):
                lists.append(statement)

        return lists

    def get_last(self):
        return self.statements[-1]    