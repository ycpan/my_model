from sner import Ner
import sys
class STF_NER:
    def __init__(self):
        self.tagger = Ner(host='localhost',port=8046)
    def get_ner(self,l):
        res = self.tagger.get_entities(l)
        return res
        

if __name__ == '__main__':
    # 本工具使用之前需要先分词，最好用斯坦福分词工具。详见stf_token.py
    ner = STF_NER()    
    ## test
    test_string = "Alice went to the Museum of Natural History."
    res = ner.get_ner(test_string)
    print(res)
