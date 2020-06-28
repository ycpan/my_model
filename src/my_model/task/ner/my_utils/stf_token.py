from pprint import pprint
import sys
from pycorenlp.corenlp import StanfordCoreNLP
class STF_TOKEN:
    def __init__(self):
        self.host = "http://124.193.223.50"
        self.port = "8047"
        self.nlp = StanfordCoreNLP(self.host + ":" + self.port)
    def token(self,line):
	#分词
        output = self.nlp.annotate(
            line,
            properties={
                "outputFormat": "json",
                #"annotators": "depparse,ner,entitymentions,sentiment"
                "annotators": "tokenize"
                #"annotators": "tokenize"
            }
        )
        #pprint(output)
        res =  [d['originalText'] for d in output['tokens']]
        res = ' '.join(res).split(' ')
        return res
    def token_ssplit(self,line):
	# 分词 + 分句
        output = self.nlp.annotate(
            line,
            properties={
                "outputFormat": "json",
                #"annotators": "depparse,ner,entitymentions,sentiment"
                "annotators": "tokenize, ssplit"
                #"annotators": "tokenize"
            }
        )
        res =  [' '.join([d['originalText'] for d in l['tokens']]).split(' ') for l in  output['sentences']]
        return res
if __name__ == '__main__':
    stf_token = STF_TOKEN()

    #text = "Joshua Brown, 40, was killed in Florida in May when his Tesla failed to " 
    text ="130 2016-2018 Chung Sye-kyun (supported by DP)"
 
    res = stf_token.token(text)
    #res = stf_token.token_ssplit(text)

    #print(' '.join(res))
    print(res)
