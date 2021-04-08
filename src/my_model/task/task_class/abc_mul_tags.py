# this is multi tags abc class method,selcet m form N,m<N,m>1,N>2
#from my_model.task.task_class.abc_class import ABCClass 

#from my_model.layers.base_process import InputFeatures
from my_model.task.abc_task import ABCTask
import json
def get_mul_tags(tags_path,max_seq_length=512,embeddings_type='other',args=None):
    data_dir=None
    vocab_path=None
    tokenizer=None
    fe = ABCMulTags(data_dir,'./',tags_path,vocab_path,embeddings_type=embeddings_type,tokenizer=tokenizer,max_seq_length=max_seq_length,args=args)
    return fe.get_label_ids
class ABCMulTags(ABCTask):
    @staticmethod
    def parse_word_label(line): 
        res_json = json.loads(line)
        word = res_json['statement_split']
        word = word.strip('：')
        label = res_json['law_item']
        return word,label
    def _read_data(self,input_file):
        """Read a BIO data!"""
        rf = open(input_file,'r')
        lines = []
        #words = []
        #labels = []
        for line in rf:
            #word = line.strip().split(' ')[0]
            #label = line.strip().split(' ')[-1]

            word,label = self.parse_word_label(line)
            # here we dont do "DOCSTART" check
            #checkfor la in label:
            #check    if la:
            #check        if word:
            #check            #words.append(word)
            #check            #labels.append(la)
            if not label:
                raise ValueError('label is empty,and word is {}'.format(word))
            yield (label,word)
        #    lines.append((label,word))
        #rf.close()
        #return lines
    def get_label_ids(self,label):
        label_ids = [0] * len(self.label_list)
        for la in label:
            idx = self.label_map.get(la,0)
            if idx != 0:
                label_ids[idx] = 1
    
        return label_ids
    #def _convert_single_example(self,ex_index, example, mode):

    #    input_ids,mask,segment_ids,ntokens = self._convert_single_feature(ex_index,example,mode)
    #    label_ids = self._convert_single_feature(ex_index,example,mode)
    #    feature = InputFeatures(
    #        input_ids=input_ids,
    #        mask=mask,
    #        segment_ids=segment_ids,
    #        label_ids=label_ids,
    #    )
    #    # we need ntokens because if we do predict it can help us return to original token.
    #    return feature,label_ids,ntokens
