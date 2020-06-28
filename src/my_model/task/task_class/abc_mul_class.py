# this is a multi class abc method,select one from N,N>2
#from my_model.layers.base_process import FeatureExamples
#from my_model.embeddings.bert_base.bert import tokenization
import json
#from my_model.layers.base_process import InputFeatures
#from my_model.abc.model_abc import ABC_MODEL
from my_model.task.abc_task import ABCTask
def get_mul_class(tags_path,max_seq_length=512,embeddings_type='other',args=None):
    data_dir=None
    vocab_path=None
    tokenizer=None
    fe = ABCMulClass(data_dir,'',tags_path,vocab_path,embeddings_type=embeddings_type,tokenizer=tokenizer,max_seq_length=max_seq_length,args=args)
    return fe.get_label_ids
class ABCMulClass(ABCTask):
    #@classmethod
    #def __init__(self,data_dir,tags_path,vocab_path,embeddings_type,tokenizer=None,max_seq_length=256,args):
    #    self.name = 'ABCMulClass'
    #    super().__init__(args)
    @staticmethod
    def parse_word_label(line): 
        res_json = json.loads(line)
        word = res_json['statement_split']
        word = word.strip('：')
        label = res_json['count']
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
            if not label:
                raise ValueError('label is empty,and word is {}'.format(word))
            yield (label,word)
    def get_label_ids(self,label):

        label_ids = [self.label_map.get(la,0) for la in label]
        return label_ids
        ## after that we don't add "[SEP]" because we want a sentence don't have
        ## stop tag, because i think its not very necessary.
        ## or if add "[SEP]" the model even will cause problem, special the crf layer was used.
        ##nntokens是字符，对应的数字索引是input_ids
        #input_ids = tokenizer.convert_ntokens_to_ids(nntokens)
        #vocab_words = tf.contrib.lookup.index_table_from_file(
        #    #args.vocab_words)
        #    vocab_file, num_oov_buckets=1)
    
        #input_ids = self.convert_tokens_to_ids(self.vocab_table,ntokens)
        #mask = [1]*len(input_ids)
        ##use zero to padding and you should
        #while len(input_ids) < self.max_seq_length:
        #    input_ids.append(0)
        #    mask.append(0)
        #    #segment_ids.append(0)
        #    ntokens.append("[PAD]")
        #assert len(input_ids) == self.max_seq_length
        #assert len(mask) == self.max_seq_length
        ##assert len(segment_ids) == self.max_seq_length
        ##assert len(ntokens) == self.max_seq_length
        #if ex_index < 3:
        #    self.logging.info("*** Example ***")
        #    self.logging.info("guid: %s" % (example.guid))
        #    self.logging.info("tokens: %s" % " ".join([str(x) for x in ntokens]))
        #    self.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #    self.logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        #    #self.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #    self.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        #feature = InputFeatures(
        #    input_ids=input_ids,
        #    mask=mask,
        #    segment_ids=None,
        #    label_ids=label_ids,
        #)
        ## we need ntokens because if we do predict it can help us return to original token.
        #return feature,label_ids,ntokens

if __name__ == '__main__':

    data_path = '/home/anylangtech/.userdata/yongcanpan/gitdir/law/3.train_data/output/sim.txt'
    data_dir = 'InputExample_output'
    set_type = 'test'

    tags_path =  '/home/anylangtech/.userdata/yongcanpan/gitdir/law/4.train_model/law_item.dic'
    vocab_path =  '/home/anylangtech/.userdata/yongcanpan/gitdir/law/4.train_model/law_vocab_v1.txt' 
    #data_path = '/home/yongcanpan/MyModel/examples/conll2003/test.txt'
    #data_dir = 'InputExample_output'
    #set_type = 'test'
    #tags_path =  '/home/yongcanpan/MyModel/examples/conll2003/tags.txt'
    #vocab_path =  '/home/anylangtech/.userdata/yongcanpan/NER/sub_NER/bilstm_glove_preject/3.dataset_conll2003/vocab.words.txt' 
    #task_name = 'law_item_class'
    tokenizer=None 
    max_seq_length=256
    embeddings_type='other'
    #embeddings_type='bert'
    #fe = FeatureExamples(data_dir,tags_path,vocab_path,task_name,embeddings_type=embeddings_type,tokenizer=None,max_seq_length=512,args=None)
    fe = ABCMulClass(data_dir,tags_path,vocab_path,embeddings_type=embeddings_type,tokenizer=None,max_seq_length=512,args=None)
    fe.save_example_tf_record(data_path,set_type)
