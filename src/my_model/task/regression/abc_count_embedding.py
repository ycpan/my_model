#from my_model.task.task_class.logid_abc_class import LogidABCClass
from my_model.layers.base_process import InputFeatures
from my_model.layers.base_process import create_int_feature
#from my_model.abc.model_abc import ABC_MODEL
from my_model.task.abc_task import ABCTask
import json
import tensorflow as tf
import collections
class RegressionABCClass(ABCTask):
    #@classmethod
    #def __init__(self,data_dir,tags_path,vocab_path,embeddings_type,tokenizer=None,max_seq_length=256,args):
    #    self.name = 'ABCClass'
    #    super().__init__(args)
    @staticmethod
    def parse_word_label(line): 
        res_json = json.loads(line)
        word = res_json['statement_split']
        word = word.strip('：')

        label = res_json['sentencing']
        new_label = []
        for item in label:
            try:
                int(item)
                new_label.append(item)
            except Exception as e:
                new_label.append(0)
                
        return word,new_label[0:1]
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
                #raise ValueError('label is empty,and word is {}'.format(word))
                word = ''
            yield (label,word)
    def get_label_ids(self,label):
        label_ids = label
        return label_ids
    #def _convert_single_example(self,ex_index, example, mode):

    #    #here start with zero this means that "[PAD]" is zero
    #    textlist = example.text.split(' ')
    #    label = example.label
    #    ntokens = textlist
    #    if len(ntokens) >= self.max_seq_length - 1:
    #        ntokens = ntokens[0:(self.max_seq_length - 1)]
    #        label = label[0:(self.max_seq_length - 1)]
    #    label_ids = label
    #    ## after that we don't add "[SEP]" because we want a sentence don't have
    #    ## stop tag, because i think its not very necessary.
    #    ## or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    #    ##nntokens是字符，对应的数字索引是input_ids
    #    #input_ids = tokenizer.convert_ntokens_to_ids(nntokens)
    #    #vocab_words = tf.contrib.lookup.index_table_from_file(
    #    #    #args.vocab_words)
    #    #    vocab_file, num_oov_buckets=1)
    #
    #    input_ids = self.convert_tokens_to_ids(self.vocab_table,ntokens)
    #    mask = [1]*len(input_ids)
    #    #use zero to padding and you should
    #    while len(input_ids) < self.max_seq_length:
    #        input_ids.append(0)
    #        mask.append(0)
    #        #segment_ids.append(0)
    #        ntokens.append("[PAD]")
    #    assert len(input_ids) == self.max_seq_length
    #    assert len(mask) == self.max_seq_length
    #    #assert len(segment_ids) == self.max_seq_length
    #    #assert len(ntokens) == self.max_seq_length
    #    if ex_index < 3:
    #        self.logging.info("*** Example ***")
    #        self.logging.info("guid: %s" % (example.guid))
    #        if ntokens:
    #            self.logging.info("tokens: %s" % " ".join([str(x) for x in ntokens]))
    #        if input_ids:
    #            self.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #        if mask:
    #            self.logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
    #        #self.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #        if label_ids:
    #            self.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    #    feature = InputFeatures(
    #        input_ids=input_ids,
    #        mask=mask,
    #        segment_ids=None,
    #        label_ids=label_ids,
    #    )
    #    # we need ntokens because if we do predict it can help us return to original token.
    #    return feature,label_ids,ntokens

    #def filed_based_convert_examples_to_features(self,examples, output_file,set_type='train',mode=None):
    #    writer = tf.python_io.TFRecordWriter(output_file)
    #    #batch_tokens = []
    #    #batch_labels = []
    #    #example_count = 0
    #    cnt = 0
    #    for (ex_index, example) in enumerate(examples):
    #        #example_count = ex_index
    #        if ex_index % 5000 == 0:
    #            self.logging.info("Writing example %d " % (ex_index))
    #        convert_single_example = self.get_convert_single_example()
    #        feature,label_ids,ntokens = convert_single_example(ex_index, example, mode)
    #
    #        features = collections.OrderedDict()
    #        features["input_ids"] = create_int_feature(feature.input_ids)
    #        if feature.label_ids[0]!=0:
    #            
    #            self.logging.warning('feature.label_ids is {}'.format(' '.join(ntokens)))
    #            try:
    #                features["label_ids"] = create_int_feature(feature.label_ids)
    #            except Exception as e:
    #                self.logging.error(e)
    #        else:
    #            #self.logging.warning('feature.label_ids is None,ignore it.the label_ids is {}'.format(' '.join(ntokens)))
    #            continue
    #        #batch_tokens.append(ntokens)
    #        #batch_labels.extend(label_ids)
    #        if feature.mask:
    #            features["mask"] = create_int_feature(feature.mask)
    #        if feature.segment_ids:
    #            features["segment_ids"] = create_int_feature(feature.segment_ids)
    #        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    #        writer.write(tf_example.SerializeToString())
    #        cnt += 1
    #        #yield ntokens,label_ids
    #    return cnt
        # sentence token in each batch
        #if set_type == 'train':
        #    self.len_train_examples = example_count
        #    self.FeatureExamples_params['len_train_examples']=self.len_train_examples
        #    with open(self.data_dir+"/params.json",'w') as w:
        #        json.dump(self.FeatureExamples_params,w,ensure_ascii=False)
        #writer.close()
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
    fe = ABCClass(data_dir,tags_path,vocab_path,embeddings_type=embeddings_type,tokenizer=None,max_seq_length=512,args=None)
    fe.save_example_tf_record(data_path,set_type)
