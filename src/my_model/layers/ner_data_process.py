from base_process import FeatureExamples
from my_model.embeddings.bert_base.bert import tokenization
import json
from my_model.layers.base_process import InputFeatures
class NerProcess(FeatureExamples):
    #@classmethod
    @staticmethod
    def parse_word_label(line): 
        word = line.strip().split(' ')[0]
        label = line.strip().split(' ')[-1]
        return word,label
    def _read_data(self,input_file):
        """Read a BIO data!"""
        rf = open(input_file,'r')
        lines = []
        words = []
        labels = []
        for line in rf:
            #word = line.strip().split(' ')[0]
            #label = line.strip().split(' ')[-1]

            word,label = self.parse_word_label(line)
            # here we dont do "DOCSTART" check
            if len(line.strip())==0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append((l,w))
                words=[]
                labels = []
            words.append(word)
            labels.append(label)
        rf.close()
        return lines
    def _convert_single_example(self,ex_index, example, mode):
        #here start with zero this means that "[PAD]" is zero
        textlist = example.text.split(' ')
        labellist = example.label.split(' ')
        tokens = textlist
        labels = labellist
        #for i,(word,label) in enumerate(zip(textlist,labellist)):
        #    token = tokenizer.tokenize(word)
        #    tokens.extend(token)
        #    for i,_ in enumerate(token):
        #        if i==0:
        #            labels.append(label)
        #        else:
        #            labels.append("X")
        ## only Account for [CLS] with "- 1".
        if len(tokens) >= self.max_seq_length - 1:
            tokens = tokens[0:(self.max_seq_length - 1)]
            labels = labels[0:(self.max_seq_length - 1)]
        #ntokens = []
        #segment_ids = []
        label_ids = []
        #ntokens.append("[CLS]")
        #segment_ids.append(0)
        #label_ids.append(self.label_map["[CLS]"])
        for i, token in enumerate(tokens):
            #ntokens.append(token)
            #segment_ids.append(0)
            label_ids.append(self.label_map[labels[i]])
        ## after that we don't add "[SEP]" because we want a sentence don't have
        ## stop tag, because i think its not very necessary.
        ## or if add "[SEP]" the model even will cause problem, special the crf layer was used.
        ##ntokens是字符，对应的数字索引是input_ids
        #input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        #vocab_words = tf.contrib.lookup.index_table_from_file(
        #    #args.vocab_words)
        #    vocab_file, num_oov_buckets=1)
    
        input_ids = self.convert_tokens_to_ids(self.vocab_table,tokens)
        mask = [1]*len(input_ids)
        #use zero to padding and you should
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            mask.append(0)
            #segment_ids.append(0)
            label_ids.append(0)
            #ntokens.append("[PAD]")
        assert len(input_ids) == self.max_seq_length
        assert len(mask) == self.max_seq_length
        #assert len(segment_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length
        #assert len(ntokens) == self.max_seq_length
        if ex_index < 3:
            self.logging.info("*** Example ***")
            self.logging.info("guid: %s" % (example.guid))
            self.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            self.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            self.logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
            #self.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            self.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        feature = InputFeatures(
            input_ids=input_ids,
            mask=mask,
            segment_ids=None,
            label_ids=label_ids,
        )
        # we need ntokens because if we do predict it can help us return to original token.
        return feature,label_ids
    def _bert_convert_single_example(self,ex_index, example, mode):
        """
        :param ex_index: example num
        :param example:
        :param self.label_list: all labels
        :param self.max_seq_length:
        :param tokenizer: WordPiece tokenization
        :param mode:
        :return: feature
    
        IN this part we should rebuild input sentences to the following format.
        example:[Jim,Hen,##son,was,a,puppet,##eer]
        labels: [I-PER,I-PER,X,O,O,O,X]
    
        """
        textlist = example.text.split(' ')
        labellist = example.label.split(' ')
        tokens = []
        labels = []
        for i,(word,label) in enumerate(zip(textlist,labellist)):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i,_ in enumerate(token):
                if i==0:
                    labels.append(label)
                else:
                    labels.append("X")
        # only Account for [CLS] with "- 1".
        if len(tokens) >= self.max_seq_length - 1:
            tokens = tokens[0:(self.max_seq_length - 1)]
            labels = labels[0:(self.max_seq_length - 1)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(self.label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(self.label_map[labels[i]])
        # after that we don't add "[SEP]" because we want a sentence don't have
        # stop tag, because i think its not very necessary.
        # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
        #ntokens是字符，对应的数字索引是input_ids
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
    
        mask = [1]*len(input_ids)
        #use zero to padding and you should
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            ntokens.append("[PAD]")
        assert len(input_ids) == self.max_seq_length
        assert len(mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(label_ids) == self.max_seq_length
        assert len(ntokens) == self.max_seq_length
        if ex_index < 3:
            self.logging.info("*** Example ***")
            self.logging.info("guid: %s" % (example.guid))
            self.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            self.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            self.logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
            self.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            self.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        feature = InputFeatures(
            input_ids=input_ids,
            mask=mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
        )
        # we need ntokens because if we do predict it can help us return to original token.
        return feature,label_ids
if __name__ == '__main__':

    #data_path = '/home/anylangtech/.userdata/yongcanpan/gitdir/law/3.train_data/output/output.txt'
    #data_dir = 'InputExample_output'
    #set_type = 'test'

    #tags_path =  '/home/anylangtech/.userdata/yongcanpan/gitdir/law/4.train_model/law_name/tags.txt'
    #vocab_path =  '/home/anylangtech/.userdata/yongcanpan/gitdir/law/4.train_model/vocab.words.txt' 
    data_path = '/home/yongcanpan/MyModel/examples/task_ner/conll2003/test.txt'
    data_dir = 'InputExample_output'
    set_type = 'test'
    tags_path =  '/home/yongcanpan/MyModel/examples/task_ner/conll2003/tags.txt'
    vocab_path =  '/home/anylangtech/.userdata/yongcanpan/NER/sub_NER/bilstm_glove_preject/3.dataset_conll2003/vocab.words.txt' 
    #task_name = 'law_item_class'
    tokenizer=None 
    max_seq_length=256
    embeddings_type='other'
    #embeddings_type='bert'
    #fe = FeatureExamples(data_dir,tags_path,vocab_path,task_name,embeddings_type=embeddings_type,tokenizer=None,max_seq_length=512,args=None)
    fe = NerProcess(data_dir,tags_path,vocab_path,embeddings_type=embeddings_type,tokenizer=None,max_seq_length=512,args=None)
    fe.save_example_tf_record(data_path,set_type)
