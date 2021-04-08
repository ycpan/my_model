import os
from my_model.embeddings.bert_base.bert import tokenization
import collections
import pickle
import tensorflow as tf
from my_model.embeddings.bert_base.bert import tokenization
from my_model.abc.base import MY_BASE
from pathlib import Path
import json
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    #if six.PY3:
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))
    #elif six.PY2:
    #    if isinstance(text, str):
    #        return text.decode("utf-8", "ignore")
    #    elif isinstance(text, unicode):
    #        return text
    #    else:
    #        raise ValueError("Unsupported string type: %s" % (type(text)))
    #else:
    #    raise ValueError("Not running on Python2 or Python 3?")

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
    #with tf.io.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab
def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        #TODO: modify for oov, using [unk] replace, if you using english language do not change this
        # output.append(vocab.[item])
        output.append(vocab.get(item, 100))
    return output
def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label

class DataProcessor(MY_BASE):
    def __init__(self,args):
        #import ipdb
        #ipdb.set_trace()
        self.name = 'DataProcessor'
        if hasattr(self,'log_dir'):
            log_dir = self.log_dir
        else:
            log_dir = './'
        if hasattr(self,'log_name'):
            log_name = self.log_name
        else:
            log_name = '{}.log'.format(self.name) 
        super().__init__(log_dir=log_dir,log_name=log_name,args=args)
        self.args = args
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_path,set_type):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    #def get_dev_examples(self, data_path):
    #    """Gets a collection of `InputExample`s for the dev set."""
    #    raise NotImplementedError()

    def get_labels(self,tags_path,embeddings_type):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    #@classmethod
    @staticmethod
    def parse_word_label(line):
        word = line.strip().split(' ')[0]
        label = line.strip().split(' ')[-1]
        return word,label
    def _read_data(self,input_file):
        """Read a BIO data!"""
        rf = open(input_file,'r')
        lines = [];words = [];labels = []
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

class NerProcessor(DataProcessor):
    def __init__(self,args):
        super().__init__(args)
        self.args = args
    def get_examples(self, data_path,set_type):
        return self._create_example(
            self._read_data(data_path),set_type)

    #def get_dev_examples(self, data_path):
    #    return self._create_example(
    #        self._read_data(os.path.join(data_path, "dev.txt")), "dev"
    #    )

    #def get_test_examples(self,data_path):
    #    return self._create_example(
    #        self._read_data(os.path.join(data_path, "test.txt")), "test"
    #    )


    def get_labels(self,tags_path,embeddings_type):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        
        labels = []
        if embeddings_type.lower() == 'bert':
            labels = ["[PAD]", "X","[CLS]","[SEP]"]
        tags = open(tags_path).read().splitlines() 
        labels.extend(tags)

        #return ["[PAD]","B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]
        return labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=texts, label=labels))
        return examples


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.mask = mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example

class FeatureExamples(NerProcessor):
    #def __init__(self,data_dir,tags_path,vocab_path,task_name,embeddings_type='other',do_lower_case=False,tokenizer=None,max_seq_length=256,args=None):
    def __init__(self,data_dir,tags_path,vocab_path,embeddings_type='other',do_lower_case=False,tokenizer=None,max_seq_length=256,args=None):
        self.name = "FeatureExamples"
        self.data_dir = data_dir
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        self.tags_path = tags_path
        self.vocab_path = vocab_path
        #self.task_name = task_name
        #processors = {"ner": NerProcessor}
        self.max_seq_length = max_seq_length
        self.embeddings_type = embeddings_type
        #task_name = self.task_name.lower()
        
        #if task_name not in processors:
        #    raise ValueError("Task not found: %s" % (task_name))
        #self = processors[task_name]()
        # tags_path 表示原标签，labels表示包含其它标签，比如[CLS],[X],[PAD]等。
        self.label_list = self.get_labels(self.tags_path,self.embeddings_type)
        self.vocab_table = load_vocab(self.vocab_path)
    
        if self.embeddings_type.lower() == 'bert':
            self.tokenizer = tokenization.FullTokenizer(
                vocab_file=self.vocab_path, do_lower_case=do_lower_case)
        else:
            self.tokenizer = tokenizer
        self.label_map = {}
        for (i,label) in enumerate(self.label_list):
            self.label_map[label] = i
        with open(data_dir+"/label2id.pkl",'wb') as w:
            pickle.dump(self.label_map,w)
        #import ipdb
        #ipdb.set_trace()
        self.indices = [idx for tag,idx in self.label_map.items() if tag.strip() != 'O']
        
        #import ipdb
        #ipdb.set_trace()
        #if hasattr(self,'log_dir'):
        #    log_dir = self.log_dir
        #else:
        #    log_dir = self.data_dir
        #if hasattr(self,'log_name'):
        #    log_name = self.log_name
        #else:
        #    log_name = '{}.log'.format(self.name) 
        super().__init__(args)
        self.args = args
        
    #def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file,data_dir,vocab_file,logging,mode=None):
    def filed_based_convert_examples_to_features(self,examples, output_file,mode=None):
        writer = tf.python_io.TFRecordWriter(output_file)
        batch_tokens = []
        batch_labels = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                self.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
            convert_single_example = self.get_convert_single_example()
            feature,label_ids = convert_single_example(ex_index, example, mode)
            #batch_tokens.extend(ntokens)
            batch_labels.extend(label_ids)
    
            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            if feature.mask:
                features["mask"] = create_int_feature(feature.mask)
            if feature.segment_ids:
                features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
        # sentence token in each batch
        writer.close()
        return batch_tokens,batch_labels
    def get_example_tf_record(self,input_data_path,is_write=True,set_type='train'):
        example_file_tf_record = os.path.join(self.data_dir, "{}.tf_record".format(set_type))
        try:
            with open(self.data_dir+"/params.json",'r') as w:
                params = json.load(w)
                self.len_train_examples = params['len_train_examples']
        except Exception as e:
            raise ValueError('params.josn not exist,you may set is_write=True to get len_train_examples and rewrite tf_record file')
            
        if is_write:
            examples = self.get_examples(input_data_path,set_type)
            if set_type == 'train':
                self.len_train_examples = len(examples)
                params = {'len_train_examples':self.len_train_examples}
                
                
                with open(self.data_dir+"/params.json",'w') as w:
                    json.dump(params,w,ensure_ascii=False)
            _,_ = self.filed_based_convert_examples_to_features( examples, example_file_tf_record)
        return example_file_tf_record
    @staticmethod
    def convert_tokens_to_ids(vocab, tokens):
        return convert_by_vocab(vocab, tokens)
    @staticmethod
    def convert_ids_to_tokens(inv_vocab, ids):
        return convert_by_vocab(inv_vocab, ids)
    
    #def _convert_single_example(ex_index, example, label_list, self.max_seq_length, tokenizer, mode,data_dir,vocab,self.logging):
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
    def get_convert_single_example(self):
        if self.embeddings_type == 'bert':
            return self._bert_convert_single_example
        else:
            return self._convert_single_example
if __name__ == '__main__':

    #data_path = '/home/anylangtech/.userdata/yongcanpan/gitdir/law/3.train_data/output/output.txt'
    #data_dir = 'InputExample_output'
    #set_type = 'test'

    #tags_path =  '/home/anylangtech/.userdata/yongcanpan/gitdir/law/4.train_model/law_name/tags.txt'
    #vocab_path =  '/home/anylangtech/.userdata/yongcanpan/gitdir/law/4.train_model/vocab.words.txt' 
    data_path = '/home/yongcanpan/MyModel/examples/conll2003/test.txt'
    data_dir = 'InputExample_output'
    set_type = 'test'
    tags_path =  '/home/yongcanpan/MyModel/examples/conll2003/tags.txt'
    vocab_path =  '/home/anylangtech/.userdata/yongcanpan/NER/sub_NER/bilstm_glove_preject/3.dataset_conll2003/vocab.words.txt' 
    #task_name = 'law_item_class'
    tokenizer=None 
    max_seq_length=256
    embeddings_type='other'
    #embeddings_type='bert'
    #fe = FeatureExamples(data_dir,tags_path,vocab_path,task_name,embeddings_type=embeddings_type,tokenizer=None,max_seq_length=512,args=None)
    fe = FeatureExamples(data_dir,tags_path,vocab_path,embeddings_type=embeddings_type,tokenizer=None,max_seq_length=512,args=None)
    fe.save_example_tf_record(data_path,set_type)
