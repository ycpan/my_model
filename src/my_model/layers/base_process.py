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
        raise NotImplementedError()
    def _read_data(self,input_file):
        raise NotImplementedError()

#class NerProcessor(DataProcessor):
#    def __init__(self,args):
#        super().__init__(args)
#        self.args = args
    def get_examples(self, data_path,set_type):
        return self._create_example(
            self._read_data(data_path),set_type)



    def get_labels(self,tags_path,embeddings_type):
        """
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        
        labels = []
        if embeddings_type.lower() == 'bert':
            labels = ["[PAD]", "X","[CLS]","[SEP]"]
        if tags_path:
            with open(tags_path) as f:
                tags = f.read().splitlines() 
                labels.extend(tags)

        #return ["[PAD]","B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X","[CLS]","[SEP]"]
        return labels

    def _create_example(self, lines, set_type):
        #examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            texts = tokenization.convert_to_unicode(line[1])
            labels = line[0]
            example = InputExample(guid=guid, text=texts, label=labels)
            del guid
            del texts
            del labels
            yield example
            #examples.append(InputExample(guid=guid, text=texts, label=labels))
        #return examples


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 input_ids,
                 mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        """
        input_ids for example ner and class,if one class,input_ids = [class]
        """
        self.input_ids = input_ids
        self.mask = mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example

class FeatureExamples(DataProcessor):
    #def __init__(self,data_dir,tags_path,vocab_path,task_name,embeddings_type='other',do_lower_case=False,tokenizer=None,max_seq_length=256,args=None):
    def __init__(self,data_dir,tags_path,vocab_path,embeddings_type='other',do_lower_case=False,tokenizer=None,max_seq_length=256,args=None):
        self.name = "FeatureExamples"
        self.data_dir = data_dir
        if self.data_dir:
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
        self.label_map = {}
        if self.tags_path and self.embeddings_type:
            self.label_list = self.get_labels(self.tags_path,self.embeddings_type)
            self.num_labels = len(self.label_list)
            for (i,label) in enumerate(self.label_list):
                self.label_map[label] = i
        if self.vocab_path:
            self.vocab_table = load_vocab(self.vocab_path)
        self.FeatureExamples_params = {}
    
        if self.embeddings_type and self.vocab_path:
            if self.embeddings_type.lower() == 'bert':
                self.tokenizer = tokenization.FullTokenizer(
                vocab_file=self.vocab_path, do_lower_case=do_lower_case)
            else:
                self.tokenizer = tokenizer
        if data_dir:
            with open(data_dir+"/label2id.pkl",'wb') as w:
                pickle.dump(self.label_map,w)
        self.indices = [idx for tag,idx in self.label_map.items() if tag.strip() != 'O']
        
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
    def generate_based_convert_examples_to_features(self,examples, output_file,is_tokened,set_type='train',mode=None):
        for (ex_index, example) in enumerate(examples):
            convert_single_example = self.get_convert_single_example()
            feature,label_ids,ntokens = convert_single_example(ex_index, example,is_tokened, mode)
            yield ntokens,label_ids
    def filed_based_convert_examples_to_features(self,examples, output_file,is_tokened,set_type='train',mode=None):
        def list_to_str(l):
            return ' '.join([str(i) for i in l])
        #writer = tf.io.TFRecordWriter(output_file)
        writer = tf.TFRecordWriter(output_file)
        #batch_tokens = []
        #batch_labels = []
        #example_count = 0
        cnt = 0
        if not self.data_dir:
            raise ValueError('data dir path is none,you should specify a dir that the data will be saved ')
        sample_file = open(os.path.join(self.data_dir,'sample.txt'),'w')
        for (ex_index, example) in enumerate(examples):
            #example_count = ex_index
            if ex_index % 5000 == 0:
                self.logging.info("Writing example %d " % (ex_index))
            convert_single_example = self.get_convert_single_example()
            feature,label_ids,ntokens = convert_single_example(ex_index, example,is_tokened, mode)
    
            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            if feature.label_ids:
                try:
                    features["label_ids"] = create_int_feature(feature.label_ids)
                except Exception as e:
                    self.logging.error(e)
            else:
                self.logging.warning('feature.label_ids is None,ignore it.the label_ids is {}'.format(' '.join(ntokens)))
                continue
            #batch_tokens.append(ntokens)
            #batch_labels.extend(label_ids)
            if feature.mask:
                features["mask"] = create_int_feature(feature.mask)
            if feature.segment_ids:
                features["segment_ids"] = create_int_feature(feature.segment_ids)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            cnt += 1
            if ex_index < 500:
                sample_file.write('{}\t{}\n'.format(list_to_str(ntokens),list_to_str(label_ids)))
        return cnt
            #yield ntokens,label_ids
        # sentence token in each batch
        #if set_type == 'train':
        #    self.len_train_examples = example_count
        #    self.FeatureExamples_params['len_train_examples']=self.len_train_examples
        #    with open(self.data_dir+"/params.json",'w') as w:
        #        json.dump(self.FeatureExamples_params,w,ensure_ascii=False)
        #writer.close()
        #return batch_tokens,batch_labels
    #def save_example_tf_record(self,input_data_path,set_type='train'):
    #    examples = self.get_examples(input_data_path,set_type)
    #    if set_type == 'train':
    #        self.len_train_examples = len(examples)
    #    example_file_tf_record = os.path.join(self.data_dir, "train.tf_record")
    #    _,_ = self.filed_based_convert_examples_to_features( examples, example_file_tf_record)
    #    return example_file_tf_record
    def get_example_tf_record(self,input_data_path,is_tokened=True,is_write=True,set_type='train'):
        example_file_tf_record = os.path.join(self.data_dir, "{}.tf_record".format(set_type))
        if is_write:
            examples = self.get_examples(input_data_path,set_type)
            len_examples = self.filed_based_convert_examples_to_features( examples, example_file_tf_record,is_tokened,set_type=set_type)
            if set_type == 'train':
                self.len_train_examples = len_examples
                self.FeatureExamples_params['len_train_examples']=self.len_train_examples
                with open(self.data_dir+"/params.json",'w') as w:
                    json.dump(self.FeatureExamples_params,w,ensure_ascii=False)
        else:
            try:
                with open(self.data_dir+"/params.json",'r') as w:
                    params = json.load(w)
                    self.len_train_examples = params['len_train_examples']
            except Exception as e:
                raise ValueError('params.josn not exist,you may set is_write=True to get len_train_examples and rewrite tf_record file')
            
        return example_file_tf_record
    @staticmethod
    def convert_tokens_to_ids(vocab, tokens):
        return convert_by_vocab(vocab, tokens)
    @staticmethod
    def convert_ids_to_tokens(inv_vocab, ids):
        return convert_by_vocab(inv_vocab, ids)
    
    #def _convert_single_example(ex_index, example, label_list, self.max_seq_length, tokenizer, mode,data_dir,vocab,self.logging):
    def _convert_single_example(self,ex_index, example,is_tokened, mode):

        input_ids,mask,segment_ids,ntokens = self._convert_single_feature(ex_index,example,is_tokened,mode)
        label_ids = self._convert_single_label(ex_index,example,mode)
        feature = InputFeatures(
            input_ids=input_ids,
            mask=mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
        )
        # we need ntokens because if we do predict it can help us return to original token.
        return feature,label_ids,ntokens
            
    def _other_convert_single_example(self,ex_index, example,is_tokened, mode):
        raise NotImplementedError()
    def get_convert_single_example(self):
        if self.embeddings_type == 'other':
            return self._convert_single_example
        else:
            return self._other_convert_single_example
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
