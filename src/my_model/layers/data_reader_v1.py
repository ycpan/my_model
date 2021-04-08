from pathlib import Path
import tensorflow as tf
import functools
from my_model.abc.base import MY_BASE
from my_model.layers.data_process import NerProcessor
from my_model.embeddings.bert_base.bert import tokenization
import pickle
import os
import collections
#def get_input_fn(file_path):
#    return INPUT_FN.input_fn####
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

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode,output_dir,logging):
    """
    :param ex_index: example num
    :param example:
    :param label_list: all labels
    :param max_seq_length:
    :param tokenizer: WordPiece tokenization
    :param mode:
    :return: feature

    IN this part we should rebuild input sentences to the following format.
    example:[Jim,Hen,##son,was,a,puppet,##eer]
    labels: [I-PER,I-PER,X,O,O,O,X]

    """
    label_map = {}
    #here start with zero this means that "[PAD]" is zero
    for (i,label) in enumerate(label_list):
        label_map[label] = i
    with open(output_dir+"/label2id.pkl",'wb') as w:
        pickle.dump(label_map,w)
    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i,(word,label) in enumerate(zip(textlist,labellist)):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i,_ in enumerate(token):
            if i==0:
                labels.append(label)
            else:
                labels.append("X")
    # only Account for [CLS] with "- 1".
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 1)]
        labels = labels[0:(max_seq_length - 1)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    # after that we don't add "[SEP]" because we want a sentence don't have
    # stop tag, because i think its not very necessary.
    # or if add "[SEP]" the model even will cause problem, special the crf layer was used.
    #ntokens是字符，对应的数字索引是input_ids
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)

    mask = [1]*len(input_ids)
    #use zero to padding and you should
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        mask.append(0)
        segment_ids.append(0)
        label_ids.append(0)
        ntokens.append("[PAD]")
    assert len(input_ids) == max_seq_length
    assert len(mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(ntokens) == max_seq_length
    if ex_index < 3:
        logging.info("*** Example ***")
        logging.info("guid: %s" % (example.guid))
        logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
        logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
    feature = InputFeatures(
        input_ids=input_ids,
        mask=mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
    )
    # we need ntokens because if we do predict it can help us return to original token.
    return feature,ntokens,label_ids
def tf_record_input_fn_builder(input_file, seq_length, is_training, drop_remainder,batch_size):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),

    }
    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn
def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file,output_dir,logging,mode=None):
    writer = tf.python_io.TFRecordWriter(output_file)
    batch_tokens = []
    batch_labels = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature,ntokens,label_ids = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode,output_dir,logging)
        batch_tokens.extend(ntokens)
        batch_labels.extend(label_ids)
        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["mask"] = create_int_feature(feature.mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    # sentence token in each batch
    writer.close()
    return batch_tokens,batch_labels

def get_input_fn(data_dir,output_dir,vocab_file,tags,batch_size,set_type,logging,do_lower_case=False,max_seq_length=256,task_name='ner',file_name="train.tf_record"):
    processors = {"ner": NerProcessor}
    task_name = task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    label_list = processor.get_labels(tags)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)
    examples = processor.get_examples(data_dir,set_type)
    example_file = os.path.join(output_dir, "train.tf_record")
    _,_ = filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, example_file,output_dir,logging)
    input_fn = tf_record_input_fn_builder(
        input_file=example_file,
        seq_length=max_seq_length,
        is_training=True,
        drop_remainder=True,
        batch_size=batch_size)
    return input_fn,label_list,len(examples)
#class INPUT_FN:
#class DATA_READER(MY_BASE):
#    def __init__(self):
#        self.__name__ = 'DATA_READER'
#    @abstractmethod
#    def convert_single_example():
#        raise NotImplementedError('this should be implied')
#    
#
#    @staticmethod
#    def parse_fn(line_words, line_tags):
#        # Encode in Bytes for TF
#        words = [w.encode() for w in line_words.strip().split()]
#        tags = [t.encode() for t in line_tags.strip().split()]
#        assert len(words) == len(tags), "Words and tags lengths don't match"
#        return (words, len(words)), tags
#    
#    
#    def generator_fn(self,words, tags):
#        with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
#            for line_words, line_tags in zip(f_words, f_tags):
#                yield self.parse_fn(line_words, line_tags)
#    
#    
#    def input_fn():
#        input_fn = get_input_fn()
#    def old_input_fn(self, words, tags, args=None, shuffle_and_repeat=False):
#        #args = args if args is not None else {}
#        shapes = (([None], ()), [None])
#        types = ((tf.string, tf.int32), tf.string)
#        defaults = (('<pad>', 0), 'O')
#    
#        dataset = tf.data.Dataset.from_generator(
#            functools.partial(self.generator_fn, words, tags),
#            output_shapes=shapes, output_types=types)
#    
#        if shuffle_and_repeat:
#            dataset = dataset.shuffle(args.buffer).repeat(args.epochs)
#    
#        dataset = (dataset
#                   .padded_batch(args.batch_size, shapes, defaults)
#                   .prefetch(1))
#        return dataset
#    
#    @staticmethod
#    def fwords(input_dir,name):
#        return str(Path(input_dir, '{}.words.txt'.format(name)))
#
#    @staticmethod
#    def ftags(input_dir,name):
#        return str(Path(input_dir, '{}.tags.txt'.format(name)))
