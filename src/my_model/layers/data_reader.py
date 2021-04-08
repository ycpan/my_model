from pathlib import Path
import tensorflow as tf
import functools
from my_model.abc.base import MY_BASE
class DATA_READER(MY_BASE):
    def __init__(self):
        self.__name__ = 'DATA_READER'
    @staticmethod
    def parse_fn(line_words, line_tags):
        # Encode in Bytes for TF
        words = [w.encode() for w in line_words.strip().split()]
        tags = [t.encode() for t in line_tags.strip().split()]
        assert len(words) == len(tags), "Words and tags lengths don't match"
        return (words, len(words)), tags
    
    
    def generator_fn(self,words, tags):
        with Path(words).open('r') as f_words, Path(tags).open('r') as f_tags:
            for line_words, line_tags in zip(f_words, f_tags):
                yield self.parse_fn(line_words, line_tags)
    
    
    def input_fn(self, words, tags, args=None, shuffle_and_repeat=False):
        #args = args if args is not None else {}
        shapes = (([None], ()), [None])
        types = ((tf.string, tf.int32), tf.string)
        defaults = (('<pad>', 0), 'O')
    
        dataset = tf.data.Dataset.from_generator(
            functools.partial(self.generator_fn, words, tags),
            output_shapes=shapes, output_types=types)
    
        if shuffle_and_repeat:
            dataset = dataset.shuffle(args.buffer).repeat(args.epochs)
    
        dataset = (dataset
                   .padded_batch(args.batch_size, shapes, defaults)
                   .prefetch(1))
        return dataset
    
    @staticmethod
    def fwords(input_dir,name):
        return str(Path(input_dir, '{}.words.txt'.format(name)))

    @staticmethod
    def ftags(input_dir,name):
        return str(Path(input_dir, '{}.tags.txt'.format(name)))
