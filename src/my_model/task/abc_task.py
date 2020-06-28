from my_model.abc.model_abc import ABC_MODEL
def get_task_feature(vocab_path,embeddings_type='other',tokenizer=None,max_seq_length=512,args=None):
    data_dir = None
    tags_path=None
    fe = ABCTask(data_dir,'',tags_path,vocab_path,embeddings_type=embeddings_type,tokenizer=tokenizer,max_seq_length=max_seq_length,args=args)
    return fe.get_feature
class ABCTask(ABC_MODEL):

    def get_label_ids(self,label:list):
        raise NotImplementedError()
    def _convert_single_label(self,ex_index,example,mode):
        label = example.label
        label_ids = self.get_label_ids(label)
        #label_ids = [self.label_map.get(la,0) for la in label]
        if ex_index < 3:
            self.logging.info("*** Example ***")
            self.logging.info("guid: %s" % (example.guid))
            if label_ids:
                self.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        return label_ids
    def get_feature(self,text:str,is_tokened=False):
        if not is_tokened:
            text = self.tokenizer(text)
            text = ' '.join(text)
        textlist = text.split(' ')
        ntokens = textlist
        if len(ntokens) >= self.max_seq_length - 1:
            ntokens = ntokens[0:(self.max_seq_length - 1)]
        input_ids = self.convert_tokens_to_ids(self.vocab_table,ntokens)
        mask = [1]*len(input_ids)
        #use zero to padding and you should
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            mask.append(0)
        return ntokens,input_ids,mask
        
    def _convert_single_feature(self,ex_index,example,is_tokened,mode):

        #here start with zero this means that "[PAD]" is zero
        #textlist = example.text.split(' ')
        #ntokens = textlist
        #if len(ntokens) >= self.max_seq_length - 1:
        #    ntokens = ntokens[0:(self.max_seq_length - 1)]
        #input_ids = self.convert_tokens_to_ids(self.vocab_table,ntokens)
            #segment_ids.append(0)
        ntokens,input_ids,mask = self.get_feature(example.text,is_tokened=is_tokened)
        assert len(input_ids) == self.max_seq_length
        assert len(mask) == self.max_seq_length
        #assert len(segment_ids) == self.max_seq_length
        #assert len(ntokens) == self.max_seq_length
        if ex_index < 3:
            self.logging.info("*** Example ***")
            self.logging.info("guid: %s" % (example.guid))
            self.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            self.logging.info("input_mask: %s" % " ".join([str(x) for x in mask]))
            #self.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        return input_ids,mask,None,ntokens
