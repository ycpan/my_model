from bert.modeling import BertModel
class TeacherModel(BertModel):
    def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=True,
               scope=None):
        super().__init__(config,is_training,input_ids,input_mask=None,token_type_ids=None,use_one_hot_embeddings=True,scope=None)
    def get_heads(self):
        return self.
