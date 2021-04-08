import sys
import tensorflow as tf
#sys.path.append('')
#from ModelLayer.Bilstm_CRF import Bilstm_CRF
#from ModelLayer.Bilstm_CRF_bak import Bilstm_CRF
#
# params = {
#     'dim': 256,
#     'dropout': 0.2,
#     'num_oov_buckets': 1,
#     'epochs': 1,
#     # 'batch_size': 20,
#     'batch_size': 100,
#     'buffer': 15000,
#     # 'lstm_size': 100,
#     'lstm_size': 512,
#     'learnning_rate': 0.0002
# }
def Bilstm_CRF_save_model(save_path):

    #def serving_input_receiver_fn():
    #    """Serving input_fn that builds features from placeholders

    #    Returns
    #    -------
    #    tf.estimator.export.ServingInputReceiver
    #    """
    #    words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    #    nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    #    receiver_tensors = {'words': words, 'nwords': nwords}
    #    features = {'words': words, 'nwords': nwords}
    #    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)
    def serving_input_receiver_fn(max_seq_length=128):
        input_ids = tf.placeholder(dtype=tf.int64, shape=[None, max_seq_length], name='input_ids')
        #input_mask = tf.placeholder(dtype=tf.int64, shape=[None, max_seq_length], name='input_mask')
        #segment_ids = tf.placeholder(dtype=tf.int64, shape=[None, max_seq_length], name='segment_ids')
        #label_ids = tf.placeholder(dtype=tf.int64, shape=[None, ], name='unique_ids')
    
        #receive_tensors = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids,
        #                   'label_ids': label_ids}
        #features = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, "label_ids": label_ids}
        receive_tensors = {'input_ids': input_ids}
        features = {'input_ids': input_ids}
        return tf.estimator.export.ServingInputReceiver(features, receive_tensors)


    from bert_base.bert import modeling
    from thu_classification import ThuProcessor
    from thu_classification import model_fn_builder
    bert_config = modeling.BertConfig.from_json_file('./checkpoint1/bert_config.json')
    processor = ThuProcessor()
    label_list = processor.get_labels()
    print('len label list is {}'.format(len(label_list)))
    model_fn = model_fn_builder(
        bert_config=bert_config,
        #num_labels=len(label_list),
        num_labels=2,
        init_checkpoint='./checkpoint1/bert_model.ckpt',
        learning_rate=2e-5,
        num_train_steps=10,
        num_warmup_steps=10)

    # params是一个dict 里面的key是model_fn 里面用到的参数名称，value是对应的数据
    params = {
        'batch_size': 32,
    }
    estimator = tf.estimator.Estimator(model_fn,'class_output/result_dir', params=params)
    estimator.export_saved_model('{}/saved_model'.format(save_path), serving_input_receiver_fn)

#params = {
#    'dim': 300,
#    'dropout': 0.5,
#    'num_oov_buckets': 1,
#    'epochs': 25,
#    'batch_size': 200,
#    # 'batch_size': 100,
#    'buffer': 15000,
#    # 'lstm_size': 100,
#    'lstm_size': 150,
#    'learnning_rate': 0.001
#}
#md = None

#md = Bilstm_CRF(gpu='1,2,3,4,0', input_dir='../3.dataset', params=params)
#md = Bilstm_CRF(gpu=' ', input_dir='../3.dataset', params=params)
#md.embeding='glove'
Bilstm_CRF_save_model('deploy')


