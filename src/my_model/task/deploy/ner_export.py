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
        #label_ids = tf.placeholder(dtype=tf.int64, shape=[None, max_seq_length], name='label_ids')
    
        #receive_tensors = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids,
        #                  'label_ids': label_ids}
        #features = {'input_ids': input_ids, 'input_mask': input_mask, 'segment_ids': segment_ids, "label_ids": label_ids}
        receive_tensors = {'input_ids': input_ids}
        features = {'input_ids': input_ids}
        return tf.estimator.export.ServingInputReceiver(features, receive_tensors)


    from bert_base.bert import modeling
    from bert_base.train.bert_lstm_ner import NerProcessor
    from bert_base.train.bert_lstm_ner import model_fn_builder
    from bert_base.train.train_helper import get_args_parser
    args = get_args_parser()
    bert_config = modeling.BertConfig.from_json_file('./checkpoint1/bert_config.json')
    processor = NerProcessor('output/result_dir')
    label_list = processor.get_labels()
    import ipdb
    #ipdb.set_trace()
    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)
    run_config = tf.estimator.RunConfig(
        model_dir='output/result_dir',
        save_summary_steps=500,
        save_checkpoints_steps=500,
        session_config=session_config)
    print('len label list is {}'.format(len(label_list)))
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        #num_labels=2,
        init_checkpoint='./checkpoint1/bert_model.ckpt',
        learning_rate=2e-5,
        num_train_steps=10,
        num_warmup_steps=10,
		args=args)

    # params是一个dict 里面的key是model_fn 里面用到的参数名称，value是对应的数据
    params = {
        'batch_size': 32,
    }
    #estimator = tf.estimator.Estimator(model_fn,'output/result_dir', params=params)
    estimator = tf.estimator.Estimator(model_fn,'output/result_dir', params=params,config=run_config)
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
Bilstm_CRF_save_model('ner_deploy')


