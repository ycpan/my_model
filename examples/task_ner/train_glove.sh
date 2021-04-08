#my_model-cli train --task ner --model_name bilstm_crf --data_dir /home/anylangtech/.userdata/yongcanpan/NER/sub_NER/3.dataset_conll2003 --buffer 15000 --epochs 25 --batch_size 8 --dropout 0.5 --dim 300 --output output
my_model-cli train --task ner --model_name bilstm_crf --data_dir /home/yongcanpan/MyModel/examples/conll2003 --buffer 15000 --epochs 25 --batch_size 8 --dropout 0.5 --dim 300 --output output_glove --embedding 'glove' --learning_rate 2e-5 --learning_rate_decay exp --learning_rate_peak 0.005 --decay_rate 0.96 --lstm_size 128 --num_oov_buckets 1 --decay_steps 100 --warmup_steps 1500 --vocab_words /home/anylangtech/.userdata/yongcanpan/NER/sub_NER/bilstm_glove_preject/3.dataset_conll2003/vocab.words.txt  --vocab_tags /home/yongcanpan/MyModel/examples/conll2003/tags.txt --log_dir output_glove  --max_seq_length 256 --device_map 1 --glove /home/anylangtech/.userdata/yongcanpan/NER/sub_NER/bilstm_glove_preject/3.dataset_conll2003/glove.npz

