#my_model-cli train --task ner --model_name bilstm_crf --data_dir /home/anylangtech/.userdata/yongcanpan/NER/sub_NER/3.dataset_conll2003 --buffer 15000 --epochs 25 --batch_size 8 --dropout 0.5 --dim 300 --output output
my_model-cli train --task ner --model_name bilstm_crf --data_dir /home/anylangtech/.userdata/yongcanpan/gitdir/data_augumentation/dataset/large_corpra/3.dataset --buffer 15000 --epochs 25 --batch_size 8 --dropout 0.5 --dim 300 --output output --embedding 'glove' --learning_rate_decay exp --learning_rate_peak 0.005 --decay_rate 0.96 --embedding_stddev 0.05 --lstm_size 128 --num_oov_buckets 1 --decay_steps 100 --warmup_steps 1500 --vocab_words /home/anylangtech/.userdata/yongcanpan/gitdir/data_augumentation/dataset/large_corpra/3.dataset/vocab.words.txt  --vocab_tags /home/anylangtech/.userdata/yongcanpan/gitdir/data_augumentation/dataset/large_corpra/3.dataset/vocab.tags.txt --log_dir output