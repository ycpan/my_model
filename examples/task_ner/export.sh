my_model-cli export --task ner --model_name bilstm_crf --checkpoint_path output/model/ --save_path server/large_copa/saved_model --dropout 0.5 --vocab_words /home/anylangtech/.userdata/yongcanpan/gitdir/data_augumentation/dataset/large_corpra/3.dataset/vocab.words.txt  --vocab_tags /home/anylangtech/.userdata/yongcanpan/gitdir/data_augumentation/dataset/large_corpra/3.dataset/vocab.tags.txt --dim 300 --embedding 'glove' --glove /home/anylangtech/.userdata/yongcanpan/gitdir/data_augumentation/dataset/large_corpra/3.dataset/glove.npz
