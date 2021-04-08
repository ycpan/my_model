my_model-cli "eval"  --task "ner" --model_name "bilstm_crf"   --data_dir /home/anylangtech/.userdata/yongcanpan/gitdir/data_augumentation/dataset/large_corpra/3.dataset  --output output --eval_file "testb" --batch_size 8  --buffer 15000 --dropout 0.5 --embedding 'glove' --dim 300 --log_dir output --log_name 'eval'
#--output /home/anylangtech/.userdata/yongcanpan/gitdir/data_augumentation/dataset/large_corpra/output 

