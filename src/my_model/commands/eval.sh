my_model-cli "eval"  --task "ner" --model_name "bilstm_crf" --output ../task/ner/results/ --eval_data_dir /home/anylangtech/.userdata/yongcanpan/NER/sub_NER/bilstm_gove_preject/3.dataset_conll2003 --eval_file "testb" --batch_size 8  --buffer 15000 --dropout 0.5 --dim 300 --output output 

