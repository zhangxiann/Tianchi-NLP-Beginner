python run_pretraining.py --input_file=./records/*.tfrecord  --output_dir=./bert-mini --do_train=True --do_eval=True --bert_config_file=./bert-mini/bert_config.json --train_batch_size=128 --eval_batch_size=128 --max_seq_length=256 --max_predictions_per_seq=32 --learning_rate=1e-4

# 后台运行
# nohup python run_pretraining.py --input_file=./records/*.tfrecord  --output_dir=./bert-mini --do_train=True --do_eval=True --bert_config_file=./bert-mini/bert_config.json --train_batch_size=128 --eval_batch_size=128 --max_seq_length=256 --max_predictions_per_seq=32 --learning_rate=1e-4 > pre_train.log 2>&1 &