mkdir create

nohup python create_pretraining_data.py --input_file=./data/train_0 --output_file=./records/train_0.tfrecord --vocab_file=./bert-mini/vocab.txt --max_seq_length=256 --max_predictions_per_seq=32 --do_lower_case=True --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5> create/0.log 2>&1 &

nohup python create_pretraining_data.py --input_file=./data/train_1 --output_file=./records/train_1.tfrecord --vocab_file=./bert-mini/vocab.txt --max_seq_length=256 --max_predictions_per_seq=32 --do_lower_case=True --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5> create/1.log 2>&1 &

nohup python create_pretraining_data.py --input_file=./data/train_2 --output_file=./records/train_2.tfrecord --vocab_file=./bert-mini/vocab.txt --max_seq_length=256 --max_predictions_per_seq=32 --do_lower_case=True --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5> create/2.log 2>&1 &

nohup python create_pretraining_data.py --input_file=./data/train_3 --output_file=./records/train_3.tfrecord --vocab_file=./bert-mini/vocab.txt --max_seq_length=256 --max_predictions_per_seq=32 --do_lower_case=True --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5> create/3.log 2>&1 &

nohup python create_pretraining_data.py --input_file=./data/train_4 --output_file=./records/train_4.tfrecord --vocab_file=./bert-mini/vocab.txt --max_seq_length=256 --max_predictions_per_seq=32 --do_lower_case=True --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5> create/4.log 2>&1 &

nohup python create_pretraining_data.py --input_file=./data/train_5 --output_file=./records/train_5.tfrecord --vocab_file=./bert-mini/vocab.txt --max_seq_length=256 --max_predictions_per_seq=32 --do_lower_case=True --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5> create/5.log 2>&1 &

nohup python create_pretraining_data.py --input_file=./data/train_6 --output_file=./records/train_6.tfrecord --vocab_file=./bert-mini/vocab.txt --max_seq_length=256 --max_predictions_per_seq=32 --do_lower_case=True --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5> create/6.log 2>&1 &

nohup python create_pretraining_data.py --input_file=./data/train_7 --output_file=./records/train_7.tfrecord --vocab_file=./bert-mini/vocab.txt --max_seq_length=256 --max_predictions_per_seq=32 --do_lower_case=True --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5> create/7.log 2>&1 &

nohup python create_pretraining_data.py --input_file=./data/train_8 --output_file=./records/train_8.tfrecord --vocab_file=./bert-mini/vocab.txt --max_seq_length=256 --max_predictions_per_seq=32 --do_lower_case=True --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5> create/8.log 2>&1 &

nohup python create_pretraining_data.py --input_file=./data/train_9 --output_file=./records/train_9.tfrecord --vocab_file=./bert-mini/vocab.txt --max_seq_length=256 --max_predictions_per_seq=32 --do_lower_case=True --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5> create/9.log 2>&1 &