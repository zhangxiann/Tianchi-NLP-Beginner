这份代码使用 Bert 预训练新闻数据，详细的代码讲解，请查看[]()


步骤如下：


# 1. 数据准备

1.1 首先运行 `prepare_data.py`，把文本数据放到一个文件中，每篇文章之间使用空行分隔
```
python prepare_data.py
```

1.2 运行`create_vocab.py`，创建字典
```
python create_vocab.py
```

# 2. 对数据进行 MASK
```
bash create_pretraining_data.sh
```

# 3. 开始训练 Bert
```
bash run_pretraining.sh
```

# 4. 把 Tensorflow 的模型，转换为 PyTorch 的模型
```
bash convert_checkpoint.sh
```

# 5. 微调 Bert 模型，进行文本分类
```
python finetune_bert.py
```

