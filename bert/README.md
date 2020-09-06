这份代码使用 Bert 预训练新闻数据，详细的代码讲解，请查看下面 3 篇文章。



- [数据预处理](https://zhuanlan.zhihu.com/p/219698336)
- [Bert 源码讲解](https://zhuanlan.zhihu.com/p/219710200)
- [Bert 预训练与分类](https://zhuanlan.zhihu.com/p/219718670)



tensorflow 版本是 1.x。



步骤如下：


# 1. 数据准备

1.1 首先在 `bert` 文件夹里创建 `data` 文件夹，把训练数据 `train_set.csv` 和测试数据  `test_a.csv` 放到 `bert/data` 文件夹。运行 `prepare_data.py`，把文本数据放到一个文件中，每篇文章之间使用空行分隔
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

