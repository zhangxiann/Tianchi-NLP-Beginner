这是我参加天池**零基础入门NLP - 新闻文本分类**的代码仓库。
比赛地址：[https://tianchi.aliyun.com/competition/entrance/531810/information](https://tianchi.aliyun.com/competition/entrance/531810/information)
这是来源于 Datawhale 的开源代码。由于原来的代码没有注释，也没有整个流程的讲解，对于新手不太友好。

因此，为了减少对于新手的阅读难度，我添加了一些内容。包括**数据处理**和**模型**，以及详细的代码注释，希望能帮助到有需要的人。



包括 2 个分支：

- master：使用 tensorflow 1.x
- tensorflow2：使用 tensorflow 2.x



包括 2 个文件夹：

- textcnn：使用 TextCNN 模型的代码。讲解文章：[阿里天池 NLP 入门赛 TextCNN 方案流程讲解](https://zhuanlan.zhihu.com/p/183862056)

- bert：使用 Bert 模型的代码。讲解文章：[阿里天池 NLP 入门赛 Bert 方案代码流程讲解](https://zhuanlan.zhihu.com/p/219698336)。

  包括 Bert 预训练和微调 Bert 两大步，具体步骤如下：

1. **数据准备**

    1.1 首先在 `bert` 文件夹里创建 `data` 文件夹，把训练数据 `train_set.csv` 和测试数据  `test_a.csv` 放到 `bert/data` 文件夹。运行 `prepare_data.py`，把文本数据放到一个文件中，每篇文章之间使用空行分隔
  
	```
    python prepare_data.py
	```
	
	
	

    1.2 运行`create_vocab.py`，创建字典
	```
	python create_vocab.py
	```

2. **对数据进行 MASK**
   
    ```
      bash create_pretraining_data.sh
    ```

3. **开始训练 Bert**
	```
      bash run_pretraining.sh
   ```

4. **把 Tensorflow 的模型，转换为 PyTorch 的模型**
    ```
      bash convert_checkpoint.sh
    ```

5. **微调 Bert 模型，进行文本分类**
    ```
    python finetune_bert.py
    ```
