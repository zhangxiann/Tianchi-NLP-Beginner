from collections import Counter
import pandas as pd
import os.path as osp
import os

word_counter = Counter()
# 计算每个词出现的次数
data_file = osp.join('data','train_set.csv')
f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
data = f['text'].tolist()
for text in data:
    words = text.split()

    for word in words:
        word_counter[word] += 1

words = word_counter.keys()
path = os.path.join('bert-mini', "vocab.txt")
my_open = open(path, 'w')

# 打开文件，采用写入模式
# 若文件不存在,创建，若存在，清空并写入
extra_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
my_open.writelines("\n".join(extra_tokens))

my_open.writelines("\n".join(words))

my_open.close()