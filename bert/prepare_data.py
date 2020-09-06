# split data to 10 fold
fold_num = 10
import os
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')
dir = 'data'
data_file = os.path.join(dir,'train_set.csv')


def all_data2fold(fold_num):
    fold_data = []
    f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
    texts = f['text'].tolist()
    labels = f['label'].tolist()

    total = len(labels)

    index = list(range(total))
    # 打乱数据
    np.random.shuffle(index)

    # all_texts 和 all_labels 都是 shuffle 之后的数据
    all_texts = []
    all_labels = []
    for i in index:
        all_texts.append(texts[i])
        all_labels.append(labels[i])

    # 构造一个 dict，key 为 label，value 是一个 list，存储的是该类对应的 index
    label2id = {}
    for i in range(total):
        label = str(all_labels[i])
        if label not in label2id:
            label2id[label] = [i]
        else:
            label2id[label].append(i)

    # all_index 是一个 list，里面包括 10 个 list，称为 10 个 fold，存储 10 个 fold 对应的 index
    all_index = [[] for _ in range(fold_num)]
    for label, data in label2id.items():
        # print(label, len(data))
        batch_size = int(len(data) / fold_num)
        # other 表示多出来的数据，other 的数据量是小于 fold_num 的
        other = len(data) - batch_size * fold_num
        # 把每一类对应的 index，添加到每个 fold 里面去
        for i in range(fold_num):
            # 如果 i < other，那么将一个数据添加到这一轮 batch 的数据中
            cur_batch_size = batch_size + 1 if i < other else batch_size
            # print(cur_batch_size)
            # batch_data 是该轮 batch 对应的索引
            batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
            all_index[i].extend(batch_data)

    batch_size = int(total / fold_num)
    other_texts = []
    other_labels = []
    other_num = 0
    start = 0

    # 由于上面在分 batch 的过程中，每个 batch 的数据量不一样，这里是把数据平均到每个 batch
    for fold in range(fold_num):
        num = len(all_index[fold])
        texts = [all_texts[i] for i in all_index[fold]]
        labels = [all_labels[i] for i in all_index[fold]]

        if num > batch_size:  # 如果大于 batch_size 那么就取 batch_size 大小的数据
            fold_texts = texts[:batch_size]
            other_texts.extend(texts[batch_size:])
            fold_labels = labels[:batch_size]
            other_labels.extend(labels[batch_size:])
            other_num += num - batch_size
        elif num < batch_size:  # 如果小于 batch_size，那么就补全到 batch_size 的大小
            end = start + batch_size - num
            fold_texts = texts + other_texts[start: end]
            fold_labels = labels + other_labels[start: end]
            start = end
        else:
            fold_texts = texts
            fold_labels = labels

        assert batch_size == len(fold_labels)

        # shuffle
        index = list(range(batch_size))
        np.random.shuffle(index)
        # 这里是为了打乱数据
        shuffle_fold_texts = []
        shuffle_fold_labels = []
        for i in index:
            shuffle_fold_texts.append(fold_texts[i])
            shuffle_fold_labels.append(fold_labels[i])

        data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
        fold_data.append(data)

    logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))

    return fold_data


# fold_data 是一个 list，有 10 个元素，每个元素是 dict，包括 label 和 text
fold_data = all_data2fold(10)
for i in range(0, 10):
    data = fold_data[i]

    path = os.path.join(dir, "train_" + str(i))
    my_open = open(path, 'w')
    # 打开文件，采用写入模式
    # 若文件不存在,创建，若存在，清空并写入
    for text in data['text']:
        my_open.write(text)
        my_open.write('\n') # 换行
        my_open.write('\n') # 添加一个空行，作为文章之间的分隔符
    logging.info("complete train_" + str(i))
    my_open.close()

