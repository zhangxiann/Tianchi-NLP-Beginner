# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random

import tensorflow as tf

import tokenization
import os

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

# 把所有的 instance 保存到 output_files 中
def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        # 把 token 转为 id
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length
        # 补全为 0，补到长度为 max_seq_length
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        # weight 默认都是 1
        masked_lm_weights = [1.0] * len(masked_lm_ids)
        # 补全为 0，补到长度为 max_predictions_per_seq，表示每句话 mask 的数量
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        # 这里是 0
        next_sentence_label = 1 if instance.is_random_next else 0

        features = collections.OrderedDict()
        # 这是为了保存到文件中
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1
        # 打印前 20 句话的内容
        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


# 输入是文件列表，输出是 instance 的列表
def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    # all_documents 是一个 list，每个元素是一篇文章
    # 文章中的是一个 list，list中每个元素是一个单词
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                # 如果是空行，作为文章之间的分隔符，那么使用一个新的 list 来存储文章
                if not line:
                    all_documents.append([])
                # 把每篇文章转换为 list
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    # 去除空行
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)
    # vocab_words 是一个 list，每个元素是单词 (word)
    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    # 对文档重复 dupe_factor 次，随机产生训练集
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document_nsp(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_segments_from_document(document, max_segment_length):
    """Split single document to segments according to max_segment_length."""
    assert len(document) == 1
    document = document[0]
    document_len = len(document)

    # index 是一个数组，表示每句话的索引
    index = list(range(0, document_len, max_segment_length))
    # other_len 表示最后剩下的文本
    other_len = document_len % max_segment_length
    # 如果最后剩下的文本，那么把最后一个位置的索引，添加到 index 中
    if other_len > max_segment_length / 2:
        index.append(document_len)

    # segments 保存的是一篇文章的句子列表
    segments = []
    # 根据索引分割称为句子
    for i in range(len(index) - 1):
        segment = document[index[i]: index[i+1]]
        segments.append(segment)

    return segments

# NSP: next sentence prediction
# 返回一篇文章的 TrainingInstance 的列表
# TrainingInstance 是一个 class，表示一个句子的
def create_instances_from_document_nsp(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document. Remove NSP."""
    # 取出 index 对应的文章
    document = all_documents[document_index]

    # 留出 2 个位置给 [CLS], [SEP]
    # Account for [CLS], [SEP]
    max_segment_length = max_seq_length - 2

    instances = []
    # document 是一个 list，只有一个元素，表示一篇文章
    segments = create_segments_from_document(document, max_segment_length)
    for j, segment in enumerate(segments):
        is_random_next = False
        # tokens 是添加了 [CLS] 和 [SEP] 的句子
        tokens = []
        # 表示句子的 id，都是 0，表示只有 1 个句子
        segment_ids = []
        # 在开头添加 [CLS]
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in segment:
            tokens.append(token)
            segment_ids.append(0)
        # 在开头添加 [SEP]
        tokens.append("[SEP]")
        # 添加句子 id:0
        segment_ids.append(0)
        # 对一个句子进行 mask ，获得 mask 后的 tokens，mask 的位置以及真实的 token(label)
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
            tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        # TrainingInstance 表示一个句子的内容
        instance = TrainingInstance(
            tokens=tokens, # 经过 mask 的 token 列表
            segment_ids=segment_ids, # 句子 id
            is_random_next=is_random_next, # False
            masked_lm_positions=masked_lm_positions, # mask 的位置列表
            masked_lm_labels=masked_lm_labels) # mask 的真实 token
        instances.append(instance)

    return instances



# all_documents 表示所有文章
def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    # document 表示一篇文章，包括多个句子
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    #
    target_seq_length = max_num_tokens
    # 产生一个 (2, max_num_tokens) 之间的数，作为实际的句子长度
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    # current_chunk 表示训练集的候选集，从中分割为 2 部分
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
    # 遍历所有的句子
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        # 当多个句子的长度和 current_length 大于 target_seq_length 时
        if i == len(document) - 1 or current_length >= target_seq_length:
            # 如果 current_chunk 不是空
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                # a_end 默认为 1，表示第第一段文本 tokens_a 有几句话，
                a_end = 1
                # 如果 current_chunk 的数量大于 2，那么随机选择前几句作为 tokens_a
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                # 把 a_end 个句子添加到 tokens_a
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                # tokens_b 表示第二段文本
                tokens_b = []
                # Random next
                is_random_next = False
                # 如果只有一句话，选择随机句子的第二段文本
                # 或者 50% 的概率，选择随机句子的第二段文本
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    # target_b_length 表示第二段文本的长度
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    # 随机选择一篇文章
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break
                    # 随机选择一篇文章
                    random_document = all_documents[random_document_index]
                    # 随机选择一篇文章的开始位置
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        # 把一段随机的文本，添加到 tokens_b
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    # 当前文章实际用到的句子数量
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                # 把真正的第二段文本，添加到 tokens_b
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                # 对两段文本进行长度剪裁
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                # 第一段的句子 id
                segment_ids.append(0)
                # 添加第一段文本
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)
                # 添加第二段文本
                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                # 第二段的句子 id
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

# tokens 是一个句子
# 返回 mask 后的 tokens，mask 的位置以及真实的 token(label)
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        # token.startswith("##") 是应对 word piece tokenization
        # 如果一个 token 以 ## 开头，那么表示 这个 token 是属于和前一个 token 是同一个单词里的
        # 需要添加到前一个 token 对应 list
        if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        # 由于我们使用的 word tokenization，因此只会执行这个 else 的语句
        else: # 只会执行这个条件，添加的是一个 list，里面只有一个元素
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)
    # 计算实际上 mask 的数量
    # 取 max_predictions_per_seq  和(句子长度 * masked_lm_prob) 的较小值。
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        # 如果 mask 的数量大于 num_to_predict，就停止
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        # index_set 是一个 list，里面只有一个元素
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        # 如果已经包含了这些 index，那么就跳过
        if is_any_index_covered:
            continue

        for index in index_set:
            # covered_indexes 是一个 set
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            # 80% 的概率替换为 mask
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 剩下的 20%，再分为两半
                # 10% of the time, keep original
                # 20% *0.5 =10% 的概率保留原来的 token
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    # 20% *0.5 =10% 的概率替换为随机的一个 token
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    # 执行完循环后，mask 的数量小于 num_to_predict
    assert len(masked_lms) <= num_to_predict
    # 根据 index 排序
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    # masked_lm_positions 保存 mask 的位置
    masked_lm_positions = []
    # masked_lm_labels 保存 mask 的真实 token
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # 创建保存的文件夹
    dirs = 'records'

    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # tokenizer以vocab_file为词典，将词转化为该词对应的id。
    tokenizer = tokenization.WhitespaceTokenizer(
        vocab_file=FLAGS.vocab_file)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        # tf.gfile.Glob: 查找匹配 filename 的文件并以列表的形式返回，
        # filename 可以是一个具体的文件名，也可以是包含通配符的正则表达式。
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    rng = random.Random(FLAGS.random_seed)
    # instances 是一个 list，每个元素是 TrainingInstance，表示一个经过mask 后的句子，
    # TrainingInstance 的内容包括：
    # tokens: 经过 mask 的 tokens
    # segment_ids: 句子 id
    # is_random_next: True 表示 tokens 的第二句是随机查找，False 表示第二句为第一句的下文。
    # 这里不使用 NSP，只有一句话，因此为 false
    # masked_lm_positions: mask 的位置
    # masked_lm_labels: mask 的真实 token
    instances = create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()