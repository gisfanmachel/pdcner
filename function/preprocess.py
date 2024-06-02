# -*- coding: utf-8 -*-
# 文件包含了数据预处理的函数，这些函数主要用于将原始文本数据转换成模型可以处理的格式，以及构建词典树等任务
# 这些函数共同支持数据的预处理工作，包括词汇匹配、词典树的构建以及标签的提取等，为模型的训练和评估准备好格式化的数据输入。
import time
import os
import json
from tqdm import tqdm, trange
from module.lexicon_tree import Trie



# 这个函数接收一个句子和词典树作为输入，返回每个字对应的匹配词以及词边界。边界类型包括词的开始(B-)、中间(M-)、结尾(E-)、单字词(S-)，以及它们的组合形式。
#
# 参数:
#
# sent: 输入的句子，以字为单位的数组。
# lexicon_tree: 词典树对象。
# max_word_num: 最多匹配的词的数量。
# 返回值:
#
# sent_words: 句子中每个字归属的词组。
# sent_boundaries: 句子中每个字所属的边界类型。
def sent_to_matched_words_boundaries(sent, lexicon_tree, max_word_num=None):
    """
    输入一个句子和词典树, 返回句子中每个字所属的匹配词, 以及该字的词边界
    字可能属于以下几种边界:
        B-: 词的开始, 0
        M-: 词的中间, 1
        E-: 词的结尾, 2
        S-: 单字词, 3
        BM-: 既是某个词的开始, 又是某个词中间, 4
        BE-: 既是某个词开始，又是某个词结尾, 5
        ME-: 既是某个词的中间，又是某个词结尾, 6
        BME-: 词的开始、词的中间和词的结尾, 7

    Args:
        sent: 输入的句子, 一个字的数组
        lexicon_tree: 词典树
        max_word_num: 最多匹配的词的数量
    Args:
        sent_words: 句子中每个字归属的词组
        sent_boundaries: 句子中每个字所属的边界类型
    """
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]
    sent_boundaries = [[] for _ in range(sent_length)]  # each char has a boundary

    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        if len(words) == 0 and len(sent_boundaries[idx]) == 0:
            sent_boundaries[idx].append(3) # S-
        else:
            if len(words) == 1 and len(words[0]) == 1: # single character word
                if len(sent_words[idx]) == 0:
                    sent_words[idx].extend(words)
                    sent_boundaries[idx].append(3) # S-
            else:
                if max_word_num:
                    need_num = max_word_num - len(sent_words[idx])
                    words = words[:need_num]
                sent_words[idx].extend(words)
                for word in words:
                    if 0 not in sent_boundaries[idx]:
                        sent_boundaries[idx].append(0) # S-
                    start_pos = idx + 1
                    end_pos = idx + len(word) - 1
                    for tmp_j in range(start_pos, end_pos):
                        if 1 not in sent_boundaries[tmp_j]:
                            sent_boundaries[tmp_j].append(1) # M-
                        sent_words[tmp_j].append(word)
                    if 2 not in sent_boundaries[end_pos]:
                        sent_boundaries[end_pos].append(2) # E-
                    sent_words[end_pos].append(word)

    assert len(sent_words) == len(sent_boundaries)

    new_sent_boundaries = []
    idx = 0
    for boundary in sent_boundaries:
        if len(boundary) == 0:
            print("Error")
            new_sent_boundaries.append(0)
        elif len(boundary) == 1:
            new_sent_boundaries.append(boundary[0])
        elif len(boundary) == 2:
            total_num = sum(boundary)
            new_sent_boundaries.append(3 + total_num)
        elif len(boundary) == 3:
            new_sent_boundaries.append(7)
        else:
            print(boundary)
            print("Error")
            new_sent_boundaries.append(8)
    assert len(sent_words) == len(new_sent_boundaries)

    return sent_words, new_sent_boundaries

# 这个函数用于获取句子的匹配词，并按照BMES（Begin, Middle, End, Single）进行分组。
#
# 参数:
#
# sent: 一个字的数组。
# lexicon_tree: 词汇表树。
# max_word_num: 最大词数。
# 返回值:
#
# sent_words: 包含BMES分组的列表。
# sent_group_mask: 标记每个分组是否包含词
def sent_to_distinct_matched_words(sent, lexicon_tree):
    """
    得到句子的匹配词, 并进行分组, 按照BMES进行分组
    Args:
        sent: 一个字的数组
        lexicon_tree: 词汇表树
        max_word_num: 最大词数
    """
    sent_length = len(sent)
    sent_words = [[[], [], [], []] for _ in range(sent_length)] # 每个字都有对应BMES
    sent_group_mask = [[0, 0, 0, 0] for _ in range(sent_length)]

    for idx in range(sent_length):
        sub_sent = sent[idx:idx+lexicon_tree.max_depth]
        words = lexicon_tree.enumerateMatch(sub_sent)
        if len(words) == 0:
            continue
        else:
            for word in words:
                word_length = len(word)
                if word_length == 1:
                    sent_words[idx][3].append(word)
                    sent_group_mask[idx][3] = 1
                else:
                    sent_words[idx][0].append(word) # begin
                    sent_group_mask[idx][0] = 1
                    for pos in range(1, word_length-1):
                        sent_words[idx+pos][1].append(word) # middle
                    sent_words[idx+word_length-1][2].append(word) # end
        if len(sent_words[idx][1]) > 0:
            sent_group_mask[idx][1] = 1
        if len(sent_words[idx][2]) > 0:
            sent_group_mask[idx][2] = 1

    return sent_words, sent_group_mask

# 此函数与 sent_to_matched_words_boundaries 类似，但只返回匹配的词，不包含边界信息。
def sent_to_matched_words(sent, lexicon_tree, max_word_num=None):
    """same to sent_to_matched_words_boundaries, but only return words"""
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]

    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        if len(words) == 0:
            continue
        else:
            if len(words) == 1 and len(words[0]) == 1: # single character word
                if len(sent_words[idx]) == 0:
                    sent_words[idx].extend(words)
            else:
                if max_word_num:
                    need_num = max_word_num - len(sent_words[idx])
                    words = words[:need_num]
                sent_words[idx].extend(words)
                for word in words:
                    start_pos = idx + 1
                    end_pos = idx + len(word) - 1
                    for tmp_j in range(start_pos, end_pos):
                        sent_words[tmp_j].append(word)
                    sent_words[end_pos].append(word)

    return sent_words

# 这个函数返回句子中所有匹配词的集合。
def sent_to_matched_words_set(sent, lexicon_tree, max_word_num=None):
    """return matched words set"""
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]
    matched_words_set = set()
    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]  # speed using max depth
        words = lexicon_tree.enumerateMatch(sub_sent)

        _ = [matched_words_set.add(word) for word in words]
    matched_words_set = list(matched_words_set)
    matched_words_set = sorted(matched_words_set)
    return matched_words_set

# 这个函数用于从数据文件和词汇文件中获取匹配的词。
#
# 参数:
#
# files: 输入数据文件。
# vocab_files: 输入词汇文件。
# scan_nums: 要扫描的词汇文件数量。
# 返回值:
#
# total_matched_words: 所有找到的匹配词。
# lexicon_tree: 构建的词典树。

def get_corpus_matched_word_from_vocab_files(files, vocab_files, scan_nums=None):
    """
    the corpus's matched words from vocab files
    Args:
        files: input data files
        vocab_files: input vocab files
        scan_num: -1 total,
    Returns:
        total_matched_words:
        lexicon_tree:
    """
    # 1.获取词汇表
    vocabs = set()
    if scan_nums is None:
        length = len(vocab_files)
        scan_nums = [-1] * length

    for file, need_num in zip(vocab_files, scan_nums):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines)
            if need_num >= 0:
                total_line_num = min(total_line_num, need_num)

            line_iter = trange(total_line_num)
            for idx in line_iter:
                line = lines[idx]
                line = line.strip()
                items = line.split()
                word = items[0].strip()
                vocabs.add(word)
    vocabs = list(vocabs)
    vocabs = sorted(vocabs)
    # 2.建立词典树
    lexicon_tree = Trie()
    for word in vocabs:
        lexicon_tree.insert(word)

    total_matched_words = get_corpus_matched_word_from_lexicon_tree(files, lexicon_tree)
    return total_matched_words, lexicon_tree

# 此函数接收数据文件和词典树，返回所有匹配的词。
def get_corpus_matched_word_from_lexicon_tree(files, lexicon_tree):
    """
    数据类型统一为json格式, {'text': , 'label': }
    Args:
        files: corpus data files
        lexicon_tree: built lexicon tree

    Return:
        total_matched_words: all found matched words
    """
    total_matched_words = set()
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines)
            line_iter = trange(total_line_num)
            for idx in line_iter:
                line = lines[idx]
                line = line.strip()

                sample = json.loads(line)
                if 'text' in sample:
                    text = sample['text']
                elif 'text_a' in sample and 'text_b' in sample:
                    text_a = sample['text_a']
                    text_b = sample['text_b']
                    text = text_a + ["[SEP]"] + text_b
                sent = [ch for ch in text]
                sent_matched_words = sent_to_matched_words_set(sent, lexicon_tree)
                _ = [total_matched_words.add(word) for word in sent_matched_words]

    total_matched_words = list(total_matched_words)
    total_matched_words = sorted(total_matched_words)
    with open("matched_word.txt", "w", encoding="utf-8") as f:
        for word in total_matched_words:
            f.write("%s\n"%(word))

    return total_matched_words

# 这个函数通过查找分词词汇表和全量词词汇表的交集，将交集词插入到词典树中。
#
# 参数:
#
# seg_vocab: 分词词汇表文件。
# word_vocab: 全量的词文件。
# lexicon_tree: 词典树。

def insert_seg_vocab_to_lexicon_tree(seg_vocab, word_vocab, lexicon_tree):
    """
    通过查找seg_vocab和word_vocab的重合词, 将重合词插入到lexicon_tree里面
    Args:
        seg_vocab: seg_vocab中的词文件
        word_vocab: 全量的词文件
        lexicon_tree:
    """
    seg_words = set()
    whole_words = set()
    with open(seg_vocab, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_line_num = len(lines)
        line_iter = trange(total_line_num)

        for idx in line_iter:
            line = lines[idx]
            line = line.strip()
            if line:
                seg_words.add(line)

    with open(word_vocab, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_line_num = len(lines)
        line_iter = trange(total_line_num)

        for idx in line_iter:
            line = lines[idx]
            line = line.strip()
            if line:
                whole_words.add(line)

    overleap_words = seg_words & whole_words
    overleap_words = list(overleap_words)
    overleap_words = sorted(overleap_words)
    print("Overleap words number is: \n", len(overleap_words))

    for word in overleap_words:
        lexicon_tree.insert(word)

    return lexicon_tree

# 此函数用于从词汇文件构建词典树。
#
# 参数:
#
# vocab_files: 词汇表文件列表。
# scan_nums: 每个文件要扫描的行数。
# 返回值:
#
# lexicon_tree: 构建的词典树。
def build_lexicon_tree_from_vocabs(vocab_files, scan_nums=None):
    # build_lexicon_tree_from_vocabs 方法的原理是构建一个词典树（Trie 树），这是一种用于存储字符串数据集以进行快速检索的数据结构。在自然语言处理（NLP）中，词典树通常用于快速匹配和检索文本中的词或短语。
    #
    # 词典树是一种特殊的树形结构，其中每个节点代表一个字符，从根节点到某一节点的路径表示一个词。以下是 build_lexicon_tree_from_vocabs 方法的工作原理：
    #
    # 初始化 Trie 树:
    #
    # 创建一个 Trie 树的根节点，这个根节点不包含任何字符，它的功能是作为树的起始点。
    # 读取词汇文件:
    #
    # 遍历传入的词汇文件列表，每个文件代表一个词汇表。
    # 构建 Trie 节点:
    #
    # 对于每个文件中的每行（代表一个词），从根节点开始，为每个字符创建子节点。
    # 每个字符都是 Trie 树中的一个节点，如果一个字符在树中不存在，则创建一个新的节点。
    # 插入词汇:
    #
    # 将每个词插入 Trie 树中。这意味着按照词中的字符顺序，从根节点开始，为每个字符创建或找到相应的子节点。
    # 如果字符是词的最后一个字符，则将该节点标记为词的结束（is_word = True）。
    # 维护最大深度:
    #
    # 更新 Trie 树的最大深度，这通常是最长词的长度。
    # 返回 Trie 树:
    #
    # 完成所有词汇的插入后，返回构建好的 Trie 树。
    # 词典树的这种结构使得它在词频统计、自动补全、拼写检查等任务中非常有用。在中文NLP任务中，词典树可以用来快速检索句子中的词，以及它们的边界信息，这对于诸如命名实体识别（NER）等任务至关重要。
    #
    # 在 preprocess.py 文件中，build_lexicon_tree_from_vocabs 方法通过构建词典树，为后续的句子处理提供了快速匹配已知词汇的能力，这有助于提高诸如实体识别等任务的效率和准确性。

    # 1.获取词汇表
    print(vocab_files)
    vocabs = set()
    if scan_nums is None:
        length = len(vocab_files)
        scan_nums = [-1] * length

    for file, need_num in zip(vocab_files, scan_nums):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_line_num = len(lines)
            if need_num >= 0:
                total_line_num = min(total_line_num, need_num)

            line_iter = trange(total_line_num)
            for idx in line_iter:
                line = lines[idx]
                line = line.strip()
                items = line.split()
                word = items[0].strip()
                vocabs.add(word)
    vocabs = list(vocabs)
    vocabs = sorted(vocabs)
    # 2.建立词典树
    lexicon_tree = Trie()
    for word in vocabs:
        lexicon_tree.insert(word)

    return lexicon_tree

# 这个函数从数据文件中提取所有的标签，并写入到标签文件中。
#
# 参数:
#
# files: 数据文件列表。
# label_file: 标签文件路径。
# defalut_label: 默认标签。
def get_all_labels_from_corpus(files, label_file, defalut_label='O'):
    """
    Args:
        files: data files
        label_file:
    """
    labels = [defalut_label]
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    label = sample['label']
                    if isinstance(label, list):
                        for l in label:
                            if l not in labels:
                                labels.append(l)
                    else:
                        labels.append(label)

    with open(label_file, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write("%s\n"%(label))
