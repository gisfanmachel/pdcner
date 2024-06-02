# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from tqdm import tqdm, trange
import torch
import pickle
# 提供了一些实用工具函数，例如加载预训练词嵌入、构建词嵌入、保存预测结果等。
def load_pretrain_embed(embedding_path, max_scan_num=1000000, add_seg_vocab=False):
    """
    从pretrained word embedding中读取前max_scan_num的词向量
    Args:
        embedding_path: 词向量路径
        max_scan_num: 最多读多少
    """
    ## 如果是使用add_seg_vocab, 则全局遍历
    if add_seg_vocab:
        max_scan_num = -1

    embed_dict = dict()
    embed_dim = -1
    with open(embedding_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if max_scan_num == -1:
            max_scan_num = len(lines)
        max_scan_num = min(max_scan_num, len(lines))
        line_iter = trange(max_scan_num)
        for idx in line_iter:
            line = lines[idx]
            line = line.strip()
            items = line.split()
            if len(items) == 2:
                embed_dim = int(items[1])
                continue
            elif len(items) == 201:
                token = items[0]
                embedd = np.empty([1, embed_dim])
                embedd[:] = items[1:]
                embed_dict[token] = embedd
            elif len(items) > 201:
                print("++++longer than 201+++++, line is: %s\n" % (line))
                token = items[0:-200]
                token = "".join(token)
                embedd = np.empty([1, embed_dim])
                embedd[:] = items[-200:]
                embed_dict[token] = embedd
            else:
                print("-------error word-------, line is: %s\n"%(line))

    return embed_dict, embed_dim

# 为语料库构建预训练的词嵌入（word embeddings）。这个过程涉及加载预训练的词向量，并将其与语料库中出现的词进行匹配

# 参数说明
# embedding_path: 预训练词嵌入文件的路径。
# word_vocab: 语料库的词汇表对象，包含语料库中所有独特词的列表。
# embed_dim: 词嵌入的维度，默认为200。
# max_scan_num: 加载预训练词嵌入时扫描的最大词数。
# saved_corpus_embedding_dir: 保存构建的词嵌入文件的目录。
# add_seg_vocab: 是否添加额外的词汇表（例如分词词汇表）。
# 函数逻辑
# 检查缓存:
# 函数首先检查是否已经为当前语料库构建了词嵌入，并且是否保存在saved_corpus_embedding_dir指定的目录中。如果是，则直接从文件加载，避免重复构建。
#
# 加载预训练词嵌入:
# 如果缓存中没有构建好的词嵌入，则从embedding_path指定的文件中加载预训练的词嵌入。函数load_pretrain_embed被调用来读取文件，并返回一个字典embed_dict，其中包含已加载词的词向量，以及embed_dim。
#
# 初始化词嵌入矩阵:
# 创建一个矩阵pretrained_emb，其大小为[word_vocab.item_size, embed_dim]，用于存储语料库中每个词的词嵌入。
#
# 匹配和赋值:
# 遍历词汇表中的每个词，检查它是否存在于embed_dict中：
#
# 如果存在，将预训练的词向量赋值给pretrained_emb矩阵中对应的行。
# 如果不存在，为该词随机初始化一个词向量，并记录未匹配的词的数量。
# 保存构建的词嵌入:
# 将构建的词嵌入矩阵pretrained_emb保存到文件中，以便于将来重用。
#
# 打印统计信息:
# 打印匹配的词数、未匹配的词数以及未匹配词的百分比。
#
# 返回结果:
# 函数返回构建的词嵌入矩阵pretrained_emb和词嵌入的维度embed_dim。
#
# 用途
# 这个函数的目的是为特定的语料库创建一个词嵌入矩阵，该矩阵可以用于BERT模型的训练，以提供给模型额外的词向量信息。这对于处理BERT词汇表中未包含的词特别有用，因为它们可以通过预训练的词嵌入得到有效的表示。
#
# 注意事项
# 该函数假设预训练词嵌入文件的格式是每行一个词向量，词向量中的元素由空格分隔。
# 如果embedding_path指定的文件非常大，max_scan_num可以用来限制加载的词向量数量，以减少内存使用。
# 函数提供了一个可选的参数add_seg_vocab，允许在构建词嵌入时考虑额外的词汇表。这在处理某些特定的NLP任务时可能很有用。
def build_pretrained_embedding_for_corpus(
        embedding_path,
        word_vocab,
        embed_dim=200,
        max_scan_num=1000000,
        saved_corpus_embedding_dir=None,
        add_seg_vocab=False
):
    """
    Args:
        embedding_path: 预训练的word embedding路径
        word_vocab: corpus的word vocab
        embed_dim: 维度
        max_scan_num: 最大浏览多大数量的词表
        saved_corpus_embedding_dir: 这个corpus对应的embedding保存路径
    """
    saved_corpus_embedding_file = os.path.join(saved_corpus_embedding_dir, 'saved_word_embedding_{}.pkl'.format(max_scan_num))

    if os.path.exists(saved_corpus_embedding_file):
        with open(saved_corpus_embedding_file, 'rb') as f:
            pretrained_emb = pickle.load(f)
        return pretrained_emb, embed_dim

    embed_dict = dict()
    if embedding_path is not None:
        embed_dict, embed_dim = load_pretrain_embed(embedding_path, max_scan_num=max_scan_num, add_seg_vocab=add_seg_vocab)

    scale = np.sqrt(3.0 / embed_dim)
    pretrained_emb = np.empty([word_vocab.item_size, embed_dim])

    matched = 0
    not_matched = 0

    for idx, word in enumerate(word_vocab.idx2item):
        if word in embed_dict:
            pretrained_emb[idx, :] = embed_dict[word]
            matched += 1
        else:
            pretrained_emb[idx, :] = np.random.uniform(-scale, scale, [1, embed_dim])
            not_matched += 1

    pretrained_size = len(embed_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, oov:%s, oov%%:%s" % (
    pretrained_size, matched, not_matched, (not_matched + 0.) / word_vocab.item_size))

    with open(saved_corpus_embedding_file, 'wb') as f:
        pickle.dump(pretrained_emb, f, protocol=4)

    return pretrained_emb, embed_dim

def reverse_padded_sequence(inputs, lengths, batch_first=True):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError("inputs is incompatible with lengths.")
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = ind.expand_as(inputs)
    if inputs.is_cuda:
        ind = ind.cuda()
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)

    return reversed_inputs


def random_embedding(self, vocab_size, embedding_dim):
    pretrain_emb = np.empty([vocab_size, embedding_dim])
    scale = np.sqrt(3.0 / embedding_dim)
    for index in range(vocab_size):
        pretrain_emb[index,:] = np.random.uniform(-scale, scale, [1, embedding_dim])
    return pretrain_emb


def gather_indexes(sequence_tensor, positions):
    """
    gather specific tensor based on the positions
    Args:
        sequence_tensor: [B, L, D]
        positions: [B, P]
    """
    batch_size = sequence_tensor.size(0)
    seq_length = sequence_tensor.size(1)
    dim = sequence_tensor.size(2)

    whole_seq_length = torch.tensor([seq_length for _ in range(batch_size)], dtype=torch.long)
    whole_seq_length = whole_seq_length.to(sequence_tensor.device)

    flat_offsets = torch.cumsum(whole_seq_length, dim=-1)
    flat_offsets = flat_offsets - whole_seq_length # [B]
    flat_offsets = flat_offsets.unsqueeze(-1) # [B, 1]
    flat_positions = positions + flat_offsets # [B, P]
    flat_positions = flat_positions.contiguous().view(-1)
    flat_sequence_tensor = sequence_tensor.contiguous().view(batch_size * seq_length, -1) # [B * L, D]

    # output_tensor = flat_sequence_tensor[flat_positions]
    output_tensor = flat_sequence_tensor.index_select(0, flat_positions)
    output_tensor = output_tensor.contiguous().view(batch_size, -1)

    return output_tensor

def save_preds_for_seq_labelling(token_ids, tokenizer, true_labels, pred_labels, file):
    """
    save sequence labelling result into files
    Args:
        token_ids:
        tokenizer:
        true_labels:
        pred_labels:
        file:
    """
    error_num = 1
    with open(file, 'w', encoding='utf-8') as f:
        for w_ids, t_labels, p_labels in zip(token_ids, true_labels, pred_labels):
            tokens = tokenizer.convert_ids_to_tokens(w_ids)
            token_num = len(t_labels)
            tokens = tokens[1:token_num+1]

            assert len(tokens) == len(t_labels), (len(tokens), len(t_labels))
            assert len(tokens) == len(p_labels), (len(tokens), len(p_labels))

            for w, t, p in zip(tokens, t_labels, p_labels):
                if t == p:
                    f.write("%s\t%s\t%s\n"%(w, t, p))
                else:
                    f.write("%s\t%s\t%s\t%d\n"%(w, t, p, error_num))
                    error_num += 1

            f.write("\n")
