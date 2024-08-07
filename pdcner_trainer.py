# -*- coding: utf-8 -*-

"""
The training of   Pig Disease Chinese Named Entity Recognition (PDCNER) model
"""

import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from tqdm.auto import tqdm, trange
from transformers.configuration_bert import BertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_bert import BertTokenizer

from feature.task_dataset import TaskDataset
from feature.vocab import ItemVocabFile, ItemVocabArray
from function.metrics_nky import seq_f1_with_mask
from function.preprocess import build_lexicon_tree_from_vocabs, get_corpus_matched_word_from_lexicon_tree
from function.utils import build_pretrained_embedding_for_corpus, save_preds_for_seq_labelling
from module.sampler import SequentialDistributedSampler
from wcbert_modeling_nky import WCBertCRFForTokenClassification, BertWordLSTMCRFForTokenClassification
from wcbert_parser import get_argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print("No Tensorboard Found!!!")

### to enable fp16 training, note pytorch >= 1.16.0 #########
# from torch.cuda.amp import autocast
from torch.cuda.amp import autocast as autocast, GradScaler
from apex import amp

# 在训练最开始之前实例化一个GradScaler对象
scaler = GradScaler()

_use_apex = True
_use_native_amp = False

###### for multi-gpu DistributedDataParallel training  #########
os.environ['NCCL_DEBUG'] = 'INFO'  # print more detailed NCCL log information
os.environ['NCCL_IB_DISABLE'] = '1'  # force IP sockets usage

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
logfile = './data/log/wcbert_token_file_{}.txt'.format(time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(time.time())))
fh = logging.FileHandler(logfile)
fh.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fh)

PREFIX_CHECKPOINT_DIR = "checkpoint"
# 为数据集加载标签，这将用于在可视化中对每个数据样本进行颜色编码。
# 当模型的前向传播给定output_hidden_states=True参数时，BertLayer嵌入层和12个BertLayer层中的每一层都可以返回它们的输出（也称为hidden_states ）。因此，模型的维数(13, number_of_data_points, max_sequence_length, embeddings_dimension)。
# 因为我们只对12个 BertLayer层的嵌入感兴趣，所以我们将不需要的BertLayer层嵌入切掉，只留下 (12, number_of_data_points, max_sequence_length, embeddings_dimension)
# dim_reducer: scikit-learn的t-SNE降维实现，将嵌入从BERT默认的768维降为2维。你还可以使用PCA，这取决于哪种更适合你的数据集；
dim_reducer = TSNE(n_components=2)
def visualize_layerwise_embeddings(hidden_states, masks, labels, epoch, title, layers_to_visualize):
    num_layers = len(layers_to_visualize)

    fig = plt.figure(figsize=(24, (int(num_layers / 4)) * 6))  # 每个子图的大小为6x6，每一行将包含4个图
    ax = [fig.add_subplot(int(num_layers / 4), 4, i + 1) for i in range(num_layers)]

    labels = labels.detach().cpu().numpy().reshape(-1)
    for i, layer_i in enumerate(layers_to_visualize):
        # 由层输出的嵌入，一个形状为(number_of_data_points, max_sequence_length, embedddings_dimension)的张量
        layer_embeds = hidden_states[layer_i]
        # layer_embeds.sum(dim=1) 沿着序列长度维度（dim=1）对嵌入向量求和，将嵌入向量的维度从 (number_of_data_points, max_sequence_length, embeddings_dimension) 减少到 (number_of_data_points, embeddings_dimension)。
        # masks.sum(dim=1, keepdim=True) 沿着序列长度维度对掩码求和，并使用 keepdim=True 保持维度，这样求和结果的维度是 (number8_of_data_points, 1)。
        # torch.div(..., masks.sum(dim=1, keepdim=True)) 将求和后的嵌入向量除以对应的掩码求和，得到平均嵌入向量。这一步确保了只有非掩码（即有效的标记）对嵌入向量的平均值有贡献。
        # 通过对序列的所有非掩码标记的嵌入取平均值，为每个数据点创建一个单一的嵌入，从而得到一个形状为(number_of_data_points, embedddings_dimension)的张量
        # 因此，layer_averaged_hidden_states 的最终维度是 (number_of_data_points, embeddings_dimension)。这意味着每个数据点现在由一个单一的嵌入向量表示，该向量是其序列中所有有效标记嵌入向量的平均值。
        layer_averaged_hidden_states = torch.div(layer_embeds.sum(dim=1), masks.sum(dim=1, keepdim=True))
        # t-SNE维减少嵌入，形状为(number_of_data_points, embeddings_dimension)的张量
        # layer_dim_reduced_embeds 是通过将 layer_averaged_hidden_states 通过 t-SNE 降维得到的。t-SNE 算法将高维数据映射到二维空间，因此其输出的维度是固定的，为 (number_of_data_points, 2)。
        # 这里的 number_of_data_points 是批处理中数据点的数量，而 2 表示 t-SNE 降维后的二维坐标。所以无论原始嵌入向量的维度是多少，经过 t-SNE 处理后，每个数据点都将被表示为一个包含两个坐标值的向量，这两个坐标值代表了在二维空间中的位置。
        layer_dim_reduced_embeds = dim_reducer.fit_transform(layer_averaged_hidden_states.detach().cpu().numpy())


        labels=generate_label_array(len(layer_dim_reduced_embeds[:, 0]))
        df = pd.DataFrame.from_dict(
            {'x': layer_dim_reduced_embeds[:, 0], 'y': layer_dim_reduced_embeds[:, 1], 'label': labels})

        sns.scatterplot(data=df, x='x', y='y', hue='label', ax=ax[i])

    plt.savefig(f'/data/LEBERT-main/tmp/nky-pig/20240610/epoch_{epoch}.png', format='png', pad_inches=0)


def generate_label_array(n):
    return np.array(['label' + str(i + 1) for i in range(n)])

def visualize_embeddings_pca(embeddings, labels, title="Embedding Visualization", save_path=None):
    pca = PCA(n_components=2)  # 降维到2维
    embeddings_2d = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], label=label)

    plt.title(title)
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_embeddings_tsne(embeddings, labels, title="Embedding Visualization", save_path=None):
    # 将embedding通过PCA或t-SNE转换为二维或三维数据
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)

    # 创建散点图
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, c=labels, cmap='viridis')
    # 添加图例
    plt.legend(handles=scatter.legend_elements()[0], labels=sorted(set(labels)))
    # 添加标题和轴标签
    plt.title('BERT Embedding Visualization')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    # 显示图表
    plt.show()

# 用于设置随机种子以确保结果的可重复性。
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 用于生成数据加载器，根据训练或评估模式选择适当的采样器。
def get_dataloader(dataset, args, mode='train'):
    """
    generator datasetloader for training.
    Note that: for training, we need random sampler, same to shuffle
               for eval or predict, we need sequence sampler, same to no shuffle
    Args:
        dataset:
        args:
        mode: train or non-train
    """
    print("Dataset length: ", len(dataset))
    if args.local_rank != -1:
        if mode == 'train':
            sampler = SequentialDistributedSampler(dataset, do_shuffle=True)
        else:
            sampler = SequentialDistributedSampler(dataset)
    else:
        if mode == 'train':
            sampler = SequentialSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
    if mode == 'train':
        batch_size = args.per_gpu_train_batch_size
    else:
        batch_size = args.per_gpu_eval_batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader

# 设置优化器和学习率调度器。
def get_optimizer(model, args, num_training_steps):
    """
    Setup the optimizer and the learning rate scheduler

    we provide a reasonable default that works well
    If you want to use something else, you can pass a tuple in the Trainer's init,
    or override this method in a subclass.
    """
    no_bigger = ["word_embedding", "attn_w", "word_transform", "word_word_weight", "hidden2tag",
                 "lstm", "crf"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_bigger)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_bigger)],
            "lr": 0.0001
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
    )

    return optimizer, scheduler

# 打印和记录训练和评估的日志信息。
def print_log(logs, epoch, global_step, eval_type, tb_writer, iterator=None):
    if epoch is not None:
        logs['epoch'] = epoch
    if global_step is None:
        global_step = 0
    if eval_type in ["Dev", "Test"]:
        print("#############  %s's result  #############" % (eval_type))
    if tb_writer:
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                tb_writer.add_scalar(k, v, global_step)
            else:
                logger.warning(
                    "Trainer is attempting to log a value of "
                    '"%s" of type %s for key "%s" as a scalar. '
                    "This invocation of Tensorboard's writer.add_scalar() "
                    "is incorrect so we dropped this attribute.",
                    v,
                    type(v),
                    k,
                )
        tb_writer.flush()

    output = {**logs, **{"step": global_step}}
    if iterator is not None:
        iterator.write(output)
    else:
        logger.info(output)

# 这是训练模型的主要函数，它执行以下操作
# 准备数据加载器。
# 设置优化器和学习率调度器。
# 检查是否使用混合精度训练。
# 初始化分布式训练（如果适用）。
# 执行训练循环，包括前向传播、损失计算、反向传播和参数更新。
# 在每个epoch结束时保存模型的检查点。
def train(model, args, train_dataset, dev_dataset, test_dataset, label_vocab, tb_writer, model_path=None):


    """
    train the model
    """
    ## 1.prepare data
    train_dataloader = get_dataloader(train_dataset, args, mode='train')
    if args.max_steps > 0:
        t_total = args.max_steps
        num_train_epochs = (
                args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = int(len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs)
        num_train_epochs = args.num_train_epochs

    ## 2.optimizer and model
    optimizer, scheduler = get_optimizer(model, args, t_total)
    # 混合精度训练
    if args.fp16 and _use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Check if saved optimizer or scheduler states exist
    if (model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
    ):
        optimizer.load_state_dict(
            torch.load(os.path.join(model_path, "optimizer.pt"), map_location=args.device)
        )
        scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

    if args.local_rank != -1:
        model = model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    ## 3.begin train
    total_train_batch_size = args.per_gpu_train_batch_size * args.gradient_accumulation_steps
    if args.local_rank == 0 or args.local_rank == -1:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataloader.dataset))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epoch = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    if model_path is not None:  # load checkpoint and continue training
        try:
            global_step = int(model_path.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
            )
            model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin")))
            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            global_step = 0
            logger.info("  Starting fine-tuning.")

    tr_loss = 0.0
    logging_loss = 0.0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")

        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                # Skip past any already trained steps if resuming training
                steps_trained_in_current_epoch -= 1
                continue
            model.train()

            # new batch data: [input_ids, token_type_ids, attention_mask, matched_word_ids,
            # matched_word_mask, boundary_ids, labels
            batch_data = (batch[0], batch[2], batch[1], batch[3], batch[4], batch[5], batch[6])
            new_batch = batch_data
            batch = tuple(t.to(args.device) for t in new_batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                      "matched_word_ids": batch[3], "matched_word_mask": batch[4],
                      "boundary_ids": batch[5], "labels": batch[6], "flag": "Train"}
            batch_data = None
            new_batch = None

            if args.fp16 and _use_native_amp:
                with autocast():
                    outputs = model(**inputs)
                    loss = outputs[0]
            else:
                outputs = model(**inputs)
                loss = outputs[0]
                # 对输出的emeding进行可视化
                # 确保model的forward方法返回embedding]
                # bert的嵌入层
                embeddings = outputs[2]
                # 多的12层+bert的嵌入层
                hidden_states=outputs[3]



                if (step + 1) == len(epoch_iterator):
                    # 假设labels是当前batch的标签
                    # labels = batch[6]  # 根据你的数据格式调整
                    # visualize_embeddings_tsne(embeddings.detach().cpu().numpy(), labels)

                    # 只输出指定的8个layer信息，可修改
                    visualize_layerwise_embeddings(hidden_states=hidden_states,
                                                masks=batch[1],
                                                labels=batch[6],
                                                epoch=epoch,
                                                title='train_data',
                                                layers_to_visualize=[0, 1, 2, 3, 8, 9, 10, 11]
                                                )



                # #修改损失函数
                outputs_1 = model(**inputs)
                outputs_2 = model(**inputs)
                seq_output_1 = outputs_1[2]
                seq_output_2 = outputs_2[2]
                # print("seq_output_1:{}".format(seq_output_1.shape))
                # print("seq_output_2:{}".format(seq_output_2.shape))
                c_loss = cts_loss(seq_output_1[:, 0, :],
                                  seq_output_2[:, 0, :], temp=1.0,
                                  batch_size=seq_output_1[:, 0, :].shape[0])
                loss = loss + 0.1 * c_loss



            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16 and _use_native_amp:
                scaler.scale(loss).backward()
            elif args.fp16 and _use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                # # 正向传播时：开启自动求导的异常侦测
                # torch.autograd.set_detect_anomaly(True)
                #
                # # 反向传播时：在求导时开启侦测
                # with torch.autograd.detect_anomaly():
                #     loss.backward()
                loss.backward()

            tr_loss += loss.item()

            ## update gradient
            if (step + 1) % args.gradient_accumulation_steps == 0 or \
                    ((step + 1) == len(epoch_iterator)):
                if args.fp16 and _use_native_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                elif args.fp16 and _use_apex:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.fp16 and _use_native_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                model.zero_grad()
                global_step += 1

                ## logger and evaluate
                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    logs = {}
                    logs["loss"] = (tr_loss - logging_loss) / args.logging_steps
                    # backward compatibility for pytorch schedulers
                    logs["learning_rate"] = (
                        scheduler.get_last_lr()[0]
                        if version.parse(torch.__version__) >= version.parse("1.4")
                        else scheduler.get_lr()[0]
                    )
                    logging_loss = tr_loss
                    if args.local_rank == 0 or args.local_rank == -1:
                        print_log(logs, epoch, global_step, "", tb_writer)

                ## save checkpoint
                if False and args.save_steps > 0 and global_step % args.save_steps == 0 and \
                        (args.local_rank == 0 or args.local_rank == -1):
                    output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
                    os.makedirs(output_dir, exist_ok=True)

                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                    if False and args.evaluate_during_training:
                        # for dev
                        metrics, _ = evaluate(
                            model, args, dev_dataset, label_vocab, global_step, description="Dev")
                        if args.local_rank == 0 or args.local_rank == -1:
                            print_log(metrics, epoch, global_step, "Dev", tb_writer)

                        # for test
                        metrics, _ = evaluate(
                            model, args, test_dataset, label_vocab, global_step, description="Test")
                        if args.local_rank == 0 or args.local_rank == -1:
                            print_log(metrics, epoch, global_step, "Test", tb_writer)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        # save after each epoch
        output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
        os.makedirs(output_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        # evaluate after each epoch
        if args.evaluate_during_training:
            # for dev
            metrics, _ = evaluate(args.label_file, model, args, dev_dataset, label_vocab, global_step,
                                  description="Dev",
                                  write_file=True)
            if args.local_rank == 0 or args.local_rank == -1:
                print_log(metrics, epoch, global_step, "Dev", tb_writer)

            # for test
            metrics, _ = evaluate(args.label_file, model, args, test_dataset, label_vocab, global_step,
                                  description="Test",
                                  write_file=True)
            if args.local_rank == 0 or args.local_rank == -1:
                print_log(metrics, epoch, global_step, "Test", tb_writer)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    # save the last one
    output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{global_step}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # model.save_pretrained(os.path.join(output_dir, "pytorch-model.bin"))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    print("global_step: ", global_step)
    logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
    return global_step, tr_loss / global_step

# 用于评估模型在给定数据集上的性能。它计算准确率、精确度、召回率和F1分数，并将预测结果保存到文件
def evaluate(label_file, model, args, dataset, label_vocab, global_step, description="dev", write_file=False):
    """
    evaluate the model's performance
    """
    dataloader = get_dataloader(dataset, args, mode='dev')
    if (not args.do_train) and (not args.no_cuda) and args.local_rank != -1:
        model = model.cuda()
        # 由此可以猜测：在分布式训练中，如果对同一模型进行多次调用则会触发以上报错，即
        # nn.parallel.DistributedDataParallel方法封装的模型，forword()函数和backward()函数必须交替执行，如果执行多个（次）forward()然后执行一次backward()则会报错。
        # 在该出错文件上找到被调用的DistributedDataParallel()，将broadcast_buffers设置为False
        # broadcast_buffers=False,
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    batch_size = dataloader.batch_size
    if args.local_rank == 0 or args.local_rank == -1:
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Batch size = %d", batch_size)
    eval_losses = []
    model.eval()

    all_input_ids = None
    all_label_ids = None
    all_predict_ids = None
    all_attention_mask = None

    for batch in tqdm(dataloader, desc=description):
        # new batch data: [input_ids, token_type_ids, attention_mask, matched_word_ids,
        # matched_word_mask, boundary_ids, labels
        batch_data = (batch[0], batch[2], batch[1], batch[3], batch[4], batch[5], batch[6])
        new_batch = batch_data
        batch = tuple(t.to(args.device) for t in new_batch)
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2],
                  "matched_word_ids": batch[3], "matched_word_mask": batch[4],
                  "boundary_ids": batch[5], "labels": batch[6], "flag": "Predict"}
        batch_data = None
        new_batch = None

        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        input_ids = batch[0].detach().cpu().numpy()
        label_ids = batch[6].detach().cpu().numpy()

        pred_ids = preds.detach().cpu().numpy()
        attention_mask = batch[1].detach().cpu().numpy()
        if all_label_ids is None:
            all_input_ids = input_ids
            all_label_ids = label_ids
            all_predict_ids = pred_ids
            all_attention_mask = attention_mask
        else:
            all_input_ids = np.append(all_input_ids, input_ids, axis=0)
            all_label_ids = np.append(all_label_ids, label_ids, axis=0)
            all_predict_ids = np.append(all_predict_ids, pred_ids, axis=0)
            all_attention_mask = np.append(all_attention_mask, attention_mask, axis=0)

    ## calculate metrics
    output_dir = args.output_dir
    acc, p, r, f1, all_true_labels, all_pred_labels = seq_f1_with_mask(description, global_step, output_dir, label_file,
                                                                       all_label_ids, all_predict_ids,
                                                                       all_attention_mask, label_vocab)
    metrics = {}
    metrics['acc'] = acc
    metrics['p'] = p
    metrics['r'] = r
    metrics['f1'] = f1

    ## write labels into file
    if write_file:
        file_path = os.path.join(args.output_dir, "{}-{}-{}.txt".format(args.model_type, description, str(global_step)))
        tokenizer = BertTokenizer.from_pretrained(args.vocab_file)
        save_preds_for_seq_labelling(all_input_ids, tokenizer, all_true_labels, all_pred_labels, file_path)

    return metrics, (all_true_labels, all_pred_labels)

# 定义了一个对比损失函数，用于对比学习。
# 用了对比学习，正样本拉近距离，负样本拉远距离，类似这样

# cts_loss函数是一个基于对比学习的损失函数，用于拉近相似样本之间的距离，同时推远不相似样本之间的距离。以下是该函数的详细解释：
# 参数说明
# z_i: 一批样本的嵌入表示，形状为 [B, D]，其中 B 是批量大小，D 是嵌入维度。
# z_j: 与 z_i 相对应的另一批样本的嵌入表示，形状也为 [B, D]。
# temp: 温度参数，用于控制softmax函数的平滑程度。
# batch_size: 当前处理的批量大小。
# 函数逻辑
# 合并嵌入:
# z = torch.cat((z_i, z_j), dim=0) 将两个批次的嵌入合并为一个 [2B, D] 的矩阵。
# 计算相似性矩阵:
# sim = torch.mm(z, z.T) / temp 计算合并后嵌入的点积，并通过温度参数进行缩放，得到一个 [2B, 2B] 的相似性矩阵。
# 提取正样本相似性:
# sim_i_j 和 sim_j_i 分别是通过在相似性矩阵上取对角线和负对角线元素得到的两个 [B, 1] 的向量，代表正样本之间的相似性。
# 组合正样本:
# positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) 将两个正样本相似性向量合并为一个 [2B, 1] 的矩阵。
# 创建掩码:
# mask = mask_correlated_samples(batch_size) 创建一个掩码，用于在相似性矩阵中选择负样本。
# 提取负样本相似性:
# negative_samples = sim[mask].reshape(N, -1) 根据掩码从相似性矩阵中提取负样本相似性。
# 创建标签:
# labels = torch.zeros(N).to(positive_samples.device).long() 创建一个全0的标签向量，表示所有样本都是类别0。
# 拼接正负样本:
# logits = torch.cat((positive_samples, negative_samples), dim=1) 将正样本和负样本相似性拼接起来，形成一个 [N, C] 的 logits 矩阵，其中 C 是类别数（正样本数加负样本数）。
#
# 计算交叉熵损失:
# loss_ce = ce_loss(logits, labels) 使用交叉熵损失函数计算最终的损失。
#
# 返回损失:
# 函数返回计算得到的损失值。
#
# 与原始损失函数的不同之处
# 基于相似性: cts_loss基于样本之间的相似性进行计算，而不是直接基于标签。
# 负采样: 通过负采样引入额外的对比项，鼓励模型将正样本与负样本区分开来。
# 温度调节: 使用温度参数来控制softmax函数的平滑程度，影响学习过程。
# 这种损失函数通常用于无监督或自监督学习场景，可以帮助模型学习区分不同样本的特征表示，从而提高模型对输入数据的理解。在某些任务中，如异常检测或相似性度量，这种损失函数可能比传统的监督学习损失函数更有效。
def cts_loss(z_i, z_j, temp, batch_size):  # B * D    B * D

    N = 2 * batch_size

    z = torch.cat((z_i, z_j), dim=0)  # 2B * D

    sim = torch.mm(z, z.T) / temp  # 2B * 2B

    sim_i_j = torch.diag(sim, batch_size)  # B*1
    sim_j_i = torch.diag(sim, -batch_size)  # B*1

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

    mask = mask_correlated_samples(batch_size)

    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)  # N * C
    # print("labels.shape is {}".format(labels.shape))
    # print("logits.shape is {}".format(logits.shape))
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(logits, labels)
#    loss = ce_loss(logits.clone(), labels.clone())
    return loss_ce


def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask

# 程序的入口点，执行以下操作：
#
# 解析命令行参数。
# 初始化日志记录和TensorBoard记录器。
# 设置随机种子。
# 准备数据和模型。
# 如果设置了do_train，则调用train函数开始训练。
# 如果设置了do_eval，则在开发集上评估模型。
# 如果设置了do_predict，则在测试集上评估模型。

#  --data_dir="data/dataset/NER/nky-pig" \
#                   --output_dir="data/result/NER/nky-pig/pdcncercrf_20240511-2" \
#                   --config_name="data/berts/bert/config.json" \
#                   --model_name_or_path="data/berts/bert/pytorch_model.bin" \
#                   --vocab_file="data/berts/bert/vocab.txt" \
#                   --word_vocab_file="data/vocab/tencent_vocab.txt" \
#                   --max_scan_num=1000000 \
#                   --max_word_num=5 \
#                   --label_file="data/dataset/NER/nky-pig/labels.txt" \
#                   --word_embedding="data/embedding/word_embedding.txt" \
#                   --saved_embedding_dir="data/dataset/NER/nky-pig" \
#                   --model_type="WCBertCRF_Token" \
#                   --seed=106524 \
#                   --per_gpu_train_batch_size=4 \
#                   --per_gpu_eval_batch_size=16 \
#                   --learning_rate=1e-5 \
#                   --max_steps=-1 \
#                   --max_seq_length=256 \
#                   --num_train_epochs=20 \
#                   --warmup_steps=190 \
#                   --save_steps=600 \
#                   --logging_steps=100
def main():
    args = get_argparse().parse_args()
    args.no_cuda = not torch.cuda.is_available()

    # 下列代码是为了在pycharm里 debug
    # 针对分布式训练参数的环境变量设置，如果是从sh启动不需要这些环境变量设置
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "13517"

    # 脚本支持多GPU训练，使用了PyTorch的分布式通信包。
    ########### for multi-gpu training ##############
    if torch.cuda.is_available() and args.local_rank != -1:
        args.n_gpu = 1
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    #################################################

    args.device = device
    logger.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = SummaryWriter(log_dir=args.logging_dir)
    set_seed(args.seed)

    ## 1.prepare data
    # a. lexicon tree
    lexicon_tree = build_lexicon_tree_from_vocabs([args.word_vocab_file], scan_nums=[args.max_scan_num])
    embed_lexicon_tree = lexicon_tree

    # b. word vocab, label vocab
    train_data_file = os.path.join(args.data_dir, "train.json")
    # if only has test_set no dev_set, such as msra NER
    if "msra" in args.data_dir:
        dev_data_file = os.path.join(args.data_dir, "test.json")
    else:
        dev_data_file = os.path.join(args.data_dir, "dev.json")
    test_data_file = os.path.join(args.data_dir, "test.json")
    data_files = [train_data_file, dev_data_file, test_data_file]
    matched_words = get_corpus_matched_word_from_lexicon_tree(data_files, embed_lexicon_tree)
    word_vocab = ItemVocabArray(items_array=matched_words, is_word=True, has_default=False, unk_num=5)
    label_vocab = ItemVocabFile(files=[args.label_file], is_word=False)
    tokenizer = BertTokenizer.from_pretrained(args.vocab_file)

    with open("word_vocab.txt", "w", encoding="utf-8") as f:
        for idx, word in enumerate(word_vocab.idx2item):
            f.write("%d\t%s\n" % (idx, word))

    # c. prepare embeddinggit
    pretrained_word_embedding, embed_dim = build_pretrained_embedding_for_corpus(
        embedding_path=args.word_embedding,
        word_vocab=word_vocab,
        embed_dim=args.word_embed_dim,
        max_scan_num=args.max_scan_num,
        saved_corpus_embedding_dir=args.saved_embedding_dir,
    )

    # 根据参数args.model_type，实例化了两种不同的模型：WCBertCRFForTokenClassification或BertWordLSTMCRFForTokenClassification。
    # d. define model
    config = BertConfig.from_pretrained(args.config_name)
    if args.model_type == "WCBertCRF_Token":
        model = WCBertCRFForTokenClassification.from_pretrained(
            args.model_name_or_path, config=config,
            pretrained_embeddings=pretrained_word_embedding,
            num_labels=label_vocab.get_item_size())
    elif args.model_type == "BertWordLSTMCRF_Token":
        model = BertWordLSTMCRFForTokenClassification.from_pretrained(
            args.model_name_or_path, config=config,
            pretrained_embeddings=pretrained_word_embedding,
            num_labels=label_vocab.get_item_size()
        )

    if not args.no_cuda:
        model = model.cuda()
    args.label_size = label_vocab.get_item_size()
    dataset_params = {
        'tokenizer': tokenizer,
        'word_vocab': word_vocab,
        'label_vocab': label_vocab,
        'lexicon_tree': lexicon_tree,
        'max_seq_length': args.max_seq_length,
        'max_scan_num': args.max_scan_num,
        'max_word_num': args.max_word_num,
        'default_label': args.default_label,
    }

    if args.do_train:
        train_dataset = TaskDataset(train_data_file, params=dataset_params, do_shuffle=args.do_shuffle)
        dev_dataset = TaskDataset(dev_data_file, params=dataset_params, do_shuffle=False)
        test_dataset = TaskDataset(test_data_file, params=dataset_params, do_shuffle=False)
        # args.model_name_or_path = None
        train(model, args, train_dataset, dev_dataset, test_dataset, label_vocab, tb_writer)

    if args.do_eval:
        logger.info("*** Dev Evaluate ***")
        dev_dataset = TaskDataset(dev_data_file, params=dataset_params, do_shuffle=False)
        global_steps = args.model_name_or_path.split("/")[-2].split("-")[-1]
        eval_output, _ = evaluate(args.label_file, model, args, dev_dataset, label_vocab, global_steps, "dev",
                                  write_file=True)
        eval_output["global_steps"] = global_steps
        print("Dev Result: acc: %.4f, p: %.4f, r: %.4f, f1: %.4f\n" %
              (eval_output['acc'], eval_output['p'], eval_output['r'], eval_output['f1']))

    # return eval_output
    if args.do_predict:
        logger.info("*** Test Evaluate ***")
        test_dataset = TaskDataset(test_data_file, params=dataset_params, do_shuffle=False)
        global_steps = args.model_name_or_path.split("/")[-2].split("-")[-1]
        eval_output, _ = evaluate(args.label_file, model, args, test_dataset, label_vocab, global_steps, "test",
                                  write_file=True)
        eval_output["global_steps"] = global_steps
        print("Test Result: acc: %.4f, p: %.4f, r: %.4f, f1: %.4f\n" %
              (eval_output['acc'], eval_output['p'], eval_output['r'], eval_output['f1']))


if __name__ == "__main__":
    main()
