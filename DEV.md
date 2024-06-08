不升级  cuda 10.0
见羽雀笔记

nvidia apex 的安装

https://blog.csdn.net/HUSTHY/article/details/109485088
$ git clone https://github.com/NVIDIA/apex
$ cd apex
$ pip install -v --no-cache-dir ./

sh 脚本文件
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --master_port 13517 --nproc_per_node=1 \
       Trainer.py  --do_train --do_eval --do_predict --evaluate_during_training \
                  --data_dir="data/dataset/NER/weibo" \
                  --output_dir="data/result/NER/weibo/lebertcrf" \
                  --config_name="data/berts/bert/config.json" \
                  --model_name_or_path="data/berts/bert/pytorch_model.bin" \
                  --vocab_file="data/berts/bert/vocab.txt" \
                  --word_vocab_file="data/vocab/tencent_vocab.txt" \
                  --max_scan_num=1000000 \
                  --max_word_num=5 \
                  --label_file="data/dataset/NER/weibo/labels.txt" \
                  --word_embedding="data/embedding/word_embedding.txt" \
                  --saved_embedding_dir="data/dataset/NER/weibo" \
                  --model_type="WCBertCRF_Token" \
                  --seed=106524 \
                  --per_gpu_train_batch_size=4 \
                  --per_gpu_eval_batch_size=16 \
                  --learning_rate=1e-5 \
                  --max_steps=-1 \
                  --max_seq_length=256 \
                  --num_train_epochs=20 \
                  --warmup_steps=190 \
                  --save_steps=600 \
                  --logging_steps=100
执行  sh run_demo2.sh

weibo 3-4小时 单卡
Test Result: acc: 0.9687, p: 0.6553, r: 0.7368, f1: 0.6937
resume3-4小时 单卡
Test Result: acc: 0.9692, p: 0.9469, r: 0.9626, f1: 0.9547
note43-4小时 单卡
Test Result: acc: 0.9692, p: 0.7925, r: 0.8125, f1: 0.8023
wsra 2天？中断 单卡
nky
Test Result: acc: 0.9002, p: 0.6680, r: 0.7622, f1: 0.7120

猪  单卡 10多小时
Test Result: acc: 0.9356, p: 0.8471, r: 0.8807, f1: 0.8636

第二次 单卡 20小时

label is bodyparts
Result: acc: 0.8754, p: 0.8844, r: 0.8764, f1: 0.8804

label is symptom
Result: acc: 0.8027, p: 0.8123, r: 0.8003, f1: 0.8063

label is type
Result: acc: 0.9118, p: 0.9552, r: 0.9343, f1: 0.9446

label is disease
Result: acc: 0.9338, p: 0.9433, r: 0.9510, f1: 0.9472


label is control
Result: acc: 0.5786, p: 0.7300, r: 0.5748, f1: 0.6432


label is medicine
Result: acc: 0.9010, p: 0.8367, r: 0.8723, f1: 0.8542


all

Test Result: acc: 0.9137, p: 0.7596, r: 0.8146, f1: 0.7861

鸡  单卡 一晚上

label is bodyparts
Result: acc: 0.8754, p: 0.8844, r: 0.8764, f1: 0.8804
label is symptom
Result: acc: 0.8027, p: 0.8123, r: 0.8003, f1: 0.8063
label is type
Result: acc: 0.9118, p: 0.9552, r: 0.9343, f1: 0.9446
label is disease
Result: acc: 0.9338, p: 0.9433, r: 0.9510, f1: 0.9472
label is I-disease
No Result
label is control
Result: acc: 0.5786, p: 0.7300, r: 0.5748, f1: 0.6432
label is I-type
No Result
label is I-bodyparts
No Result
label is I-symptom
No Result
label is medicine
Result: acc: 0.9010, p: 0.8367, r: 0.8723, f1: 0.8542
label is I-medicine
No Result
label is I-control
No Result
all:
Calling BertTokenizer.from_pretrained() with the path to a single file or url is deprecated
Test Result: acc: 0.9137, p: 0.7596, r: 0.8146, f1: 0.7861

猪和鸡 一起 4张卡 并行 3个小时 没有修改损失函数
disease--->	Result: acc: 0.9248, p: 0.8963, r: 0.9423, f1: 0.9188
type--->	Result: acc: 0.9247, p: 0.9620, r: 0.9325, f1: 0.9470
symptom--->	Result: acc: 0.7571, p: 0.7611, r: 0.7775, f1: 0.7692
control--->	Result: acc: 0.5878, p: 0.6515, r: 0.5513, f1: 0.5972
bodyparts--->	Result: acc: 0.7630, p: 0.7926, r: 0.7829, f1: 0.7877
medicine--->	Result: acc: 0.6522, p: 0.4909, r: 0.5094, f1: 0.5000
all--->	Result: acc: 0.8838, p: 0.6852, r: 0.7625, f1: 0.7218


猪和鸡一起 4张卡，并行，修改损失函数-0408
disease--->	Result: acc: 0.9305, p: 0.8848, r: 0.9359, f1: 0.9097
type--->	Result: acc: 0.9221, p: 0.9557, r: 0.9264, f1: 0.9408
symptom--->	Result: acc: 0.7500, p: 0.7371, r: 0.7689, f1: 0.7526
control--->	Result: acc: 0.5845, p: 0.6667, r: 0.5641, f1: 0.6111
bodyparts--->	Result: acc: 0.7244, p: 0.7757, r: 0.7615, f1: 0.7685
medicine--->	Result: acc: 0.6680, p: 0.5660, r: 0.5660, f1: 0.5660
all--->	Result: acc: 0.8791, p: 0.6724, r: 0.7520, f1: 0.7100



pycharm debug
    # 下列代码是为了在pycharm里 debug
    # 针对分布式训练参数的环境变量设置，如果是从sh启动不需要这些环境变量设置
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "13517"

--local_rank=0
--do_train
--do_eval
--do_predict
--evaluate_during_training
--data_dir="data/dataset/NER/weibo"
--output_dir="data/result/NER/weibo/lebertcrf"
--config_name="data/berts/bert/config.json"
--model_name_or_path="data/berts/bert/pytorch_model.bin"
--vocab_file="data/berts/bert/vocab.txt"
--word_vocab_file="data/vocab/tencent_vocab.txt"
--max_scan_num=1000000
--max_word_num=5
--label_file="data/dataset/NER/weibo/labels.txt"
--word_embedding="data/embedding/word_embedding.txt"
--saved_embedding_dir="data/dataset/NER/weibo"
--model_type="WCBertCRF_Token"
--seed=106524
--per_gpu_train_batch_size=4
--per_gpu_eval_batch_size=16
--learning_rate=1e-5
--max_steps=-1
--max_seq_length=256
--num_train_epochs=20
--warmup_steps=190
--save_steps=600
--logging_steps=100

WCBertCRF_Token
BertWordLSTMCRF_Token



升级cuda 11.4，最终失败

安装 torch

torch=1.8.0  支持不了现在用的高算力的A6000，所以torch要升级
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

torch如果安装不上，从官网
conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia




安装 tensorflow
tensorflow=2.3.1  版本与现在的GPU，cuda 11.4不兼容
会报错：Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory

pip install tensorflow-gpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple


ImportError: cannot import name 'container_abcs' from 'torch._six' (/root/anaconda3/envs/pytorch/lib/python3.8/site-packages/torch/_six.py)
因为1.8版本之后container_abcs就已经被移除了。

需要找 与当前的torch1.12.1对应的apex版本
之前通过git源代码方式安装的apex版本太低，引用的torch太低
现在torch版本升级了，所以apex也要升级

升级 apex后，又引起了新的问题
File "/root/anaconda3/envs/pytorch/lib/python3.8/site-packages/apex/__init__.py", line 13, in <module>
    from pyramid.session import UnencryptedCookieSessionFactoryConfig
ImportError: cannot import name 'UnencryptedCookieSessionFactoryConfig' from 'pyramid.session' (unknown location)


损失函数的修改
---------------------------
修改
D:\work\LEBERT-main\wcbert_modeling.py
507行
    # return (loss, preds)
    # 修改部分
    return (loss, preds, sequence_output)

D:\work\LEBERT-main\Trainer.py


 289行
 # 修改部分
    # outputs_1 = model(**inputs)
    # outputs_2 = model(**inputs)
    # seq_output_1 = outputs_1[2]
    # seq_output_2 = outputs_2[2]
    # print("seq_output_1:{}".format(seq_output_1.shape))
    # print("seq_output_2:{}".format(seq_output_2.shape))
    # c_loss = cts_loss(seq_output_1[:, 0, :],
    #                   seq_output_2[:, 0, :], temp=1.0,
    #                   batch_size=seq_output_1[:, 0, :].shape[0])
    # loss = loss + 0.1 * c_loss

511行
增加
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
    print("labels.shape is {}".format(labels.shape))
    print("logits.shape is {}".format(logits.shape))
    ce_loss = nn.CrossEntropyLoss()
    loss_ce = ce_loss(logits, labels)
#    loss = ce_loss(logits.clone(), labels.clone())
    return loss_ce

437行
        # broadcast_buffers=False,
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )
 215行
         model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True
        )


  检查之前运行的程序python进程有没有释放，杀死没有释放的python进程

  ps aux | grep python 得到进程PID
  kill {PID}


  bert的embedding的可视化

  embedding  4,256,786维度的张量
  由层输出的嵌入，一个形状为(number_of_data_points, max_sequence_length, embedddings_dimension)的张量
    number_of_data_points 是批处理中数据点的数量。
    max_sequence_length 是序列的最大长度。
    embeddings_dimension 是嵌入的维度，例如BERT模型中通常是768。