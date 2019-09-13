XLNet for Chinese, TensorFlow & PyTorch

XLNet中文预训练模型
--------------------------------------------------------------
XLNet是CMU和谷歌大脑在2019年6月份，提出的一个新的预训练模型。在多个任务的性能超越Bert。它是在保留自回归语言模型(Autoregressive Language Modeling)的形式下，

结合了自编码语言模型(Autoencoding Language Modeling)的优势，提出了排列语言模型(Permutation Language Modeling)。并且它基于Transfomer-XL,

有更好的处理长文本的能力。

本项目参考<a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">[2]</a>的工作，结合海量数据，训练了一个24层的中文XLNet_zh_Large模型，含3亿多参数。

#### 训练数据和计算资源 Training Corpus & Training Details

训练数据，包括新闻、互动讨论、百科，超过30G原始文本，近100亿个中文字； 本项目与中文预训练RoBERTa<a href="https://github.com/brightmart/roberta_zh">RoBERTa_zh</a>项目，使用相同的训练数据。
 
使用Google TPU v3-256 训练2天得到；包含32个v3-8机器，每个v3-8机器包含128G的显存；训练了20万步，使用序列长度(sequence_length)512，批次(batch_size)为512。

#### 注意事项 Notices

XLNet_zh_Large还没有完整测试，可能在你的任务中有极好的表现，也可能在部分任务中有糟糕的表现。我们预计既会有好消息，也有坏消息；但目前在句子对任务中(LCQMC任务)是坏消息。

#### 提供您的测试对比 Performance
如果你使用本项目的中文预训练模型，请告诉你的测试对比效果：你可以直接发生pull request将你的任务中的测试对比添加到README.md中，或发在issue中；

你也可以加入中文预训练模型transformers讨论群(QQ:836811304)，并把测试对比告知我们。

XLNet中文预训练模型-下载 Download Pre-trained XLNet trained with Chinese, by Chinese, for Chinese
--------------------------------------------------------------
XLNet_zh_Large，<a href="">Google drive</a> 或 <a href="https://pan.baidu.com/s/1dy0Z27DoZdMpSmoz1Q4G5A">百度网盘</a>，TensorFlow版本

#### 如何保留从左到右的方式预测（就像传统的语言模型一样），但还能利用下文的信息？

    
    1.input_list:   [1, 2, 3, 4, 5, 6]
    2.sampled_list: [2, 4, 6, 5, 3, 1]
    3.array_2d:
                    [[0. 1. 1. 1. 1. 1.]
                     [0. 0. 0. 0. 0. 0.]
                     [0. 1. 0. 1. 1. 1.]
                     [0. 1. 0. 0. 0. 0.]
                     [0. 1. 0. 1. 0. 1.]
                     [0. 1. 0. 1. 0. 0.]]

    import numpy as np
    import random
    def xlnet_mask(input_list):
        """
        输入一个列表（如：[x1,x2,x3,x4]），采样到一个新的组合（如：[x3,x2,x4,x1]）返回一个矩阵
        要实现的是让当前单词Xi只能看到这个新顺序中自己前面的单词
        即：对于序列[x3,x2,x4,x1]
            x2能看到x3;
            x4能看到x3,x2
            x1能看到x3,x2,x4
            x3什么也看不到
        看到在程序里，是1，看不到是0.
        :param input_list:
        :return: matrix
        e.g
        [[0,1,1,1],  # x1
         [0,0,1,0],  # x2
         [0,0,0,0],  # x3
         [0,1,1,0]]  # x4
    
        """
        print("1.input_list:",input_list)
        random.shuffle(input_list) # 打乱循序
        sampled_list=input_list
        print("2.sampled_list:",sampled_list)
        num_size=len(input_list)
        
        array_2d=np.zeros((num_size,num_size))
        for index,current_element in enumerate(sampled_list):
            previous_element_list=sampled_list[0:index] # 被采样的组合中当前元素中自己前面的单词
            for previous_element in previous_element_list:
                array_2d[current_element-1][previous_element-1]=1
        
        print("3.array_2d:\n",array_2d)
        return array_2d
    
    input_list=[1,2,3,4,5,6]
    array_2d=xlnet_mask(input_list)
    

效果测试与对比 Performance
--------------------------------------------------------------
请您报告并添加。数据集或任务不限，包括XNLI、LCQMC、阅读理解数据集CMRC、CCF-Sentiment-Analysis等等。

模型加载（以Sentence Pair Matching即句子对任务，LCQMC为例）
--------------------------------------------------------------

预训练
--------------------------------------------------------------
1、生成tfrecords:

    SAVE_DIR=gs://xlnet_zh/tf_records_xlnet
    INPUT=gs://raw_text/data_2019_raw/*.txt 
    nohup python -u data_utils.py \
        --bsz_per_host=32 \
        --num_core_per_host=8 \
        --seq_len=512 \
        --reuse_len=256 \
        --input_glob=${INPUT} \
        --save_dir=${SAVE_DIR} \
        --num_passes=20 \
        --bi_data=True \
        --sp_path=spiece.model \
        --mask_alpha=6 \
        --mask_beta=1 \
        --num_predict=85 \
        --uncased=False \
        --num_task=200 \
        --task=1 &

2、训练模型:

    DATA=gs://xlnet_zh/tf_records_xlnet/tfrecords/
    MODEL_DIR=gs://xlnet_zh/xlnet_zh_large
    TPU_NAME=xlnet-zh-large-v3-256 
    TPU_ZONE=europe-west4-a
    nohup python train.py \
        --record_info_dir=$DATA \
        --model_dir=$MODEL_DIR \
        --train_batch_size=512 \
        --num_hosts=32 \
        --num_core_per_host=8 \
        --seq_len=512 \
        --reuse_len=256 \
        --mem_len=384 \
        --perm_size=256 \
        --n_layer=24 \
        --d_model=1024 \
        --d_embed=1024 \
        --n_head=16 \
        --d_head=64 \
        --d_inner=4096 \
        --untie_r=True \
        --mask_alpha=6 \
        --mask_beta=1 \
        --num_predict=85 \
        --uncased=False \
        --train_steps=200000 \
        --save_steps=3000 \
        --warmup_steps=10000 \
        --max_save=30 \
        --weight_decay=0.01 \
        --adam_epsilon=1e-6 \
        --learning_rate=1e-5 \
        --dropout=0.1 \
        --dropatt=0.1 \
        --tpu=$TPU_NAME \
        --tpu_zone=$TPU_ZONE \
        --use_tpu=True \
        --track_mean=True &


fine-tuning(以LCQMC任务为例)
--------------------------------------------------------------
    XLNET_DIR=gs://xlnet_zh/xlnet_zh_large
    MODEL_DIR=gs://xlnet_zh/fine_tuning_test/lcqmc_01
    DATA_DIR=gs://xlnet_zh/fine_tuning_test/lcqmc_01/lcqmc_tfrecords
    RAW_DIR=gs://roberta_zh/compare_model_performance/lcqmc
    TPU_NAME=grpc://03.06.08.09:8470
    TPU_ZONE=us-central1-a
    nohup python -u run_classifier.py \
        --spiece_model_file=./spiece.model \
        --model_config_path=${XLNET_DIR}/config.json \
        --init_checkpoint=${XLNET_DIR}/model.ckpt-192000 \
        --task_name=lcqmc \
        --do_train=True \
        --do_eval=True \
        --eval_all_ckpt=True \
        --uncased=False \
        --data_dir=${RAW_DIR} \
        --output_dir=${DATA_DIR} \
        --model_dir=${MODEL_DIR} \
        --train_batch_size=128 \
        --eval_batch_size=8 \
        --num_hosts=1 \
        --num_core_per_host=8 \
        --num_train_epochs=3 \
        --max_seq_length=128 \
        --learning_rate=2e-5 \
        --save_steps=1000 \
        --use_tpu=True \
        --tpu=${TPU_NAME} \
        --tpu_zone=${TPU_ZONE} >> xlnet_large_lcqmc_1.out &

    注: TPU_NAME is dummy, you should change IP to real one
	
Learning Curve 学习曲线
--------------------------------------------------------------
<img src="https://github.com/brightmart/xlnet_zh/blob/master/resources/XLNet_zh_Large.jpeg"  width="70%" height="60%" />

###### Research supported with Cloud TPUs from Google's TensorFlow Research Cloud (TFRC)


Reference
--------------------------------------------------------------
[1] <a href="https://arxiv.org/pdf/1906.08237.pdf">XLNet: Generalized Autoregressive Pretraining for Language Understanding</a>

[2] <a href="https://github.com/ymcui/Chinese-PreTrained-XLNet">Chinese-PreTrained-XLNet</a>

[3] <a href="https://new.qq.com/omn/20190624/20190624A0CDGK00">XLNet：运行机制及和Bert的异同比较</a>





