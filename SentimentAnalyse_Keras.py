# -*- coding: utf-8 -*-
import jieba
import pandas as pd         # 导入Pandas
import re
import collections
import numpy as np

from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

import random


class KerasAnalyse:
    debug = True            # 当前程序是否用于实验测试

    sentences = []          # 训练集
    labels = []             # 训练集对应标签
    sentence_tokens = []    # 保存每个句子分词结果

    maxlen = 0                              # 句子最大长度
    word_freqs = collections.Counter()      # 词频
    num_recs = 0                            # 样本数

    # 超属性，根据测试样本情况
    MAX_FEATURES = 48000                    # nb_words  48931
    MAX_SENTENCE_LENGTH = 900               # max_len   966

    EMBEDDING_SIZE = 128                    # 此两参数凭经验调优
    HIDDEN_LAYER_SIZE = 64

    BATCH_SIZE = 32
    NUM_EPOCHS = 10

    def __init__(self):
        print("initial ...")

    # 读取训练语料
    def readTestData(self):
        neg = pd.read_excel('neg.xls', header=None, index=None)
        pos = pd.read_excel('pos.xls', header=None, index=None)
        for i in range(len(pos)):
            sentence = pos.get(0)[i]
            if sentence != "":                  # 空的评论不保存
                self.sentences.append(sentence)
                self.labels.append(1)
        for i in range(len(neg)):
            sentence = neg.get(0)[i]
            if sentence != "":
                self.sentences.append(sentence)
                self.labels.append(0)
        # print(neg.get(0)[0])
        # print(pos.get(0)[0])

    # 预执行，对语料进行分词、去停用词，相关量统计
    def pre_process(self):
        self.readTestData()
        stopwords = {}.fromkeys([line.rstrip() for line in open('stopwords_HIT.txt')])  # 导入哈工大中文停用词语库

        for line in self.sentences:
            seg_list = jieba.cut(line, cut_all=False)       # 分词，cut_all=false 时为全模式
            final_list = []

            for seg in seg_list:                            # 剔除停用词
                if seg not in stopwords:
                    if not re.match(r"\d+\w{0,2}\W*\d*$", seg) and not seg == " ":      # 去掉空格、纯数字以及表示质量或容量的词
                        final_list.append(seg)
            self.sentence_tokens.append(final_list)

            if len(final_list) > self.maxlen:       # 获得句子长度
                self.maxlen = len(final_list)
            for word in final_list:                 # 统计词频
                self.word_freqs[word] += 1
            self.num_recs += 1

        print('max_len ', self.maxlen)              # 句子最大长度
        print('nb_words ', len(self.word_freqs))    # 词计数

    # 执行
    def train(self):
        # 准备数据
        vocab_size = min(self.MAX_FEATURES, len(self.word_freqs)) + 2             # 外加一个伪单词 UNK 和填充单词 0
        word2index = {x[0]: i + 2 for i, x in enumerate(self.word_freqs.most_common(self.MAX_FEATURES))}
        word2index["PAD"] = 0
        word2index["UNK"] = 1
        index2word = {v: k for k, v in word2index.items()}

        X = np.empty(self.num_recs, dtype=list)
        y = np.zeros(self.num_recs)
        i = 0

        for line in self.sentence_tokens:
            # print("分词结果：", line, "; 标签", self.labels[i])
            seqs = []
            for word in line:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            X[i] = seqs                             # 保存句子的数字序列
            y[i] = int(self.labels[i])              # 对应的标签
            i += 1
        X = sequence.pad_sequences(X, maxlen=self.MAX_SENTENCE_LENGTH)

        ## 数据划分
        Xtrain, Xtest , ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)       # 随机划分训练集和测试集

        ## 网络构建
        model = Sequential()
        model.add(Embedding(vocab_size, self.EMBEDDING_SIZE, input_length=self.MAX_SENTENCE_LENGTH))
        model.add(LSTM(self.HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1))
        model.add(Activation("sigmoid"))
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        ## 网络训练
        model.fit(Xtrain, ytrain, batch_size=self.BATCH_SIZE, epochs=self.NUM_EPOCHS, validation_data=(Xtest, ytest))

        ## 预测
        score, acc = model.evaluate(Xtest, ytest, batch_size=self.BATCH_SIZE)

        ## 模型保存
        model.save('model_chinese_appraiseOfGoods.h5')

        print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
        print('{}   {}      {}'.format('预测', '真实', '句子'))
        for i in range(5):  # 从测试集抽样部分数据进行预测
            idx = np.random.randint(len(Xtest))
            xtest = Xtest[idx].reshape(1, self.MAX_SENTENCE_LENGTH)
            ylabel = ytest[idx]
            ypred = model.predict(xtest)[0][0]  # 调用模型进行预测
            sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
            print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))
        # end of train

    def predict(self):
        ## 加载模型
        model = load_model('model_chinese_appraiseOfGoods.h5')

        stopwords = {}.fromkeys([line.rstrip() for line in open('stopwords_HIT.txt')])  # 导入哈工大中文停用词语库
        comments = pd.read_excel('sum.xls')                                             # 读入评论内容
        comments = comments[comments['rateContent'].notnull()]                          # 仅读取非空评论
        # print(comments['rateContent'])

        INPUT_SENTENCES = []                # 保存测试评论
        input_comments = []                 # 用来保存输入肥None评论集(测试时使用)

        for comment in comments['rateContent']:
            # print(comment)
            if self.debug and comment is not None:                  # 测试时，对评论内容非空的添加集合
                input_comments.append(comment)
            seg_list = jieba.cut(comment, cut_all=False)            # 分词，cut_all=false 时为全模式
            final_list = []
            for seg in seg_list:  # 剔除停用词
                if seg not in stopwords:
                    if not re.match(r"\d+\w{0,2}\W*\d*$", seg) and not seg == " ":  # 去掉空格、纯数字以及表示质量或容量的词
                        final_list.append(seg)
            INPUT_SENTENCES.append(final_list)

        # 准备数据
        word2index = {x[0]: i + 2 for i, x in enumerate(self.word_freqs.most_common(self.MAX_FEATURES))}
        word2index["PAD"] = 0
        word2index["UNK"] = 1

        XX = np.empty(len(INPUT_SENTENCES), dtype=list)
        i = 0
        for line in INPUT_SENTENCES:
            seqs = []
            for word in line:
                if word in word2index:
                    seqs.append(word2index[word])
                else:
                    seqs.append(word2index["UNK"])
            XX[i] = seqs
            i += 1

        XX = sequence.pad_sequences(XX, maxlen=self.MAX_SENTENCE_LENGTH)
        labels = [int(round(x[0])) for x in model.predict(XX)]
        label2word = {1: '积极', 0: '消极'}
        for i in range(len(INPUT_SENTENCES)):
            print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))

        if self.debug:
            f = open("out.txt", "w")            # 打开文件
            step = int(round(len(INPUT_SENTENCES) / 100))
            for i in range(0,100):
                index = i * 100 + int(round(random.random() * step))
                print('{}   {}   {}'.format(label2word[labels[index]], INPUT_SENTENCES[index], input_comments[index]), file=f)


#### 单元测试
kerasUnitTest = KerasAnalyse()
kerasUnitTest.readTestData()
kerasUnitTest.pre_process()
# kerasUnitTest.train()
kerasUnitTest.predict()
