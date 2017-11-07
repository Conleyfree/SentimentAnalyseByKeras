# -*- coding: utf-8 -*-
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
import nltk
import numpy as np
import sys

## EDA
maxlen = 0                              # 句子最大长度
word_freqs = collections.Counter()      # 词频
num_recs = 0                            # 样本数
with open('traindata.txt','r+', encoding='UTF-8') as f:
    for line in f:
        label, sentence = line.strip().split('\t', 1)
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            word_freqs[word] += 1
        num_recs += 1
print('max_len ',maxlen)
print('nb_words ', len(word_freqs))

## 准备数据
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2         # 外加一个伪单词 UNK 和填充单词 0
word2index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word2index["PAD"] = 0
word2index["UNK"] = 1
index2word = {v:k for k, v in word2index.items()}

X = np.empty(num_recs,dtype=list)
y = np.zeros(num_recs)
i = 0
with open('traindata.txt','r+', encoding='UTF-8') as f:
    for line in f:
        label, sentence = line.strip().split('\t', 1)
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])
        X[i] = seqs             # 保存句子的数字序列
        y[i] = int(label)       # 对应的标签
        i += 1
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)

## 数据划分
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)       # 随机划分训练集和测试集

## 网络构建
EMBEDDING_SIZE = 128        # 此两参数凭经验调优
HIDDEN_LAYER_SIZE = 64

BATCH_SIZE = 32
NUM_EPOCHS = 10

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

## 网络训练
model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(Xtest, ytest))

## 预测
score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)

## 重定向标准输出
result_handle = open('result.log', 'w')
stdout_temp = sys.stdout
sys.stdout = result_handle

print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
print('{}   {}      {}'.format('预测','真实','句子'))
for i in range(5):                          # 从测试集抽样部分数据进行预测
    idx = np.random.randint(len(Xtest))
    xtest = Xtest[idx].reshape(1,40)
    ylabel = ytest[idx]
    ypred = model.predict(xtest)[0][0]                                          # 调用模型进行预测
    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
    print(' {}      {}     {}'.format(int(round(ypred)), int(ylabel), sent))

##### 自己输入
INPUT_SENTENCES = ['I love reading.','You are so boring.','I hate reading','I hate angelina jolie','Da Vinci love angelina jolie']

##### 使用 Kaggle 测试集
## with open('testdata.txt','r+', encoding='UTF-8') as f:
##    for line in f:
##        INPUT_SENTENCES.append(line)

XX = np.empty(len(INPUT_SENTENCES),dtype=list)
i=0
for sentence in INPUT_SENTENCES:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        if word in word2index:
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i+=1

XX = sequence.pad_sequences(XX, maxlen=MAX_SENTENCE_LENGTH)
# labels = [int(round(x[0])) for x in model.predict(XX) ]
labels = [x[0] for x in model.predict(XX)]
label2word = {1:'积极', 0:'消极'}
for i in range(len(INPUT_SENTENCES)):
    print('{}   {}'.format(labels[i], INPUT_SENTENCES[i]).encode(encoding='utf-8'))
    # print('{}   {}'.format(label2word[labels[i]], INPUT_SENTENCES[i]))

## 标准输出返回到控制台
sys.stdout = stdout_temp
print("运行结束，结果已保存至 result.log 文件。")