import pandas as pd         # 导入Pandas
import numpy as np          # 导入Numpy
import jieba                # 导入结巴分词

import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

print("Keras version : ", keras.__version__)

neg = pd.read_excel('neg.xls', header=None, index=None)
pos = pd.read_excel('pos.xls', header=None, index=None)     # 读取训练语料完毕

# 给训练语料贴上标签
pos['mark'] = 1
neg['mark'] = 0

pn = pd.concat([pos, neg], ignore_index=True)               # 合并语料

# 计算语料数目
neglen = len(neg)
poslen = len(pos)

# print("neg :", neg)
# print("pos :", pos)
# print(pn)

cw = lambda x: list(jieba.cut(x))                           # 定义分词函数
pn['words'] = pn[0].apply(cw)

comment = pd.read_excel('sum.xls')                          # 读入评论内容
comment = comment[comment['rateContent'].notnull()]         # 仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw)         # 评论分词

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index=True)

# 将所有词语整合在一起
w = []
for i in d2v_train:
  w.extend(i)
# print(w)

dict = pd.DataFrame(pd.Series(w).value_counts())            # 统计词的出现次数
del w, d2v_train

dict['id'] = list(range(1, len(dict)+1))
# print("dict['id'] =", dict['id'])

get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)                    # 速度太慢

maxlen = 50

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

x = np.array(list(pn['sent']))[::2]         # 训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2]       # 测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent']))             # 全集
ya = np.array(list(pn['mark']))

# 参数设置
vocab_dim = 100     # 向量维度
batch_size = 32
n_epoch = 5
input_length = maxlen

print('Build model...')
model = Sequential()

# model.add(Embedding(output_dim=vocab_dim, input_dim=10553, mask_zero=True, input_length=input_length))
# model.add(LSTM(activation="sigmoid", units=vocab_dim, recurrent_activation="hard_sigmoid"))
# model.add(Dropout(0.5))
model.add(Dense(10553, activation='relu', input_shape=(50, )))
model.add(Dropout(0.5))
model.add(Dense(5111, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# For a binary classification problem  # 优化器函数 optimizer ; 损失函数 loss ; 指标列表 metrics
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
model.fit(x, y, epochs=10, batch_size=100)