from keras.layers import Input, Dense, Activation
from keras.utils import to_categorical
from keras import utils
from keras.optimizers import Adam
from keras.models import Sequential, Model
import numpy as np
def Categorize(Candidates,AllItems):
    IndexID=[]
    Categories=[]
    for x in Candidates:
            if x not in AllItems:
                IndexID.append(len(AllItems))
                Categories.append("Unknown")
            else:
                for i in range(len(AllItems)):
                    if x==AllItems[i]:
                        IndexID.append(i)
                        Categories.append(AllItems[i])

    return (np.asarray(IndexID),np.asarray(Categories))
### train classifier
def w2v_featurizer(reps_word2vec, train_xs):
    # 就是先把文档转换为bag of words的形式
    # train_xs[i].astype("bool")的作用是筛选出这一行文本中所有出现的单词
    # 然后到reps_word2vec中就行查找这些行数（一行是一个单词的词向量表示）
    # 于是我们得到了一个新的token-document的矩阵
    # 压缩行，对所有行的元素取平均
    # 我们就得到一个500列的向量
    # 这个向量就代表一个文档。
    # 最后对feats进行正则化
    RepMat = []
    for i in range(train_xs.shape[0]):
        RepMat.append(np.mean(reps_word2vec[train_xs[i].astype("bool"),:], 0))
    feats = np.stack(RepMat)
    # normalize
    return feats / np.sqrt((feats ** 2).sum(axis=1, keepdims=True))

def train_model(train_xs, train_ys, n_batch=500, n_epochs=32):

    model = Sequential([
        Dense(32, input_shape=(train_xs.shape[1],)),
        Activation('relu'),
        Dense(16),
        Activation('relu'),
        Dense(train_ys.shape[1]),
        Activation('softmax'),
    ])

    model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    model.summary()
    model.fit(train_xs, train_ys, batch_size=n_batch,
                epochs=n_epochs)
    return model

def eval_model(model, train_xs, train_ys):
    pred_ys = model.predict(train_xs)
    pred_ys[np.isnan(pred_ys)] = 0
    ACC = np.sum(np.argmax(pred_ys, 1) == np.argmax(train_ys==1, 1)) / train_ys.shape[0]
    print("test accuracy", ACC)
    return(ACC)