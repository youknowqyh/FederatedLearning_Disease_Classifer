from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Lambda
from keras import utils
from keras.optimizers import Adam
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


def eval_model(model, train_xs, train_ys):
    pred_ys = model.predict(train_xs)
    pred_ys[np.isnan(pred_ys)] = 0
    ACC = np.sum(np.argmax(pred_ys, 1) == np.argmax(
        train_ys == 1, 1)) / train_ys.shape[0]
    print("test accuracy", ACC)
    return(ACC)


def build_model(embedding_matrix, max_words, maxlen, embedding_dim):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim,
                        mask_zero=True, input_length=maxlen))
    model.add(Lambda(lambda x: x, output_shape=lambda s: s))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.layers[0].set_weights([embedding_matrix])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model
