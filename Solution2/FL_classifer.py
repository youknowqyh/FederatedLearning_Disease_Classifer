import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from classifier import *
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Lambda
import os

maxlen = 2500 # cut the text after 2500 words
training_samples = 600  # train on 500 samples
test_samples = 100 # validate on 200 samples
max_words = 15000 # only consider the 15000 most frequent words

train_data = pd.read_json('train_data.json')
texts = train_data['notes']
labels = train_data['label']

tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ 1234567890')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=maxlen)

AllCategories = labels.unique().tolist()
AllCategories.append("Unknown")
CategoryToNumber = dict()
for x in range(len(AllCategories)):
    CategoryToNumber[AllCategories[x]] = x
NumberToCategory = dict()
for x in range(len(AllCategories)):
    NumberToCategory[x] = AllCategories[x]
labels, _ = Categorize(labels, AllCategories)
labels = to_categorical(labels)

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_test = data[training_samples: training_samples + test_samples]
y_test = labels[training_samples: training_samples + test_samples]

glove_dir = 'glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector # embeddings_index中找不到的词，其嵌入向量全为0

weights_of_10_silos = []
silo_size = int(training_samples / 10)
for i in range(10):
    model = build_model(embedding_matrix, max_words, maxlen, embedding_dim)
    x = x_train[i*silo_size:(i+1)*silo_size]
    y = y_train[i*silo_size:(i+1)*silo_size]
    model.fit(x, y, batch_size=12, epochs=6)
    weights = model.get_weights()
    weights_of_10_silos.append(weights)

Aggregated_weights = []
for idx_list, list in enumerate(weights_of_10_silos):
    if idx_list == 0:
        for w in list:
            Aggregated_weights.append(np.zeros(w.shape, dtype='float32'))
    for idx_w, w in enumerate(list):
        Aggregated_weights[idx_w] += w
for w in Aggregated_weights:
    w /= 10

model = build_model(embedding_matrix, max_words, maxlen, embedding_dim)
model.set_weights(Aggregated_weights)
for i in range(10):
    x = x_train[i*silo_size:(i+1)*silo_size]
    y = y_train[i*silo_size:(i+1)*silo_size]
    model.fit(x, y, batch_size=12, epochs=6)
eval_model(model, x_test, y_test)
