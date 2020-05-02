import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from classifier import Categorize, eval_model

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

import os
glove_dir = 'glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 300
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector # embeddings_index中找不到的词，其嵌入向量全为0

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Lambda
model = Sequential()
model.add(Embedding(max_words, embedding_dim, mask_zero=True, input_length=maxlen))
model.add(Lambda(lambda x: x, output_shape=lambda s:s))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.layers[0].set_weights([embedding_matrix])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, batch_size=32, epochs=6)
eval_model(model, x_test, y_test)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.title('Training accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.title('Training loss')
plt.legend()
plt.show()
