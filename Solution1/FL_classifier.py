#region import various module
import lab_util
import pandas as pd
import numpy as np
from classifier import *

#endregion

# region Prepare data
data = pd.read_json('train_data.json')
train_texts = data['notes'].iloc[0:500]
train_labels = data["label"].iloc[0:500]
test_texts = data['notes'].iloc[500:700]
test_labels = data["label"].iloc[500:700]
AllCategories = train_labels.unique().tolist()
AllCategories.append("Unknown")
CategoryToNumber = dict()
for x in range(len(AllCategories)):
    CategoryToNumber[AllCategories[x]] = x
NumberToCategory = dict()
for x in range(len(AllCategories)):
    NumberToCategory[x] = AllCategories[x]
train_labels, _ = Categorize(train_labels, AllCategories)
test_labels, _ = Categorize(test_labels, AllCategories)
corpus = train_texts
Tokenizer = lab_util.Tokenizer()
Tokenizer.fit(corpus)
TokenizedData = Tokenizer.tokenize(corpus)
vocab_size = Tokenizer.vocab_size
vectorizer = lab_util.CountVectorizer()
vectorizer.fit(corpus)
# endregion

from keras.models import load_model
model = load_model('saved_model/FederatedLearning_Model.h5')
model.summary()
weights = model.get_weights()
reps_word2vec = weights[0]


weights_of_10_silos = []
for i in range(10):
    model = load_model('saved_model/classifier_model' + str(i + 1) + '.h5')
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

from keras.layers import Input, Dense, Activation
from keras.utils import to_categorical
from keras import utils
from keras.optimizers import Adam
from keras.models import Sequential, Model
model = Sequential([
    Dense(32, input_shape=(500,)),
    Activation('relu'),
    Dense(16),
    Activation('relu'),
    Dense(4),
    Activation('softmax'),
])
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
model.summary()
model.set_weights(Aggregated_weights)
model.save('saved_model/FL_Classifier_Model.h5')

test_xs = vectorizer.transform(test_texts)
test_xs = w2v_featurizer(reps_word2vec, test_xs)
test_xs[np.isnan(test_xs)] = 0
test_ys = to_categorical(test_labels)
ACC = eval_model(model, test_xs, test_ys)
