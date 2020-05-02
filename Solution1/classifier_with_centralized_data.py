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

n_train = 500
train_xs = vectorizer.transform(train_texts[:n_train])
train_xs = w2v_featurizer(reps_word2vec, train_xs)
train_xs[np.isnan(train_xs)] = 0

train_ys = to_categorical(train_labels[:n_train])

test_xs = vectorizer.transform(test_texts)
test_xs = w2v_featurizer(reps_word2vec, test_xs)
test_xs[np.isnan(test_xs)] = 0

test_ys = to_categorical(test_labels)

model = train_model(train_xs, train_ys, n_batch=50, n_epochs=32)
ACC = eval_model(model, test_xs, test_ys)
