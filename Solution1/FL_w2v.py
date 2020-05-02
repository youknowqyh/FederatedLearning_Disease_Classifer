import pandas as pd
import numpy as np
import lab_util
import word2vec
from keras.models import load_model
data = pd.read_json('train_data.json')
train_texts = data['notes'].iloc[0:500]

corpus = train_texts
Tokenizer = lab_util.Tokenizer()
Tokenizer.fit(corpus)
TokenizedData = Tokenizer.tokenize(corpus)
vocab_size = Tokenizer.vocab_size
vectorizer = lab_util.CountVectorizer()
vectorizer.fit(corpus)

weights_of_10_silos = []
for i in range(10):
    model = load_model('saved_model/model' + str(i + 1) + '.h5')
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
w2v = word2vec.Word2VecModel(vocab_size, embed_dim=500, hidden_size=16, context_size=5)
w2v.build_model()
w2v.Model_Pre.set_weights(Aggregated_weights)
w2v.Model_Pre.save('saved_model/FederatedLearning_Model.h5')
reps_word2vec = Aggregated_weights[0]
words = ["she", "drug", "obesity"]
show_tokens = [vectorizer.tokenizer.word_to_token[word] for word in words]
lab_util.show_similar_words(vectorizer.tokenizer, reps_word2vec, show_tokens)
