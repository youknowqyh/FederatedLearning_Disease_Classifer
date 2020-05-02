import pandas as pd
import numpy as np
import lab_util
import word2vec

data = pd.read_json('train_data.json')
train_texts = data['notes'].iloc[0:500]

corpus = train_texts
Tokenizer = lab_util.Tokenizer()
Tokenizer.fit(corpus)
TokenizedData = Tokenizer.tokenize(corpus)
vocab_size = Tokenizer.vocab_size
vectorizer = lab_util.CountVectorizer()
vectorizer.fit(corpus)

for i in range(10):
    train_texts = data['notes'].iloc[i*50:(i+1)*50]
    corpus = train_texts
    tokenized_corpus = Tokenizer.tokenize(corpus)
    current_model = word2vec.get_w2v_model(tokenized_corpus, vocab_size, embed_dim=500, hidden_size=16, context_size=5, n_batch=500, n_epochs=3)
    current_model.save('saved_model/model' + str(i + 1) + '.h5')
