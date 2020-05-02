# FederatedLearning_Disease_Classifer

### Introduction

it'a a classifier deciding the status of a patient's disease(Y, N, U, Q) according to the medical record.

### Steps

1. split data into 10 silos (randomly) 
2. pick one label (diabetes) , implement federated averaging to conduct the training. 
3. train the model and compare its performance with model trained with centralized data. 

### Details

1. Solution1

Firstly, based on CBOW, I train a w2v embedding by federated learning.

Then use the embedding to featurize my texts to 500-D vectors, which are inputs of my classifer.

2. Solution2

Concatenate the embedding layer and the neural network of classifiers

Use glovec as word embedding.

Use keras.preprocessing.sequence.pad_sequences to padd the text into vectors with certain length.

### Accuracy

≈ 0.68
