:: Notes on RNNs

IMDB data set from Keras

GRU + tanh: Test accuracy: 0.8272
GRU + sigmoid: Test accuracy: 0.8276

LSTM + sigmoid: Test accuracy: 0.77..

After increasing maxlen from 100 to 200:

GRU + 0.6 dropout: 0.8578 accuract

Doubling max_features gives 0.8646 accuracy

maxlen=400 and max_features=4000

0.8814 accuracy

0.888 accuracy with GRU(128, 128), Dropout(0.5), sigmoid activation, maxlen=400, max_feat=40000
