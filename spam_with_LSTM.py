import numpy as np
import numpy.random as random
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from spam_utils import load_data

train_data,train_label, val_data,val_label, test_data,test_label,word_idx = load_data(3144, 1114, 1114, tokenizer=True)


maxlen = 30
train_data = pad_sequences(train_data, maxlen=maxlen) ##  pad (3345, 30)  3345samples，30
val_data   = pad_sequences(val_data,   maxlen=maxlen) ##      (1114, 30)
test_data  = pad_sequences(test_data,  maxlen=maxlen) ##      (1114, 30)


#(embedding_dim = 100)
embedding_idx = {}
with open('data/glove.6B.100d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_idx[word] = coefs
print('{} word vectors.'.format(len(embedding_idx)))


max_words = 10000
embedding_dim = 100
embedding_mtx = np.zeros((max_words, embedding_dim))  #(10000, 100)
for word, i in word_idx.items():
    if i < max_words:
        embedding_vector = embedding_idx.get(word)
        if embedding_vector is not None:
            embedding_mtx[i] = embedding_vector

# print(embedding_mtx)


from keras.layers import LSTM, Embedding, Dense, Flatten
from keras.models import Sequential, Model

model = Sequential()
model.add(Embedding(10000, embedding_dim, input_length=maxlen))
model.add(LSTM(embedding_dim))
model.add(Dense(1, activation='sigmoid', name="Dense1"))
print(model.summary())

model.layers[0].set_weights([embedding_mtx])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
History = model.fit(train_data, 
                    train_label, 
                    epochs=20,
                    batch_size=64, 
                    validation_data=(val_data, val_label))

last_layer = Model(inputs=model.input, outputs=model.get_layer('Dense1').output)  
res_train = last_layer.predict(train_data)
res_val = last_layer.predict(val_data)
res_test = last_layer.predict(test_data)


from spam_utils import draw_loss_acc, draw_PR, draw_ROC,print_PR_F1_score
print_PR_F1_score(res_test, test_label, "LSTM")
draw_loss_acc(history_dict=History.history)
draw_PR(res_test, test_label, name="LSTM")
draw_ROC(pred=res_test, label=test_label, name="LSTM")