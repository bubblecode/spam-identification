import numpy as np
import numpy.random as random
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from spam_utils import load_data

train_data,train_label, val_data,val_label, test_data,test_label,word_idx = load_data(3345, 1114, 1114, tokenizer=True)


max_words = 10000
maxlen = 30
embedding_dim = 100  

train_data = pad_sequences(train_data, maxlen=maxlen) ## pad  (3344, 30)  3345 samples
val_data   = pad_sequences(val_data, maxlen=maxlen)   ##      (1114, 30)
test_data  = pad_sequences(test_data, maxlen=maxlen)  ##      (1114, 30)


from keras.models import Sequential, Model
from keras import layers
# from keras.optimizers import RMSprop   optimizer=RMSprop(learning_rate='1e-4')：AUC -- 0.62

model = Sequential()
model.add(layers.Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(32, 4, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 4, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1, name="Dense1"))

model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
History = model.fit(train_data, 
                    train_label, 
                    epochs=20, 
                    batch_size=128,
                    validation_split=0.3) 

last_layer = Model(input = model.input, output = model.get_layer('Dense1').output)  
res_train = last_layer.predict(train_data)
res_val = last_layer.predict(val_data)
res_test = last_layer.predict(test_data)


from spam_utils import draw_loss_acc, draw_PR, draw_ROC
draw_loss_acc(history_dict=History.history)
draw_PR(res_test, test_label, name="CNN")
draw_ROC(pred=res_test, label=test_label, name="CNN")