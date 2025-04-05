#! /usr/bin/env python
# -*- coding: utf-8 -*-

from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D


import numpy as np
import pickle


string_list = []
mark_list = []
handle = open("post_data.txt", "r")
for i in range(17000):
    s = handle.readline()
    t = handle.readline()
    string_list.append(t.strip())
    m = handle.readline()
    try:
        l = int(m)
    except:
        l = 0
        continue
    if l > 14 and len(t) > 50:
        mark_list.append(1)
    else:
        mark_list.append(0)

tokenizer = Tokenizer(num_words=15000)
tokenizer.fit_on_texts(string_list)

with open('tokenizer.pickle', 'wb') as hand:
    pickle.dump(tokenizer, hand, protocol=pickle.HIGHEST_PROTOCOL)

print(len(string_list))
sequences = tokenizer.texts_to_sequences(string_list)

string_list_test = []
mark_list_test = []

for i in range(2500):
    s = handle.readline()
    t = handle.readline()
    string_list_test.append(t.strip())
    m = handle.readline()
    print(t)
    try:
        l = int(m)
    except Exception:
        l = 0
        continue
    if l > 14 and len(t) > 50:
        mark_list_test.append(1)
    else:
        mark_list_test.append(0)

sequences_test = tokenizer.texts_to_sequences(string_list_test)

x_train = sequence.pad_sequences(sequences, maxlen=400)
x_test = sequence.pad_sequences(sequences_test, maxlen=400)
for seq in sequences:
    print(len(seq))

print(len(x_test))
print(len(mark_list_test))

# print(x_train.shape)
# print(x_test.shape)

print('Build model...')
model = Sequential()

# We start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(20000,
                    50,
                    input_length=400))
model.add(Dropout(0.2))

# We add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(250,
                3,
                padding='valid',
                activation='relu',
                strides=1))
# We use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(250))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
# print("x_train shape:", x_train.shape)
# print("x_train dtype:", x_train.dtype)

x_train = np.array(x_train)
mark_list = np.array(mark_list)
x_test = np.array(x_test)
mark_list_test = np.array(mark_list_test)


model.fit(x_train, mark_list, batch_size=32, epochs=15, validation_data=(x_test, mark_list_test))
model.save("orbitar_base.h5")