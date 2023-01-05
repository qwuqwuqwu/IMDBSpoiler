# Reference
# https://www.tensorflow.org/tutorials/keras/text_classification

import numpy as np
import pandas as pd
import tensorflow as tf
import os
import re
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# -- IMPORT DATA --
dataset_dict = {}

# Add dictionary with all files and their relative paths
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         dataset_dict[filename] = os.path.join(dirname, filename)

# -- PREPROCESS DATA --
review_file_name = 'IMDB_reviews_small.json'
dataset_dict = {}
dataset_dict[review_file_name] = os.path.join("", review_file_name)
dataset_dict["IMDB_movie_details.json"] = os.path.join("", "IMDB_movie_details.json")

print(dataset_dict.keys())
df_reviews = pd.read_json(dataset_dict.get(review_file_name), lines=True)
df_movies = pd.read_json(dataset_dict.get("IMDB_movie_details.json"), lines=True)

df_movies['is_spoiler'] = 1 # All plot synopsis will be a spoiler

df_review_columns_X = [df_reviews['review_text'], df_movies['plot_synopsis']]
df_review_columns_Y = [df_reviews['is_spoiler'].astype(int), df_movies['is_spoiler'].astype(int)]

df_reviews_X = pd.concat(df_review_columns_X,ignore_index=True)
df_reviews_Y = pd.concat(df_review_columns_Y,ignore_index=True)

pd.to_numeric(df_reviews_Y)

print(df_reviews_X)
print(df_reviews_Y)


#Shuffle the data
data_size = len(df_reviews_X)
permutation = list(np.random.permutation(data_size))
df_reviews_X = df_reviews_X[permutation]
df_reviews_Y = df_reviews_Y[permutation]

df_reviews_X = np.array(df_reviews_X).tolist()
df_reviews_Y = np.array(df_reviews_Y).tolist()

docs = df_reviews_X.copy()
print(docs[0])

# prepare tokenizer
from keras.preprocessing.text import Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n') # encode sequences of words with index
tokenizer.fit_on_texts(docs) # create a hash map of numbers and words, word2idx & idx2word
sequences = tokenizer.texts_to_sequences(docs) # shape = (# of docs, length of text)
print(sequences[0])
print(docs[0])
vocabulary_size = len(tokenizer.word_counts)
print("The size of vocab.txt is", vocabulary_size)
vocabulary_size = 10000

def get_max_len(seq):
  buffer = seq.copy()
  max = 0
  for i in buffer:
    temp = len(i)
    if temp > max:
      max = temp
  return max

max_seq_len = get_max_len(sequences)
print(max_seq_len)

# since the max sequence length of the corpus is 4 (doc[9]),
# we are going to made the max_seq_len = 4 -> PADDING
from keras_preprocessing.sequence import pad_sequences

sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')
print(len(sequences[0]))

# hyper-parameters
hidden_dim = 50 # glove.6B.50d.txt "50d"
# load the whole embedding into memory
embeddings_index = dict()

f = open('glove.6B.50d.txt', encoding='utf-8')
for line in f:
  values = line.split()
  word = values[0]
  embeddings_index[word] = np.array(values[1:], dtype='float32')
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocabulary_size+1, hidden_dim))
j=0
for word, i in tokenizer.word_index.items():
  if i > vocabulary_size:
    break
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector
  else:
    j = j + 1
print(j)

#Split data into train, valid, test

data_size = sequences.shape[0]

train_data_size = round(data_size * 0.6)
valid_data_size = round(data_size * 0.3)
test_data_size = round(data_size * 0.1)

X_train = sequences[0:train_data_size,:]
X_valid = sequences[train_data_size:train_data_size + valid_data_size,:]
y_train = np.expand_dims(np.array(df_reviews_Y[0:train_data_size]), axis=1)
y_valid = np.expand_dims(np.array(df_reviews_Y[train_data_size:train_data_size + valid_data_size]), axis=1)

X_test = sequences[train_data_size + valid_data_size:train_data_size + valid_data_size + test_data_size,:]
y_test = np.expand_dims(np.array(df_reviews_Y[train_data_size + valid_data_size:train_data_size + valid_data_size + test_data_size]), axis=1)

def build_model_LSTM(vocabulary_size, hidden_dim, max_seq_len, LSTM_dim):
  model = models.Sequential()
  model.add(layers.Input(shape=(max_seq_len)))
  model.add(layers.Embedding(input_dim=vocabulary_size, output_dim=hidden_dim, weights=[embedding_matrix], input_length=max_seq_len, trainable=False))
  model.add(layers.LSTM(LSTM_dim, dropout=0.5, return_sequences = True))
  model.add(layers.LSTM(LSTM_dim, dropout=0.5))
  # model.add(layers.Flatten())
  model.add(layers.Dense(1, activation='sigmoid'))

  model.summary()
  return model

model_LSTM = build_model_LSTM(vocabulary_size+1, hidden_dim, max_seq_len, 50)

model_LSTM.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

unique, count = np.unique(y_train, return_counts=True)

weight_for_0 = count[0] / len(y_train)
weight_for_1 = count[1] / len(y_train)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

# history = model_embedding.fit(X_train, y_train, epochs=10, validation_data=[X_valid, y_valid])
history = model_LSTM.fit(X_train, y_train, epochs=20, validation_data=[X_valid, y_valid], class_weight = class_weight)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()