from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import itertools
import datetime

from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda, Dense, concatenate, Flatten
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

## Load Data
# File paths
file_dir = '../data/'
embedding_dir = '.../file/dir/containing/GoogleNews-vectors-negative300.bin/'

train_j = []
with open(file_dir+'train.json') as f:
    for line in f:
        train_j.append(eval(line))
# train_j (dict): 'edges', 'nodes', 'tokens'
test_j = []
with open(file_dir+'test.json') as f:
    for line in f:
        test_j.append(eval(line))
        
# Pre-trained embedding
embedding_file = embedding_dir + 'GoogleNews-vectors-negative300.bin'

def json_to_df(train_j):
    word_1 = []
    word_2 = []
    rel = []
    for i in range(len(train_j)):
        for j in range(len(train_j[i]['edges'])):
            tmp_1 = ''
            token_ix = train_j[i]['edges'][j][0][0]
            while(token_ix < train_j[i]['edges'][j][0][1]):
                #tmp_1.append(train_j[i]['tokens'][token_ix])
                tmp_1 += (str(train_j[i]['tokens'][token_ix]) + ' ')
                token_ix += 1
            tmp_2 = ''
            token_ix = train_j[i]['edges'][j][1][0]
            while(token_ix < train_j[i]['edges'][j][1][1]):
                #tmp_2.append(train_j[i]['tokens'][token_ix])
                tmp_2 += (str(train_j[i]['tokens'][token_ix]) + ' ')
                token_ix += 1
            
            word_1.append(tmp_1)
            word_2.append(tmp_2)
            rel.append(list(train_j[i]['edges'][j][2].keys())[0])
    
    train_df = pd.DataFrame()
    train_df['word_1'] = word_1
    train_df['word_2'] = word_2
    train_df['rel'] = rel
    
    return train_df

train_df = json_to_df(train_j)
test_df = json_to_df(test_j)


## Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<unk>']  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary=True)

term_cols = ['word_1', 'word_2']
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():
        for term in term_cols:
            t2n = []  # t2n -> term numbers representation
            for word in row[term]:
            
                # Check for unwanted words
                if word not in word2vec.vocab:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    t2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    t2n.append(vocabulary[word])

            # Replace terms as word to term as number representation
            dataset.set_value(index, term, t2n)
            
embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # The embedding matrix
embeddings[0] = 0  # Ignore the padding

# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec


# Encode relation types to labels
rel_type = list(set(train_df['rel']))
rel2label = dict()
label2rel = dict()
for i in range(len(rel_type)):
    rel2label[rel_type[i]] = i
    label2rel[i] = rel_type[i]

def encode_relation(df):
    rel_list = []
    for i in range(len(df)):
        rel_list.append( rel2label[df['rel'][i]] )

    return rel_list
    
def one_hot_relation(df):
    y_train = np.zeros((len(df), len(rel_type)))
    for i in range(len(df)):
        y_train[i, rel2label[df['rel'][i]]] = 1

    return y_train

y_train = one_hot_relation(train_df)


# Organize training data: zero padding to max_seq_length
seq_len = [len(train_df['word_1'][i]) for i in range(len(train_df))]
max_seq_length = int(np.percentile(seq_len, 75))
print ('max sequence length: ', max_seq_length)

X = train_df[term_cols]
X_train = X
#X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1)

# Split to dicts
X_train = {'left': X_train.word_1, 'right': X_train.word_2}
#X_validation = {'left': X_validation.word_1, 'right': X_validation.word_2}
X_test = {'left': test_df.word_1, 'right': test_df.word_2}

# Zero padding
for dataset, side in itertools.product([X_train, X_test], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(y_train)

## Encode input words
def input_embedding(x):  # e.g. x=X_train['left']
    encoded = np.zeros((len(x), embedding_dim))
    for i in range(len(x)):
        for j in range(max_seq_length):
            encoded[i] += embeddings[x[i][j]]
    
    return encoded/max_seq_length

X_train_left_emb = input_embedding(X_train['left'])
X_train_right_emb = input_embedding(X_train['right'])


### Model: MLP, multiclass classification
# The visible layer
left_input = Input(shape=(embedding_dim,), dtype='float32')
right_input = Input(shape=(embedding_dim,), dtype='float32')

# Embedded version of the inputs
#embedding_layer = Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_length=max_seq_length, trainable=False)
#encoded_left = embedding_layer(left_input)
#encoded_right = embedding_layer(right_input)

# Def model
input_size=max_seq_length
fc1 = Dense(64, activation='relu')(left_input)
fc2 = Dense(64, activation='relu')(right_input)
fc = concatenate([fc1, fc2])
fc = Dense(64, activation='relu')(fc)
fc = Dense(32, activation='relu')(fc)
output = Dense(3, activation='softmax')(fc)

model = Model(inputs=[left_input, right_input], outputs=[output])
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([X_train_left_emb, X_train_right_emb], [y_train],
          epochs=20, batch_size=64)


## Prediction
X_test_left_emb = input_embedding(X_test['left'])
X_test_right_emb = input_embedding(X_test['right'])

y_predict = model.predict([X_test_left_emb, X_test_right_emb])
y_predict = np.argmax(y_predict, 1)

from sklearn.metrics import f1_score
y_true = encode_relation(test_df)
f1_micro = f1_score(y_true, y_predict, average='micro')
f1_macro = f1_score(y_true, y_predict, average='macro')

print ('f1-score (micro): ', f1_micro)
print ('f1-score (macro): ', f1_macro)
