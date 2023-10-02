import pandas as pd
import numpy as np
import re
import string
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import keras
from tensorflow.python.keras.models import Input
from keras.models import Model, Input, Sequential
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
import keras as k
from tensorflow.keras.callbacks import EarlyStopping
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import os

for dirname, _, filenames in os.walk('/home/sami/PycharmProject/Datascience '):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv("/Inco_dataset/1400_inco_split_space (1).csv")
data.rename(columns={'Tokens': 'token'}, inplace=True)
data.rename(columns={'Tags': 'tag'}, inplace=True)
data.rename(columns={'Filename': 'text'}, inplace=True)

len(set(data['text']))

words = list(set(data["token"].values))
words.append("ENDPAD")
n_words = len(words)
print(n_words)

tags = list(set(data["tag"].values))
n_tags = len(tags)
print(n_tags)


class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["token"].values.tolist(),
                                                     # s["POS"].values.tolist(),
                                                     s["tag"].values.tolist())]
        self.grouped = self.data.groupby("text").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter_data = SentenceGetter(data)
print(getter_data)

sent_train = getter_data.get_next()
sentences_train = getter_data.sentences

data.groupby(['text']).count().max()
max_len = 1000
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: w for w, i in tag2idx.items()}

print(idx2tag)
print(tag2idx["O"])

# pad the sequence - train sample
X_train = [[word2idx[w[0]] for w in s] for s in sentences_train]
X_train = pad_sequences(maxlen=max_len, sequences=X_train, padding="post", value=n_words - 1)

# pad the target - train sample
y_train = [[tag2idx[w[1]] for w in s] for s in sentences_train]
y_train = pad_sequences(maxlen=max_len, sequences=y_train, padding="post", value=tag2idx["O"])
y_train = [to_categorical(i, num_classes=n_tags) for i in y_train]
# y_dev = [to_categorical(i, num_classes=n_tags) for i in y_dev]

early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
model = Sequential()
model.add(Embedding(input_dim=n_words + 1, output_dim=128, input_length=max_len))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(units=64, return_sequences=True, recurrent_dropout=0.1)))
model.add(TimeDistributed(Dense(n_tags, activation="relu")))
crf_layer = CRF(n_tags)
model.add(crf_layer)

print(model.summary())

adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)
model.compile(optimizer='adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

history = model.fit(X_train, np.array(y_train), batch_size=128, epochs=10,
                    validation_data=(X_test, np.array(y_test)), verbose=1)

model.save("model.h5")

hist = pd.DataFrame(history.history)

fig, ax = plt.subplots()
plt.style.use("ggplot")
plt.figure(figsize=(12, 12))
l1, = ax.plot(hist["crf_viterbi_accuracy"], label='Train sample', color='blue', linewidth=2)
l2, = ax.plot(hist["val_crf_viterbi_accuracy"], label='Validation sample', color='red', linewidth=2)
ax.set(xlabel='Epoch', ylabel='CRF Viterbi Accuracy')
ax.legend((l1, l2), ('Train sample', 'Validation sample'))
print(plt.show())

train_pred = model.predict(X_train, verbose=1)
test_pred = model.predict(X_test, verbose=1)


def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PAD", "O"))
        out.append(out_i)
    return out


train_pred_labels = pred2label(train_pred)
train_actual_labels = pred2label(y_train)

test_pred_labels = pred2label(test_pred)
test_actual_labels = pred2label(y_test)

print("F1-score: {:.1%}".format(f1_score(train_actual_labels, train_pred_labels)))
print("F1-score: {:.1%}".format(f1_score(test_actual_labels, test_pred_labels)))

print(classification_report(test_actual_labels, test_pred_labels))

model.evaluate(X_test, np.array(y_test))

i = 6
p = model.predict(np.array([X_test[i]]))
p = np.argmax(p, axis=-1)
true = np.argmax(y_test[i], -1)
print("{},{},{}".format("Word", "True", "Pred"))
for w, t, pred in zip(X_test[i], true, p[0]):
    #     if w != 0:
    print("{},{},{}".format(words[w - 1], tags[t], tags[pred]))

# Custom Tokenizer
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s): return s.split()


test_sentence = ''
x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in tokenize(test_sentence)]],
                            padding="post", value=0, maxlen=max_len)
p = model.predict(np.array([x_test_sent[0]]))
p = np.argmax(p, axis=-1)
print("{:15}||{}".format("Word", "Prediction"))
print(30 * "=")
for w, pred in zip(tokenize(test_sentence), p[0]):
    print("{:15}: {:5}".format(w, tags[pred]))

# get sentences
# data_test
getter_test = SentenceGetter(data)
sent_test = getter_test.get_next()
sentences_test = getter_test.sentences

# pad the sequence - test sample
X_test = [[word2idx[w[0]] for w in s] for s in sentences_test]
X_test = pad_sequences(maxlen=max_len, sequences=X_test, padding="post", value=n_words - 1)

# pad the target - dev sample
y_test = [[tag2idx[w[1]] for w in s] for s in sentences_test]
y_test = pad_sequences(maxlen=max_len, sequences=y_test, padding="post", value=tag2idx["O"])

y_test = [to_categorical(i, num_classes=n_tags) for i in y_test]

test_pred = model.predict(X_test, verbose=1)

test_pred_labels = pred2label(test_pred)
test_actual_labels = pred2label(y_test)

print("F1-score: {:.1%}".format(f1_score(test_actual_labels, test_pred_labels)))
print(classification_report(test_actual_labels, test_pred_labels))


def test_sentence_sample(test_sentence):
    # test_sentence="Anh Mai đang bị hâm"
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in tokenize(test_sentence)]], padding="post",
                                value=0, maxlen=max_len)
    p = model.predict(np.array([x_test_sent[0]]))
    p = np.argmax(p, axis=-1)
    print("{:15}||{}".format("Word", "Prediction"))
    print(30 * "=")
    for w, pred in zip(tokenize(test_sentence), p[0]):
        print("{:15}: {:5}".format(w, tags[pred]))


test_sentence_sample("Anh Mai đang bị hâm")
