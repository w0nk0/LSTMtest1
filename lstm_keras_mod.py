from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random, sys

'''
    Example script to generate text from Nietzsche's writings.
    At least 20 epochs are required before the generated text
    starts sounding coherent.
    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.
    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''

stop_production_strings = ['! ','? ','. ','_END_','\n\n']
min_production_length = 50

from rddt import redditor_text

redditor = "pineconez"

#text = open("rddt-de-300.cache").read().lower()
#text = text[:400]

#REDDIT_MODE = "TEXT"
#text = redditor_text("pineconez",10)

text = "Holy diver. You've been down to long in the midnight sea. Oh what's becoming of me. " * 20
#text = open("lyrics.txt").read()
#text = open("rddt.py").read()
print('corpus length:', len(text))

chars = set(text)
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: 2 stacked LSTM

HIDDEN_NEURONS=128

print('Build model...')
model = Sequential()
model.add(LSTM(len(chars), HIDDEN_NEURONS, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(HIDDEN_NEURONS, HIDDEN_NEURONS, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(HIDDEN_NEURONS, len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
from time import time
output_file_name="generated-Neu{}maxl{}time{}.txt".format(HIDDEN_NEURONS,maxlen,time() % 1000)
print("Will write generated text to {}".format(output_file_name))

# helper function to sample an index from a probability array
def sample(a, diversity=0.75):
    if random.random() > diversity:
        return np.argmax(a)
    while 1:
        i = random.randint(0, len(a)-1)
        if a[i] > random.random():
            return i

def w0nk0sample(a,diversity=0.8):
    if diversity >= 1.0:
        raise ValueError("Can't have a diversity >= 1.0!")
    randomized = np.array(a)
    winner = len(a)-1
    while random.random() < diversity:
        randomized[np.argmax(randomized)] *= 0.1
    return np.argmax(randomized)

# train the model, output generated text after each iteration

def make_seed(text, maxlen, splitter=". ",from_reddit=False):
    if from_reddit:
        return rddt.redditor_text(redditor,10,True)
    if splitter in text:
        parts = text.split(splitter)
        part = random.randint(0,len(parts)-1)
        sentence = parts[part]+splitter
        sentence = sentence[-maxlen+1:]
    else:
        start_index = random.randint(0, len(text) - maxlen - 1)
        sentence = text[start_index : start_index + maxlen]
    return sentence

train_seconds=20.0
train_epochs=1
for iteration in range(1, 5000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    from time import time
    start_time=time()
    trained = 0
    while time()<start_time+train_seconds:
        model.fit(X, y, batch_size=128, nb_epoch=train_epochs)
        trained += train_epochs
    train_epochs = max(1,int(trained / (time()-start_time) * (train_seconds+2)))

    seed = make_seed(text, maxlen, '. ')

    for diversity in [ 0.4, 0.6, 0.701, 0.8]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = seed
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
        whole_sentence = sentence
        if diversity == 0.701:
            sample_func = sample
            print("\n------ Classic sampling")
        else:
            sample_func = w0nk0sample
            print("\n------ W0nk0 sampling")

        for iteration2 in range(500):
            pred_sentence = sentence[-maxlen+1:] ##### CARFEUL - TODO;TEST ####
            if diversity==0.4:
                print("\rpred_sentence  :", pred_sentence,end="")
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(pred_sentence): ##### CARFEUL - TODO;TEST ####
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample_func(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
            whole_sentence += next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
            # exit if we find a temrination string in the sentence!
            if max([x in whole_sentence[min_production_length:] for x in stop_production_strings]):
                break
        print()
        with open(output_file_name,"at") as out_file:
            out_file.write("Iteration {} - ".format(iteration)+whole_sentence+"\n")
