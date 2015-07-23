from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random, sys
import html # for .escape
from rddt import redditor_text
from markovtools import FlatSubreddit
import pickle
import time

import argparse

import logging
warn = logging.warning

'''
    Example script to generate text from Nietzsche's writings.
    At least 20 epochs are required before the generated text
    starts sounding coherent.
    It is recommended to run this script on GPU, as recurrent
    networks are quite computationally intensive.
    If you try this script on new data, make sure your corpus
    has at least ~100k characters. ~1M is better.
'''
parser = argparse.ArgumentParser("Keras' demo LSTM generation")
arglist = "--redditor --redditorposts --subreddit"
for arg in arglist.split(" "):
    parser.add_argument(arg)
arglist = "--subredditposts --sentencelen --sentencestep --hidden"
for arg in arglist.split(" "):
    parser.add_argument(arg,type=int)

args = parser.parse_args()

stop_production_strings = ['! ','? ','. ','#_E','\n\n']
min_production_length = 40

REDDIT_MODE = "TEXT"

subreddit = args.subreddit
subreddit_posts = args.subredditposts or 100
redditor = args.redditor
redditor_posts = args.redditorposts or 100
HIDDEN_NEURONS = args.hidden or 300
maxlen = args.sentencelen or 16
step = args.sentencestep or 3

RUN_ID = int(time.time() % 1000)

def save_weights(model,fname):
    """saves models weights into File speicified by fname"""
    from pickle import dumps
    weight_str = dumps(model.get_weights())
    try:
        with open(weight_file+'.wpkls',"wb")  as f:
            f.write(weight_str)
        return True
    except:
        warn('Couldnt write weights to {}'.format(weight_file))

def load_weights(model,fname):
    """ sets weights from file, returns model"""
    from pickle import loads
    try:
        with open(weight_file+'.wpkls',"rb")  as f:
            wstr=f.read()
        w = loads(wstr)
        model.set_weights(w)
        return model
    except:
        warn('Couldnt write weights to ',weight_file)


def recode(u):
    import sys
    try:
        cp = sys.stdout.encoding
        u = u.replace('ä','ae')
        u = u.replace('ö','oe')
        u = u.replace('ü','ue')
        u = u.replace('ß','ss')
        u = u.replace('Ä','Ae')
        u = u.replace('Ü','Ue')
        s = u.encode(cp,errors='xmlcharrefreplace').decode(cp)
        return s
    except:
        return str(str(u).encode(errors='ignore'))

print(recode("häßlich äöüß ♣◙"))

#text = str(str(FlatSubreddit(subreddit,subreddit_posts,True).text()).encode())
if subreddit:
    print('Reading {} posts from /r/{}'.format(subreddit_posts,subreddit))
    text = recode(FlatSubreddit(subreddit,subreddit_posts,True).text())
elif redditor:
    print('Reading {} posts from /r/{}'.format(redditor_posts,redditor))
    text = redditor_text(redditor,redditor_posts)
else:
    print('Using dummy text for training.')
    text = "#_B_# Holy diver. You've been down to long in the midnight sea. #_B_# Oh what's becoming of me. #_E_# " * 10

#text = open("rddt-de-300.cache").read().lower()
#text = text[:400]
#text = open("lyrics.txt").read()
#text = open("lstm_keras_mod_fix.py").read()

print('corpus length:', len(text))
print('corpus [-70:]',text[-70:])

chars = set(text)
print('total chars in vectorizer:', len(chars))

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

with open('run{}-chars-in{}-hid{}.pkl'.format(RUN_ID,len(chars),HIDDEN_NEURONS),'wb') as f:
    pickle.dump(chars,f)
    print('Pickled chars dictionary into {}.'.format(f.name))

with open('run{}-text-in{}-hid{}.pkl'.format(RUN_ID,len(chars),HIDDEN_NEURONS),'wb') as f:
    pickle.dump(text,f)
    print('Pickled text into {}.'.format(f.name))



# cut the text in semi-redundant sequences of maxlen characters
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


print('Build model... ',len(chars),HIDDEN_NEURONS,len(chars))
model = Sequential()
model.add(LSTM(len(chars), HIDDEN_NEURONS, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(HIDDEN_NEURONS, HIDDEN_NEURONS, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(HIDDEN_NEURONS, len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

weight_file = 'run{}-weights-in{}-hid{}'.format(RUN_ID,len(chars),HIDDEN_NEURONS)
save_weights(model,weight_file)

##### FAKE##########   ####
##### FAK            ##    ##
##### FAK            ##    ##
##### FAK            ########
##### FAKE########   ########  K E :)
##### FAK            ##    ##
##### FAK            ##    ##
##### FAK            ##    ##
#from fakemodel import Fakemodel
#model=Fakemodel()
##### FAKE ##########
##### FAKE ##########
##### FAKE ##########

output_file_name="run{}-generated-Neu{}maxl{}.txt".format(RUN_ID,HIDDEN_NEURONS,maxlen)
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
        return rddt.redditor_text(redditor,20,True)
    if splitter in text:
        parts = text.split(splitter)
        part = random.randint(0,len(parts)-1)
        sentence = parts[part] + splitter
    else:
        start_index = random.randint(0, len(text) - maxlen - 1)
        sentence = text[start_index : start_index + maxlen]

    sentence = " " * maxlen + sentence
    sentence = sentence[-maxlen:]
    try:
        assert(len(sentence)==maxlen)
    except:
        print("Sentence {} is {}, not maxlen={} characters long!".format(sentence,len(sentence),maxlen))

    return sentence

train_seconds=90.0
train_epochs=1
for iteration in range(1, 5000):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    from time import time
    start_time=time()
    trained = 0
    while time()<start_time+train_seconds:
        model.fit(X, y, batch_size=min(256,len(X)), nb_epoch=train_epochs)
        trained += train_epochs
    train_epochs = max(1,int(trained / (time()-start_time) * (train_seconds+2)))

    if weight_file:
        save_weights(model,weight_file)

    seed = make_seed(text, maxlen, '. ', redditor != None)
    for diversity in [ 0.05, 0.2, 0.401, 0.5]:
        print()
        print('----- diversity:', diversity)
        if diversity == 0.401:
            sample_func = sample
            print("\n------ Classic sampling")
        else:
            sample_func = w0nk0sample
            print("\n------ W0nk0 sampling")

        generated = ''
        sentence = seed
        whole_sentence = sentence
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for iteration2 in range(500):
            #pred_sentence = sentence[-maxlen+1:] ##### CARFEUL - TODO;TEST ####
            if diversity==0.4:
                #print("\rpred_sentence  :", pred_sentence,end="")
                print("\rsentence       :", sentence,end="")

            if len(sentence)!=maxlen:
                print("!!!! sentence is {} long, maxlen is {} !!!".format(len(sentence),maxlen))
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence): ##### CARFEUL - TODO;TEST #### #pred_sentence
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample_func(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char
            whole_sentence += next_char

            writable = whole_sentence.encode(sys.stdout.encoding,errors='xmlcharrefreplace')
            sys.stdout.write(next_char)
            sys.stdout.flush()
            # exit if we find a temrination string in the sentence!
            for x in stop_production_strings:
                if x in writable.decode('utf-8',errors='ignore'):
                    break
        print()
        with open(output_file_name,"ab") as out_file:
            output = "Iteration {} - ".format(iteration) + writable.decode('ascii',errors='ignore') + "\n"
            out_file.write(output.encode('utf-8',errors='replace'))
