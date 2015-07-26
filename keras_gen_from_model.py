from __future__ import print_function

import os
os.environ['THEANO_FLAGS'] = "mode=FAST_COMPILE"

__author__ = 'http://stackoverflow.com/questions/25967922/pybrain-time-series-prediction-using-lstm-recurrent-nets'

from sys import stdout
from itertools import cycle
from random import random, choice, sample
from time import time

import numpy as np
from optparse import OptionParser
from time import time
import argparse 
from random import randint

import logging
warn = logging.warning

from rddt import redditor_text #, FlatReddit

from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, JZS1
from keras.datasets import imdb
from keras.models import model_from_json


from vectorizer import Vectorizer, VectorizerTwoChars

################ INIT GLOBAL VARIABLES ################



outfile = None

## obsolete

#MINIBATCH = 500
# CHARS = " abcdefghijklmnopqrstuvxyz.,!?\n" +"abcdefghijklmnopqrstuvxyz".swapcase()
#CHARS = " abcdefghijklmnopqrstuvxyz.,!?':/()-*1234567890\n" + "abcdefghijklmnopqrstuvxyz".swapcase()
#NOISE_FACTOR = 0.0000002
#DELTA0INIT = 0.24
#CLEAN_RESULT = True
# RETRAIN_FREQUENCY = int(16/EPOCHS_PER_CYCLE/CYCLES)


# # len-400, neurons=32 -> gets stuck on (good) always same sentecce
# # len 2000 neurons 16 - struggles to make words, then does OK with occasional glitch-fail, still stuck-ish
# # len 6000 neurons 16 - initially struggling, then partially OK partially fail
# # len 400 neurons 80 - immediately good fit, gets stuck on the sentences it knows
# # len 2000 neurons 80 - D0=0.24 quickly reproduces known stuff (by loop 2).


class TimeVectorizer(Vectorizer):
    def _vectify(self,vec):
        v=np.array(vec)
        if len(v.shape)==2:
            vec=vec[-1]
        if len(v.shape)>2:
            raise TypeError("from_vector_.. got a non- 1- or 2-dimensional vector :("+str(v))
        return list(vec)

    def from_vector(self,vec,unknown_token_value=None):
        vec = self._vectify(vec)
        return super(TimeVectorizer,self).from_vector(vec)

    def from_vector_rand(self,vec,randomization=0.5,unknown_token_value=None):
        vec = self._vectify(vec)
        return super(TimeVectorizer,self).from_vector_rand(vec,randomization)

class TimeVectorizerNoUnknown(TimeVectorizer):
    def from_vector_rand_no_dummy(self,vec,randomization=0.5,unknown_token_value=None):
        from random import random
        vec = self._vectify(vec)
        ## pick the NEXT BEST guess if the best guess is a word not in dictionary
        ##   words not in dict should have one-hot [ 1.0, ...... ], so the
        ##   wuinner may be anything but the first one-hot neuron
        #unknown_value = self.item(-1)
        # can just enumerate from >1, no?
        try:
            srt = [(v,x) for x,v in enumerate(vec[1:])] # leave out words not in dictionary?
            srt.sort()
            srt.reverse()
        except:
            vec=np.array(vec)
            print("Exception in from_vector_rand_no_dummy - vec shape={} vec={} srt={}".format(vec.shape,vec,srt))
            raise
        for winner, idx in srt:
            if random()<(1-randomization):
                break
        return self.detokenize(self.item(idx,unknown_token_value))


class TimeVectorizer2Lemma(VectorizerTwoChars):
    def _vectify(self,vec):
        v=np.array(vec)
        if len(v.shape)==2:
            vec=vec[-1]
        if len(v.shape)>2:
            raise TypeError("from_vector_.. got a non- 1- or 2-dimensional vector :("+str(v))
        return list(vec)

    def from_vector(self,vec,unknown_token_value=None):
        vec = self._vectify(vec)
        return super(TimeVectorizer2Lemma,self).from_vector(vec,unknown_token_value)

    def from_vector_rand(self,vec,randomization=0.5,unknown_token_value=None):
        vec = self._vectify(vec)
        return super(TimeVectorizer2Lemma,self).from_vector_rand(vec,randomization,unknown_token_value)

class OneCharacterVectorizer(Vectorizer):
    def detokenize(self, token):
        return "" + token

    def tokenize(self, stream):
        tokens = []
        for n in range(0, len(stream) - 1, 1):
            tokens.append(stream[n])
        return tokens

class RandomVectorizer(Vectorizer):
    def from_vector(self, vec,unknown_token_value=None):
        # print("old vec: {}".format(vec))
        vec = [v + v * random() * NOISE_FACTOR for v in vec]
        # print("new vec: {}".format(vec))
        winner = max(vec)
        win_index = vec.index(winner)
        return self.detokenize(self.item(win_index,unknown_token_value))

    def _c(self, item): return item


class WordVectorizer(RandomVectorizer):
    def tokenize(self, stream):
        return stream.split(" ")

    def detokenize(self, items):
        return " " + items

def banner(txt="",width=70, dash="-"):
    dashes = int((width - len(txt) -2) /2)
    print("\n",dash * dashes,txt, dash * dashes)

def plot_result(data, title):
    import matplotlib.pyplot as plt
    # plt.plot(range(len(0,equity_curve)), equity_curve)
    plt.plot(data)
    plt.ylabel('equity curve')
    if title:
        plt.title(title)
    plt.show()
    plt.close()


def make_dataset(data_matrix, v):
    in_batch, out_batch = [], []

    olditem = data_matrix[0]
    for item in data_matrix[1:-1]:
        # 1 line = 1 character_classification_vector
        input = [olditem]  # , input = [olditem, item]
        output = item  # =label =target lemma
        olditem = item  # to have a 1-delay previous item

        in_batch.append(input)
        out_batch.append(output)

    return np.array(in_batch), np.array(out_batch)


def make_dataset_withtime(data_matrix, v, length):
    in_batch, out_batch = [], []
    for num in range(len(data_matrix) - length - 1):  # )(data_matrix[length:-1]):
        # 1 line = 1 character_classification_vector
        input = []  # , input = [olditem, item
        output= []
        for x in range(length):
            input.append(data_matrix[num + x])
            output.append(data_matrix[num+x+1])

        in_batch.append(np.array(input))
        out_batch.append(output)

    return np.array(in_batch), np.array(out_batch)

def make_dataset_single_predict(data_matrix, v, length,step=1):
    in_batch, out_batch = [], []
    for num in range(0,len(data_matrix) - length - 1,2):  # )(data_matrix[length:-1]):
        # 1 line = 1 character_classification_vector
        input = []  # , input = [olditem, item
        output= []
        for x in range(0,length,step):
            input.append(data_matrix[num + x])
            output.append(data_matrix[num+x+1])

        in_batch.append(np.array(input))
        out_batch.append(output[-1])

    return np.array(in_batch), np.array(out_batch)

# ###### GENERATING SAMPLES ##############



def generate_text(net,vectorizer,randomness=-0.5, custom_primer=None):
    """RANDOMNESS <0: from_random_sample = sampling by distribution with vector weights
       RANDOMNESS >0: nerfing winners with probability X, then picking winner post-nerfing
    """
    stopper = ["_#", "!", ". ", "\n \n","\n\n", "?" ]
    random_factor=static_factor=randomness # +random()*randomness*randomness
    print("--------> Using static factor {} <--------".format(static_factor))

    from random import randint
    vec = vectorizer

    primer = "\\" * WINDOW_LEN + custom_primer #"#_B_#" + custom_primer + "#_E_#"
    primer = primer[-WINDOW_LEN:]
    print("Priming network with:",primer)
    primer_mat = vec.to_matrix(primer)
    P,_ = make_dataset_single_predict(primer_mat, vec, WINDOW_LEN, step=1)

    current = np.array([primer_mat])
    
    result=""
    try:
        result = "".join([vec.from_vector(x) for x in current[0]])+"/"
    except:
        result = "## "
    
    print_offset = 0
    per_line = 80
    end_generation = False
    for x in range(300):
        for item in stopper:
            if item in result[30:]:
                end_generation = True
        if end_generation:
            break
        
        if len(result) >= per_line + print_offset:
            print("")
            print_offset += per_line
        offset_result = result[print_offset:].replace("\n","\\n")
        print("\rOutput:",offset_result,end="")        
        stdout.flush()
        
        prediction = net.predict(current,verbose=0) #,batch_size=len(current[0])

        p0 = prediction # [-1]

        new_current =[]
        for c in current[0]:
            new_current.append(c)

        txt = vec.from_vector_multinomialsampled(p0[-1],-randomness) # now in git       
        rand_p = vec.vector(txt)
        ## and feed it back into the batch
        new_current.append(rand_p)
        try:
            result += txt or "<NONE>"
        except:
            print("Couldn't add '{}' to result".format(txt))

        current = np.array([new_current])
        current = np.array([new_current[-WINDOW_LEN:]])

    result = rebuild_caps(result)
    print(result.replace('\n',r'\\n'))
    banner("end predict")
    
    
    
def save_weights(model,fname):
    """saves models weights into File speicified by fname"""
    from pickle import dumps
    weight_str = dumps(model.get_weights())
    if not '.' in fname:
      fname += '.wpkls'
    try:
        with open(fname,"wb")  as f:
            f.write(weight_str)
        return True
    except:
        warn('Couldnt write weights to {}'.format(fname))

def load_weights(model,fname):
    """ sets weights from file, returns model"""
    from pickle import loads
    if not '.' in fname:
        fname += '.wpkls'
    try:
        with open(fname,"rb")  as f:
            wstr=f.read()
        w = loads(wstr)
        model.set_weights(w)
        return model
    except:
        warn('Couldnt write weights to {}'.format(fname))
        raise

def replace_caps(input, identifier="|", identifier_replace="}{"):
    input = input.replace(identifier, identifier_replace)
    uppers=[chr(n) for n in range(ord("A"),ord("Z"))]
    lowers=[identifier+chr(n) for n in range(ord("a"),ord("z"))]
    for u, repl in zip(uppers,lowers):
        input = input.replace(u,repl)
    return input
        

def rebuild_caps(input, identifier="|", identifier_replace="}{"):
    uppers=[chr(n) for n in range(ord("A"),ord("Z"))]
    lowers=[identifier+chr(n) for n in range(ord("a"),ord("z"))]
    for u, repl in zip(uppers,lowers):
        input = input.replace(repl,u)
    input = input.replace(identifier_replace, identifier)
    return input

        

################ ARGUMENTS PARSING ################

## text arguments
parser = argparse.ArgumentParser("Keras' demo LSTM generation")
arglist = "--yaml --weights --vector"
for arg in arglist.split(" "):
    parser.add_argument(arg)

## integer arguments
arglist = "--windowlen --replacecaps"
for arg in arglist.split(" "):
    parser.add_argument(arg,type=int)
args = parser.parse_args()

yamlfile = args.yaml
weightfile = args.weights
vectorfile = args.vector
WINDOW_LEN = args.windowlen or 50 # 50

RUN_ID = 0
WINDOW_STEP = 1
REPLACE_CAPS = args.replacecaps or True #

########### start

print("Initializing vectorizer")
v = TimeVectorizerNoUnknown("nothing",cutoff=100)
v.load(vectorfile)
print("Vectorizer dictionary:",v.dictionary)

print("Loading model")
from keras.models import model_from_yaml
with open(yamlfile) as yf:
    model = model_from_yaml(yf.read())
    
print("Compiling model, please wait")
model.compile('rmsprop','categorical_crossentropy')

#load weights
model = load_weights(model,weightfile)

while 1:
    seed = replace_caps(input("Seed (or 'quit')>"))
    if "quit" in seed:
        break
    generate_text(model,v,-0.2,custom_primer=seed)
    generate_text(model,v,-0.5,custom_primer=seed)
    generate_text(model,v,-0.8,custom_primer=seed)