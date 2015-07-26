from __future__ import print_function

# import os
# os.env['THEANO_FLAGS'] = "device=cpu"

__author__ = 'http://stackoverflow.com/questions/25967922/pybrain-time-series-prediction-using-lstm-recurrent-nets'

from sys import stdout
from itertools import cycle
from random import random, choice, sample
from time import time

import numpy as np
from optparse import OptionParser
import time
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

################ ARGUMENTS PARSING ################

## text arguments
parser = argparse.ArgumentParser("Keras' demo LSTM generation")
arglist = "--redditor --subreddit --textfile --loadweightsfile --vectorfile --fromyamlfile --optimizer"
for arg in arglist.split(" "):
    parser.add_argument(arg)

## integer arguments
arglist = "--subredditposts --redditorposts --windowlen --windowstep --hidden --trainlen --replacecaps"
for arg in arglist.split(" "):
    parser.add_argument(arg,type=int)
args = parser.parse_args()

################ INIT GLOBAL VARIABLES ################

RUN_ID = int(time.time() % 1000)
WINDOW_LEN = args.windowlen or 50 # 50
WINDOW_STEP = args.windowstep or 1
TRAIN_LEN = args.trainlen or 200000 # 15000
HIDDEN_NEURONS = args.hidden or 512 # 400
TEXT_FILE = args.textfile #"rddt-de-300.cache"# r"..\\rddt\\cache\\rddt-fatlogic-150.cache"
REDDITOR = args.redditor or "w0nk0" # None # "pineconez"
REDDIT_MODE = "TEXT" # TEXT or WORDS
NUM_POSTS = args.redditorposts or 100 #
REPLACE_CAPS = args.replacecaps or True #

load_weights_file = args.loadweightsfile
load_vectorizer_file = args.vectorfile


outfile = None
from time import time
output_file_name="run{}-generated-Neu{}-winlen{}.txt".format(RUN_ID,HIDDEN_NEURONS,WINDOW_LEN)
try:
    outfile = open(output_file_name, "at")
    print("Will write generated text to {}".format(output_file_name))
except:
    print("No file output")


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
    # # CURRENTLY MISSING CORRECT BATCHES SEPARATED BY x in ".!?"!!
    # if v.from_vector(data_matrix[line+1]) in ".!?":
    # print(".",end="")
    # ds.newSequence()
    # inputs, outputs = [], []
    in_batch, out_batch = [], []

    olditem = data_matrix[0]
    for item in data_matrix[1:-1]:
        # 1 line = 1 character_classification_vector
        input = [olditem]  # , input = [olditem, item]
        output = item  # =label =target lemma
        olditem = item  # to have a 1-delay previous item

        in_batch.append(input)
        out_batch.append(output)
        # end batch on !.?
        # if v.from_vector(item) in r".!?\n":
        # inputs += in_batch
        # inputs.append(in_batch)
        # outputs.append(out_batch)
        # in_batch = []
        # out_batch = []

    # put leftovers in a batch as well
    # if len(in_batch):
    # inputs.append(in_batch)
    # outputs.append(out_batch)
    #
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


def anneal_matrix(data_matrix, anneal_factor=0.5):
    activation = np.zeros(len(data_matrix[0]))
    new_matrix = []
    for vector in data_matrix:
        activation = activation + np.array(vector)
        new_matrix.append(list(activation))
        activation *= anneal_factor
        # print(activation)
    return new_matrix


def make_dataset_n_anneal(data_matrix, v, length):
    in_batch, out_batch = [], []

    for num in range(len(data_matrix) - length - 1):  # )(data_matrix[length:-1]):
        # 1 line = 1 character_classification_vector
        input = []  # , input = [olditem, item
        for x in range(length - 1):
            input.append(data_matrix[num + x])
        output = data_matrix[num + length]  # =label =target lemma

        in_batch.append(np.array(input))
        out_batch.append(output)

    return np.array(in_batch), np.array(out_batch)


def debug_vec_print(vec, x, label, print_matrix=True):
    # print("######### type(x) shape(x)",type(x),np.array(x).shape)
    x = vec._vectify(x)
    # print("######### RESHAPED _ type(x) shape(x)",type(x),np.array(x).shape)
    printable = vec.from_vector(list(x))
    if print_matrix:
        print("----------------------", label, "--------------------- :")
        y = np.zeros(len(x))
        for n, item in enumerate(x):
            if item >= 0.1:
                y[n] = item
        print(y.round(2))
    print("{}: <<{}>>".format(label, printable))
    return printable


def vector_randomized(vector, static_factor=0.5):
    a = [random() + static_factor for x in range(len(vector))]
    b = [1.3 * static_factor - 1.3 * random() for x in range(len(vector))]
    rands = np.array(a)
    rands2 = np.array(b)
    randomizered = (vector + rands2) * rands
    return randomizered

# ###### GENERATING SAMPLES ##############

def make_net(in_size, out_size, hidden_size=20):
    model = Sequential()
    model.add(JZS1(input_dim=in_size, output_dim=hidden_size, return_sequences=True))
    model.add(Dropout(0.3))

    model.add(JZS1(input_dim=hidden_size, output_dim=128))
    model.add(Dropout(0.3))

    #model.add(Dense(input_dim=hidden_size, output_dim=out_size, init="glorot_normal", activation="softmax"))
    #model.add(TimeDistributedDense(input_dim=int(hidden_size/2), output_dim=out_size))
    model.add(Dense(input_dim=128, output_dim=out_size))
    model.add(Activation('softmax'))

    # model.add(Dense(input_dim = 5, output_dim = 1, init = "uniform", activation = "tanh"))
    # model.compile(loss = "mean_squared_error", optimizer = "rmsprop",class_mode="binary")
    
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode="binary")  # or binary
    
    rmsfast = RMSprop(lr=0.005) # unused for now, default is 0.001 - starting with 0.08 usually works for a while!
    optim = SGD(lr=0.05, decay=0.008)
    print("Compiling net..with {} input, {} outputs, {} hidden please hold!".format(in_size, out_size, hidden_size))
    model.compile(loss='categorical_crossentropy', optimizer=args.optimizer or rmsfast , class_mode="categorical")  # or binary
    
    return model

def make_net_run52(in_size, out_size, hidden_size=20):
    model = Sequential()
    # model.add(LSTM(input_dim = in_size, output_dim = in_size, init="uniform", activation = "sigmoid", return_sequences=True))
    model.add(GRU(input_dim=in_size, output_dim=int(hidden_size),  return_sequences=True))
    #model.add(Dropout(0.1))
    #model.add(GRU(input_dim=hidden_size, output_dim=int(hidden_size),  return_sequences=False))

    model.add(Dropout(0.4))

    model.add(LSTM(input_dim=hidden_size, output_dim=128))
    model.add(Dropout(0.2))

    #model.add(Dense(input_dim=hidden_size, output_dim=out_size, init="glorot_normal", activation="softmax"))
    #model.add(TimeDistributedDense(input_dim=int(hidden_size/2), output_dim=out_size))
    model.add(Dense(input_dim=int(128), output_dim=out_size))
    model.add(Activation('softmax'))

    # model.add(Dense(input_dim = 5, output_dim = 1, init = "uniform", activation = "tanh"))
    # model.compile(loss = "mean_squared_error", optimizer = "rmsprop",class_mode="binary")
    
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode="binary")  # or binary
    
    rmsfast = RMSprop(lr=0.05) # unused for now
    print("Compiling net..with {} input, {} outputs, {} hidden please hold!".format(in_size, out_size, hidden_size))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode="categorical")  # or binary
    
    return model


def make_net_LSTM(in_size, out_size, hidden_size=20):
    model = Sequential()
    # model.add(LSTM(input_dim = in_size, output_dim = in_size, init="uniform", activation = "sigmoid", return_sequences=True))
    model.add(LSTM(input_dim=in_size, output_dim=int(hidden_size),  return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim=hidden_size, output_dim=int(hidden_size),  return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim=hidden_size, output_dim=int(hidden_size),  return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim=hidden_size, output_dim=int(hidden_size),  return_sequences=False))

    model.add(Dropout(0.2))

    #model.add(LSTM(input_dim=hidden_size, output_dim=hidden_size, init="glorot_normal"))
    #model.add(Dropout(0.3))

    #model.add(Dense(input_dim=hidden_size, output_dim=out_size, init="glorot_normal", activation="softmax"))
    #model.add(TimeDistributedDense(input_dim=int(hidden_size/2), output_dim=out_size))
    model.add(Dense(input_dim=int(hidden_size), output_dim=out_size))
    model.add(Activation('softmax'))

    # model.add(Dense(input_dim = 5, output_dim = 1, init = "uniform", activation = "tanh"))
    print("Compiling net..with {} input, {} outputs, {} hidden please hold!".format(in_size, out_size, hidden_size))
    # model.compile(loss = "mean_squared_error", optimizer = "rmsprop",class_mode="binary")
    
    rmsfast = RMSprop(lr=0.004)
    model.compile(loss='mse', optimizer=rmsfast, class_mode="categorical")  # or binary
    return model

def get_input_text(filename, redditor, train_len, posts=200):
    input_text="#FAILED_READING_FILE<{}>#".format(filename)
    if filename:
        with open(filename) as f:
            input_text = f.read()
    else:
        input_text = redditor_text(redditor,posts,False,REDDIT_MODE)

    start = int(random() * (len(input_text) - train_len - 100))
    if (train_len+100) >= len(input_text): start = 0
    try:
        start = input_text.index(".", start) + 1
        input_text = input_text.strip()
    except:
        start = 0
    pruned = input_text[start:start + train_len]
    try:
        pruned = ".".join(pruned.split(".")[:-1]) + "."
        pruned = pruned.strip()
    except:
        pass
    # print("Pruned:",pruned)
    return input_text, pruned

def predict_100(net,vectorizer,X,y,randomness=0.1, custom_primer=None):
    """RANDOMNESS <0: from_random_sample = sampling by distribution with vector weights
       RANDOMNESS >0: nerfing winners with probability X, then picking winner post-nerfing
    """
    random_factor=static_factor=randomness # +random()*randomness*randomness
    print("--------> Using static factor {} <--------".format(static_factor))
    def randomized(vector, static_factor=0.5):
        a=[random()+static_factor for x in range(len(vector))]
        b=[1.3*static_factor-1.3*random() for x in range(len(vector))]
        rands = np.array(a)
        rands2 = np.array(b)
        randomizered = (vector+rands2) * rands
        return randomizered

    from random import randint
    vec = vectorizer

    #current = [[vec.vector(choice(vec.dictionary)),vec.vector(choice(vec.dictionary))]]
    #primer="""It's an excuse.  If your weight is the fault of a disease, then you don't feel at fault for it. It's just a way people try to let go of the guilt they feel for their weight."""
    #mat=vectorizer.to_matrix(primer)
    #X,y = make_dataset_n(mat,vec,WINDOW_LEN)

    if False:
        print("Shapes: X", X.shape, "y", y.shape)
        print("X - {} entries".format(len(X)))
        print("x[0]", X[0].shape)

        print("Shape primer X[0]", X[0].shape)
        print("len(X)",len(X))

    if custom_primer:
        primer = " " * 100 + custom_primer #"#_B_#" + custom_primer + "#_E_#"
        primer_mat = vec.to_matrix(primer[-WINDOW_LEN:])
        P,_ = make_dataset_single_predict(primer_mat, vec, WINDOW_LEN, step=1)
        current = np.array([primer_mat])
    else:
        idx = randint(0,len(X)-2)
        current = np.array([X[idx]])

    #print ("Initial input (current):", current)
    #for i in current[0]:
    #  debug_vec_print(vec,i,"c[0]",False)
    #print ("Shape:", current.shape)

    result=""
    try:
        result = "".join([vec.from_vector(x) for x in current[0]])+"/"
    except:
        result = "## "
    print_offset = 0
    per_line = 80
    for x in range(120):
        if len(result) >= per_line + print_offset:
            print("")
            print_offset += per_line
        offset_result = result[print_offset:].replace("\n","\\n")
        print("\rOutput:",offset_result,end="")
        
        stopper = ["#_E_#", "!", ".", "\n \n","\n\n", "?" ]
        for item in stopper:
            if item in result[30:]:
                continue
        
        stdout.flush()
        #print("Shape:",current.shape)
        prediction = net.predict(current,verbose=0) #,batch_size=len(current[0])
        # input :   X[samples,[timesteps, [onehots,]]]
        # p  TDD -> X[samples,[timesteps, [onehots,]]]
        # p D    -> X[samples, [onehots,]]]

        p0 = prediction # [-1]

        if x < 1:
            print("\n","-" * 40)
            p=prediction
        new_current = []

        if (x % 100) == 99 and False:
            print("This current:")
            for c in current[0]:
                debug_vec_print(vec,c,"cur cur[0]",False)

        for c in current[0]:
            new_current.append(c)

        ## get the best/randomized NON-UNKNOWN winner
        if randomness<0.0:
          #txt = vec.from_vector_sampled(p0[-1],-randomness) # 
          txt = vec.from_vector_multinomialsampled(p0[-1],-randomness) # now in git
        else:
          txt = vec.from_vector_rand(p0[-1],random_factor,unknown_token_value="#?#") # WAS NO_DUMMY
        
        rand_p = vec.vector(txt)
        ## and feed it back into the batch
        new_current.append(rand_p)
        try:
            result += txt or "<NONE>"
        except:
            print("Couldn't add '{}' to result".format(txt))

        if x <= 1 and False:
            print("+" * 40)
            print("input      :",end="")
            vec.print_matrix(current[0])
            print("predicted :",end="")
            vec.print_matrix(p0)
            if False:
                print("newtxt vec :",end="")
                print([(it,lbl) for it,lbl in zip(p0[-1].round(2),vec.dictionary)])
            print("add char   : <{}>".format(txt))
            print("-> new in  :",end="")
            vec.print_matrix(new_current[-WINDOW_LEN:])
            #print("-" * 30)

        current = np.array([new_current])
        current = np.array([new_current[-WINDOW_LEN:]])


        if x <= 1 and False:
            try:
                #print("prediction shape: ", prediction.shape)
                #print("new_current shape: ", new_current.shape)
                debug_vec_print(vec,current[0][-1],"current[0][-1]")
                debug_vec_print(vec,p0,"pred0")
                #debug_vec_print(vec,randomized(p0,0.25),"randomized p0 0.25")
                #debug_vec_print(vec,randomized(p0,0.8),"randomized p0 0.8")
                debug_vec_print(vec,y[idx],"y[0]")
                #print("New current:",current)
                #print("Shape:", current.shape)
            except Exception:
                print("Exception in debug prints")

        if x==99 and False:
            print("x=99")
            print("prediction shape: ", prediction.shape)
            print("new_current shape: ", current.shape)

            print("PREDICTION")
            print(prediction)
            print("PREDICT_CLASSES")
            print(class_prediction)

    #print("Prediction total:","".join([vec.from_vector(list(p)) for p in prediction]))
    banner(" ++## RESULT ##++")
    result = rebuild_caps(result)
    print(result.replace('\n',r'\\n'))
    if outfile:
        outfile.write("\n"+result.replace("\n","<br>")+"\n")
        outfile.flush()
    print("(rand factor was {})\n".format(static_factor))
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
        
def run():
    # ######################### MAKE TRAINING DATA
    # ######################### MAKE TRAINING DATA
    # ######################### MAKE TRAINING DATA
    # input_text = "Ein Test-Text, hurra!"
    # pruned = " ".join(input_text.split(" ")[:4])
    banner("--")
    banner("Run() starting, getting data..")
    input_text, pruned = get_input_text(TEXT_FILE, REDDITOR, TRAIN_LEN, posts=NUM_POSTS)
    if REPLACE_CAPS:
        input_text = replace_caps(input_text)
        pruned = replace_caps(pruned)
    #codec="ascii"
    codec="cp1252"
    input_text = input_text.encode(codec,errors='xmlcharrefreplace').decode(codec)
    pruned = pruned.encode(codec,errors='xmlcharrefreplace').decode(codec)
    with open("run{}-input_text-cp1252.txt".format(RUN_ID),"wb") as f:
        f.write(input_text.encode("cp1252",errors="replace"))
 
    #input_text = pruned = "abbcccdddd eeeeeffffff abc def? " * 8

    print("---------------------\ninput -400:\n", input_text[:400])
    print("---------------------\npruned -400:\n", pruned[:400])
    
    print("Total text length: {}, training set {}".format(len(input_text),len(pruned)))
    # v=RandomVectorizer(". "+input_text)
    # v=OneCharacterVectorizer(". "+input_text)
    #v = TimeVectorizer2Lemma(input_text) <- for when using REDDIT_MODE="TEXT", not words

    LIMIT = 1000
    if REDDIT_MODE == "TEXT" : LIMIT = 100
    print("LIMITING DICTIONARY TO ", LIMIT)
    banner("Vectorizing")
    v = TimeVectorizerNoUnknown(input_text,cutoff=LIMIT)
    print("Len vectorizer:", len(v.dictionary))
    
    if load_vectorizer_file:
      v.load(load_vectorizer_file)
    
    print("Saving vectorizer")
    v.save("run{}-vector.pkl".format(RUN_ID))
    input_mat = v.to_matrix(pruned)

    # V throws ascii/unicode error
    #print("\n",type(v)," Dictionary: {}".format(v.dictionary.encode("ascii",errors="ignore")))
    #my=v.to_matrix('my')
    #print("v.to_matrix('my')", my)
    #print("my[0]",v.from_vector_rand_no_dummy(list(my[0]),0.1,unknown_token_value="#?#"))
    #print("my",v.from_vector_rand_no_dummy(list(my),0.1,unknown_token_value="#?#"))
    #for _ in range(500):
    #    x=v.vector(input_text[randint(0,len(input_text))])
    #    print(v.from_vector_rand(x,0.5,unknown_token_value="#?#"),end="")
    print(v.dictionary)
    print("")
    #print("?? my == ",v.from_matrix(my))
    #from time import sleep
    #sleep(4)
    # print("Dictionary:",["".join(str(x)) for x in v.dictionary])
    #lemma = choice(v.dictionary)
    # print("dictionary choice:",lemma)
    # print("vector", v.vector(lemma))
    # print("index", v.index(lemma))

    # # check if mapping worksS
    # for num,i in enumerate(input_mat[0:10]):
    # debug_vec_print(v,i,"input[{}]".format(num))
    # print("input_mat[:2] : ",np.array(input_mat[:2]))

    #anneal_mat = anneal_matrix(input_mat)

    # # check if anneal-mapping works
    # for num,i in enumerate(anneal_mat[0:10]):
    # debug_vec_print(v,i,"input[{}]".format(num))
    # # anneal_mat  # # !!!!!!!!!!!!!!!!!!!!!!


    # ##### ###### ###### MAKE NETWORK ###### ###### ######
    # ##### ###### ###### MAKE NETWORK ###### ###### ######
    # ##### ###### ###### MAKE NETWORK ###### ###### ######
    banner("Compiling net")
    categories = v.len()

    if args.fromyamlfile:
        with open(args.fromyamlfile,"rt") as jsonfile:
            json=jsonfile.read()
            net = model_from_yaml(json)
    else:
        net = make_net(categories, categories, hidden_size=HIDDEN_NEURONS)
    #from keras.utils.dot_utils import Grapher
    #Grapher().plot(net,'run{}-model.png'.format(RUN_ID))
    # ^ needs pydot, pydot no workie py34?

    with open("run{}-model.yaml".format(RUN_ID),"wt") as jsonfile:
        jsonfile.write(net.to_yaml())

    banner("Net compiled!")

    if load_weights_file:
        print("/// Loading weights from {} as per argument!".format(load_weights_file))
        net = load_weights(net,load_weights_file)

    banner("Make dataset..")
    # X,y = make_dataset_n(input_mat,v,WINDOW_LEN)
    X, y = make_dataset_single_predict(input_mat, v, WINDOW_LEN,step=WINDOW_STEP)
    del input_mat
    #print("----------X-----------\n", X)
    if False:
        print("Shapes: X", X.shape, "y", y.shape)
        print("X - {} entries".format(len(X)))
        print("Shape X[0]", X[0].shape)

    if True:
        debug_vec_print(v, X[0][0], "X[0][0]")
        debug_vec_print(v, X[0][1], "X[0][1]")
        debug_vec_print(v, X[0][2], "X[0][2]")
        debug_vec_print(v, X[0][-1], "X[0][-1]")
        debug_vec_print(v, y[0], "y[0]")

        for item in range(4):
            print("\nLETTERS: X[",item,"][..]")
            for letter in X[item]:
                print(v.from_vector_sampled(letter),end="")
            print("  --->  y[]: <", end=">")
            print(v.from_vector_sampled(y[item]))
        stdout.flush()

    #print("X[0]")
    #v.print_matrix(X[0])
    #print("y[0]")
    #v.print_matrix(y[0])
    from time import sleep
    #sleep(2)

    #predict_100(net,v,X,y,custom_primer="This is awesome!")
    #save_weights(net,'run{}-weights'.format(RUN_ID))
    #net.fit(X, y, nb_epoch=1, batch_size=min(512,len(y)), show_accuracy=True, validation_split=0.1, verbose=1)
    #save_weights(net,'run{}-weights'.format(RUN_ID))

    zipped = list(zip(X, y))
    train_epochs=1.0
    trained_amount=1.0
    for iteration in range(10000):
        i=iteration
        # Train in mini-batches in stead of fll set? Did I do this because of 32 bit memory limits?
        print("Saving network weights")
        save_weights(net,'run{}-weights'.format(RUN_ID))
        if args.redditor:
            try:
                primer = redditor_text('w0nk0',10,justonerandom=True)
            except:
                primer = "Getting reddit post failed :("
            primer.encode('cp1252',errors='replace').decode('cp1252',errors='replace')
            primer = primer[-WINDOW_LEN-6:] + ' #_E_#'
        else:
            primer_idx = randint(0,(len(input_text) - WINDOW_LEN))
            primer = input_text[primer_idx : primer_idx+WINDOW_LEN]
        banner("Generating")
        predict_100(net, v, X, y, randomness=[0.2,0.3,0.45][i%3],custom_primer=primer)
        predict_100(net, v, X, y, randomness=[-0.2,-0.3,-0.5,-0.7,-0.9][i%5],custom_primer=primer)
        #fit for x seconds
        initial_time = time()
        SECONDS = 60
        train_epochs = max(1,int(0.5*max(1,0.5*trained_amount + 0.5*train_epochs)))
        trained_amount=0
        banner(" ITERATION {} ".format(iteration))
        banner("Fitting 2x{} epochs at least {} seconds..".format(train_epochs, SECONDS))
        while time() < initial_time + SECONDS:
            mX, my = sampleXy(X,y,int(min(len(X),2560*1.1)))
            trained_amount+=train_epochs
            fit_result = net.fit(mX, my, nb_epoch=train_epochs, batch_size=256, show_accuracy=True, validation_split=0.1, verbose=1) #batch_size=min(128,len(X[0])),
            #try:
            #    print("Result:",[x for x in fit_result])
            #except:
            #    print("Couldn't print fit() result")
        
def sampleXy(X,y,samples=256):
    samp = sample(range(len(X)), min(len(X), samples))
    mX = []
    my = []
    for idx in samp:
        mX.append(X[idx])
        my.append(y[idx])
    mX = np.array(mX)
    my = np.array(my)
    return mX,my

        
if __name__ == "__main__":
    run()

    