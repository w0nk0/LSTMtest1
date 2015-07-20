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
from rddt import redditor_text

from keras.preprocessing import sequence
from keras.optimizers import RMSprop, SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, JZS1
from keras.datasets import imdb

from vectorizer import Vectorizer, VectorizerTwoChars

outfile = None
try:
    outfile = open("generated.txt", "at")
    print("File for output opened")
except:
    print("No file output")

WINDOW_LEN = 30 # 50
TRAIN_LEN = 350 # 15000
HIDDEN_NEURONS = 400 #400
TEXT_FILE = None #"rddt-de-300.cache"# r"..\\rddt\\cache\\rddt-fatlogic-150.cache"
REDDITOR = "tsukino_usako" # None # "pineconez"
REDDIT_MODE = "TEXT" # TEXT or WORDS

MINIBATCH = 500

## obsolete
# CHARS = " abcdefghijklmnopqrstuvxyz.,!?\n" +"abcdefghijklmnopqrstuvxyz".swapcase()
CHARS = " abcdefghijklmnopqrstuvxyz.,!?':/()-*1234567890\n" + "abcdefghijklmnopqrstuvxyz".swapcase()
NOISE_FACTOR = 0.0000002
DELTA0INIT = 0.24
CLEAN_RESULT = True
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


def make_dataset_n(data_matrix, v, length):
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
def run_test(net, vectorizer):
    txt = ""
    result = vectorizer.to_matrix(". ")[0]
    num = 0
    sample = ""

    # # initially, generate a sentence so we don't start in the middle of one
    txt = ""
    # net.reset()
    iter = 0
    while iter < 1000 and not "." in txt and not "!" in txt and not "?" in txt:
        # ####### TODO ###########
        result = net.activate(result)

        txt = vectorizer.from_vector(result)
        print(iter, "\t", txt, "\r", end="")
        iter += 1
    print("")

    # # now, loop over activations until we have enough
    while 1:
        # ####### TODO ###########
        # print(num,end="")
        stdout.flush()
        try:
            # print(output_2_char(result[:len(CHARS)-1]))
            pass
        except:
            print("Error! {}".format(result))
        num += 1

        new_result = net.activate(result)

        arr = [float(i) for i in new_result]
        txt = vectorizer.from_vector(arr)
        # print(txt,end="")
        sample += txt
        stdout.flush()

        # result= [ x + random()*NOISE_FACTOR for x in new_result ]
        result = [x + x * random() * 0.1 for x in new_result]

        if CLEAN_RESULT:
            result = vectorizer.vector(txt)
            cleaned = []
            for item in new_result:
                if item > 0.5:
                    cleaned.append(1.0)
                else:
                    cleaned.append(0.0)
            result = cleaned

        # print("Len result: {}".format(len(result)))
        if num > 300:
            break
        if num > 120:
            if " " in txt:
                break
        if len(sample) > 20:
            if "." in txt:
                break
    # print("--->",sample)
    return sample + '.'


def make_net(in_size, out_size, hidden_size=20):
    model = Sequential()
    # model.add(LSTM(input_dim = in_size, output_dim = in_size, init="uniform", activation = "sigmoid", return_sequences=True))
    model.add(LSTM(input_dim=in_size, output_dim=int(hidden_size),  return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(input_dim=int(hidden_size), output_dim=hidden_size,  return_sequences=True))
    model.add(Dropout(0.2))

    #model.add(LSTM(input_dim=hidden_size, output_dim=hidden_size, init="glorot_normal"))
    #model.add(Dropout(0.3))

    #model.add(Dense(input_dim=hidden_size, output_dim=out_size, init="glorot_normal", activation="softmax"))
    model.add(TimeDistributedDense(input_dim=hidden_size, output_dim=out_size))
    model.add(Activation('softmax'))

    # model.add(Dense(input_dim = 5, output_dim = 1, init = "uniform", activation = "tanh"))
    print("Compiling net..with {} input, {} outputs, {} hidden please hold!".format(in_size, out_size, hidden_size))
    # model.compile(loss = "mean_squared_error", optimizer = "rmsprop",class_mode="binary")
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', class_mode="binary")  # or binary
    return model


def get_input_text(filename, redditor, train_len):
    input_text="#FAILED_READING_FILE<{}>#".format(filename)
    if filename:
        with open(filename) as f:
            input_text = f.read()
    else:
        input_text = redditor_text(redditor,100)

    start = int(random() * (len(input_text) - train_len - 100))
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


def predict_100(net,vectorizer,X,y):
    random_factor=static_factor=0.2+random()*0.45
    print("--------> Using static factor {} <--------".format(static_factor))
    def randomized(vector, static_factor=0.5):
        a=[random()+static_factor for x in range(len(vector))]
        b=[1.3*static_factor-1.3*random() for x in range(len(vector))]
        rands = np.array(a)
        rands2 = np.array(b)
        randomizered = (vector+rands2) * rands
        return randomizered

    print("Shapes: X", X.shape, "y", y.shape)
    print("X - {} entries".format(len(X)))
    print("x[0]", X[0].shape)

    from random import randint
    vec = vectorizer
    #current = [[vec.vector(choice(vec.dictionary)),vec.vector(choice(vec.dictionary))]]

    #primer="""It's an excuse.  If your weight is the fault of a disease, then you don't feel at fault for it. It's just a way people try to let go of the guilt they feel for their weight."""
    #mat=vectorizer.to_matrix(primer)
    #X,y = make_dataset_n(mat,vec,WINDOW_LEN)

    print("Shape primer X[0]", X[0].shape)
    print("len(X)",len(X))
    idx = randint(0,len(X)-2)
    current = np.array([X[idx]])

    #print ("Initial input (current):", current)
    #for i in current[0]:
    #  debug_vec_print(vec,i,"c[0]",False)
    #print ("Shape:", current.shape)

    result=""
    try:
        result = "".join([vec.from_vector(x) for x in current[0]])+" -> "
    except:
        result = "## "
    for x in range(150):
        print("\r",result.replace("\n"," ")[-60:],end="")
        stdout.flush()
        #print("Shape:",current.shape)
        prediction = net.predict(current,batch_size=len(current[0]),verbose=0)
        p0 = prediction[-1]
        if x < 1:
            print("\n","-" * 40)
            p=prediction
            # print("prediction[0]")
            # vec.print_matrix(prediction[0])
            # print("prediction[-1]")
            # vec.print_matrix(prediction[-1])
            # print("p[0][0]:")
            # print(vec.from_vector(p[0][0]))
            # print("p[0][-1]:")
            # print(vec.from_vector(p[0][-1]))
            # print("p[-1][0]:")
            # print(vec.from_vector(p[-1][0]))
            # print("p[-1][-1]:")
            # print(vec.from_vector(p[-1][-1]))
        class_prediction = net.predict_classes(current,batch_size=len(current[0]),verbose=0)
        #print("** C 0 -1: '", [vec.from_vector(list(c)) for c in current[0]],"'  **")
        new_current = []

        if (x % 100) == 99 and False:
            print("This current:")
            for c in current[0]:
                debug_vec_print(vec,c,"cur cur[0]",False)

        for c in current[0]:
            new_current.append(c)

        ## get the best/randomized NON-UNKNOWN winner
        txt = vec.from_vector_rand(p0[-1],random_factor,unknown_token_value="#?#") # WAS NO_DUMMY
        rand_p = vec.vector(txt)
        ## and feed it back into the batch
        new_current.append(rand_p)
        try:
            result += txt
        except:
            print("Couldn't add '{}' to result".format(txt))

        if x <= 1:
            print("+" * 40)
            print("input      :",end="")
            vec.print_matrix(current[0])
            print("predicted :",end="")
            vec.print_matrix(prediction[0])
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
    print("\n\nRESULT:\n", result)
    if outfile:
        outfile.write("\n\n"+result+"\n")
    print("(rand factor was {})".format(static_factor))


def run():
    # ######################### MAKE TRAINING DATA
    # ######################### MAKE TRAINING DATA
    # ######################### MAKE TRAINING DATA


    # input_text = "Ein Test-Text, hurra!"
    # pruned = " ".join(input_text.split(" ")[:4])

    input_text, pruned = get_input_text(TEXT_FILE, REDDITOR, TRAIN_LEN)

    print("---------------------\ninput -400:\n", input_text[:400])
    print("---------------------\npruned -400:\n", pruned[:400])
    # v=RandomVectorizer(". "+input_text)
    # v=OneCharacterVectorizer(". "+input_text)

    #v = TimeVectorizer2Lemma(input_text) <- for when using REDDIT_MODE="TEXT", not words

    LIMIT = 1000
    if REDDIT_MODE == "TEXT" : LIMIT = 100
    print("LIMITING DICTIONARY TO ", LIMIT)

    v = TimeVectorizerNoUnknown(input_text,cutoff=LIMIT)

    # V throws ascii/unicode error
    #print("\n",type(v)," Dictionary: {}".format(v.dictionary.encode("ascii",errors="ignore")))
    print("Len vectorizer:", len(v.dictionary))
    #my=v.to_matrix('my')
    #print("v.to_matrix('my')", my)
    #print("my[0]",v.from_vector_rand_no_dummy(list(my[0]),0.1,unknown_token_value="#?#"))
    #print("my",v.from_vector_rand_no_dummy(list(my),0.1,unknown_token_value="#?#"))
    from random import randint
    #for _ in range(500):
    #    x=v.vector(input_text[randint(0,len(input_text))])
    #    print(v.from_vector_rand(x,0.5,unknown_token_value="#?#"),end="")
    print(v.dictionary)
    print("")
    #print("?? my == ",v.from_matrix(my))
    #from time import sleep
    #sleep(4)
    # print("Dictionary:",["".join(str(x)) for x in v.dictionary])

    lemma = choice(v.dictionary)
    # print("dictionary choice:",lemma)
    # print("vector", v.vector(lemma))
    # print("index", v.index(lemma))

    input_mat = v.to_matrix(pruned)

    # # check if mapping worksS
    # for num,i in enumerate(input_mat[0:10]):
    # debug_vec_print(v,i,"input[{}]".format(num))
    # print("input_mat[:2] : ",np.array(input_mat[:2]))

    anneal_mat = anneal_matrix(input_mat)

    # # check if anneal-mapping works
    # for num,i in enumerate(anneal_mat[0:10]):
    # debug_vec_print(v,i,"input[{}]".format(num))
    # # anneal_mat  # # !!!!!!!!!!!!!!!!!!!!!!

    # X,y = make_dataset_n(input_mat,v,WINDOW_LEN)
    X, y = make_dataset_n(input_mat, v, WINDOW_LEN)
    #print("----------X-----------\n", X)
    print("Shapes: X", X.shape, "y", y.shape)
    print("X - {} entries".format(len(X)))
    print("Shape X[0]", X[0].shape)

    if False:
        debug_vec_print(v, X[0][0], "X[0][0]")
        debug_vec_print(v, X[0][1], "X[0][1]")
        debug_vec_print(v, X[0][2], "X[0][2]")
        debug_vec_print(v, X[0][-1], "X[0][-1]")
        debug_vec_print(v, y[0], "y[0]")
        stdout.flush()


    #print("X[0]")
    #v.print_matrix(X[0])
    #print("y[0]")
    #v.print_matrix(y[0])
    from time import sleep
    #sleep(2)

    # ##### ###### ###### MAKE NETWORK ###### ###### ######
    # ##### ###### ###### MAKE NETWORK ###### ###### ######
    # ##### ###### ###### MAKE NETWORK ###### ###### ######
    print("make_net..")
    categories = v.len()
    net = make_net(categories, categories, hidden_size=HIDDEN_NEURONS)

    predict_100(net,v,X,y)
    net.fit(X, y, nb_epoch=1, batch_size=WINDOW_LEN, show_accuracy=True, validation_split=0.1, verbose=1)

    zipped = list(zip(X, y))
    train_epochs=1.0
    trained_amount=1.0
    for i in range(10000):
        # print("############# Predicting! iteration ", i)
        # print("############# Predicting! iteration ", i)
        predict_100(net, v, X, y)
        # make sub-sample of train data and train with it

        # Train in mini-batches in stead of fll set? Did I do this because of 32 bit memory limits?
        samp = sample(list(zipped), min(len(zipped), MINIBATCH))
        Xpart = np.array([sx for sx, sy in samp])
        ypart = np.array([sy for sx, sy in samp])

        #fit for x seconds
        initial_time = time()
        SECONDS = 30
        print("Fitting for at least {} seconds..".format(SECONDS))
        train_epochs =int(max(1,0.5*trained_amount + 0.5*train_epochs))
        trained_amount=0
        while time() < initial_time + SECONDS:
            trained_amount+=train_epochs
            net.fit(X, y, nb_epoch=train_epochs, batch_size=len(X), show_accuracy=True, validation_split=0.15, verbose=1)

        # print(run_test(net,v))


def run_old():
    return

    # ##### ###### ######  TRAIN ###### ###### ######
    # ##### ###### ######  TRAIN ###### ###### ######
    # ##### ###### ######  TRAIN ###### ###### ######
    EPOCHS = EPOCHS_PER_CYCLE * CYCLES

    errs = []
    dt0 = DELTA0INIT
    for i in range(2000):
        dt0 *= 0.98
        net, train_errors = train_network(TrainDS, net, EPOCHS, EPOCHS_PER_CYCLE, CYCLES, dt0)
        errs.extend(train_errors)

        if random() > 0.5:
            if REDDITOR and not TEXT_FILE:
                prime_txt = redditor_text(REDDITOR, 20, True)
            else:
                first_sentence = [x for x in [pruned.find(" "), pruned.find("."), pruned.find("?"), pruned.find("!")] if
                                  x > 5]
                if len(first_sentence):
                    end = min(first_sentence)
                else:
                    end = 20
                prime_txt = pruned[:end + 1]

            print(u"!!! Priming with ##{}##".format(prime_txt))
            primer = v.to_matrix(prime_txt)
            net.reset()
            for inp in primer:
                net.activate(inp)
                # if ". " in v.dictionary:
                # net.activate(v.to_vector(". "))

        result = run_test(net, v)
        if type(result) == type(""):
            result = unicode(result, errors='replace')
        else:
            result = result.encode('utf-8', errors='xmlcharreplace')

        print("\n%%%% Epoch, delta0: ", i, dt0)
        print("--------------------------")
        try:
            print(result)
        except:
            print("FAILED :(, probably Unicode")
        print("-------------------------\n")
        # # new training text
        if i % RETRAIN_FREQUENCY == RETRAIN_FREQUENCY - 1:
            print("################## New training set!")
            if 0:
                pass
            else:
                start = long(random() * (len(input_text) - TRAIN_LEN - 100))
                start = input_text.index(".", start)
                pruned = input_text[start:start + TRAIN_LEN]
            # print(pruned)
            # input_mat.extend(v.to_matrix(pruned))
            input_mat = v.to_matrix(pruned)
            # input_mat = text_2_matrix(pruned)
            TrainDS = make_dataset(input_mat, v)
            # ######## net.reset() ### maybe use this?

    plot_result(errs, "Error graph")


if __name__ == "__main__":
    run()
