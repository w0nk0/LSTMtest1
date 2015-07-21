import numpy as np
from time import sleep

class Fakemodel:
    def __init__(self, realmodel=None):
        self.data_len=0
        self.realmodel = realmodel

    def compile(optimizer,loss):
        print("FAKE_COMPILING")
        if self.realmodel:
            self.realmodel.compile(optimizer=optimier,loss=loss)

    def fit(self,X,y,nb_epoch=1,batch_size=128):
        sleep(nb_epoch)
        print("FAKE_FIT, data shape:",np.array(X).shape)
        self.check_len(X)
        if self.realmodel:
            self.realmodel.fit(X,y,nb_epoch,batch_size)

    def check_len(self,X):
        newlen = len(X[0])
        if newlen != self.data_len:
            print("New X len?! -> ",newlen)
        self.data_len = newlen

    def predict(self,X, verbose=0):
        sleep(0.05)
        self.check_len(X)
        print("\rpredict X shape:",X.shape,end="")
        if self.realmodel:
            return self.realmodel.predict(X, verbose)
        return X[0]
