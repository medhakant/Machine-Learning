import numpy as np
import pandas as pd
import sys
param = sys.argv[1:]

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork:
    def __init__(self, X, Y, n, lr, it,batch_size):
        self.weights1   = np.zeros((X.shape[1],n))
        self.weights2   = np.zeros((n+1,1))
        self.lr         = lr
        self.num_batches= int(X.shape[0]/batch_size)
        self.Xs         = np.split(X,self.num_batches)
        self.Ys         = np.split(Y,self.num_batches)
        for i in range(it):
            self.output     = np.zeros(self.Ys[i%self.num_batches].shape)
            self.inp        = self.Xs[i%self.num_batches]
            self.y          = self.Ys[i%self.num_batches]
            self.layer1     = np.ones((self.inp.shape[0],n+1))
            self.feedforward()
            self.backprop()
        
        

    def feedforward(self):
        self.layer1[:,1:] = sigmoid(np.dot(self.inp, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (self.output-self.y))
        d_weights1 = np.dot(self.inp.T, (np.dot((self.output-self.y), self.weights2.T))*sigmoid_derivative(self.layer1))
        self.weights2 -= (self.lr*d_weights2)/self.y.shape[0]
        self.weights1 -= (self.lr*d_weights1[:,1:])/self.y.shape[0]
        

data = pd.read_csv(param[0],header=None)
data = data.values
X = np.concatenate((np.ones_like(data[:,0:1]),data[:,:-1]),axis=1)
Y = data[:,-1]
Y = Y.reshape(Y.shape[0],1)
with open(param[1]) as f:
    parameter = f.readlines()
parameter = [x.strip() for x in parameter]
parameter = [x.split(",") for x in parameter]
n = NeuralNetwork(X,Y,int(parameter[4][0]),float(parameter[1][0]),int(parameter[2][0]),int(parameter[3][0]))
f_handle = open(param[2], 'a')
np.savetxt(f_handle,n.weights1,delimiter='\n')
np.savetxt(f_handle,n.weights2,delimiter='\n')
f_handle.close()
