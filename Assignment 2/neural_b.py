#Your code goes here
import numpy as np
import pandas as pd
import sys
param = sys.argv[1:]

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def softmax(z):
    e = np.exp(z)
    return e/e.sum(axis=1)[:,None]

def sigmoid_derivative(x):
    return np.exp(-x)/((1+np.exp(-x))*(1+np.exp(-x)))

class NeuralNetwork:
    def __init__(self, X, Y, lr, it, batch_size, n):
        self.weights1   = np.zeros((X.shape[1],n))
        self.weights2   = np.zeros((n+1,Y.shape[1]))
        self.lr         = lr
        self.num_batches= int(X.shape[0]/batch_size)
        self.Xs         = np.split(X,self.num_batches)
        self.Ys         = np.split(Y,self.num_batches)
        for i in range(it):
            self.output     = np.zeros(self.Ys[i%self.num_batches].shape)
            self.input      = self.Xs[i%self.num_batches]
            self.y          = self.Ys[i%self.num_batches]
            self.layer1     = np.ones((self.input.shape[0],n+1))
            self.lr = lr/(np.sqrt(i+1)*10)
            self.feedforward()
            self.backprop()
        
        

    def feedforward(self):
        self.layer1[:,1:] = sigmoid(np.dot(self.input, self.weights1))
        self.output = softmax(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (self.y - self.output))
        d_weights1 = np.dot(self.input.T, (np.dot((self.y - self.output), self.weights2[1:,:].T))*sigmoid_derivative(self.layer1[:,1:]))
        self.weights1 += self.lr*d_weights1/self.y.shape[0]
        self.weights2 += self.lr*d_weights2/self.y.shape[0]
        
data = pd.read_csv(param[0],header=None)
def preprocessY(data):
    col = data[1024].unique().size
    Y = np.zeros((data.shape[0],col))
    for i in range(col):
         Y[:,i]= (data[1024]==i).astype(int)
    return Y
    
Y = preprocessY(data)
data = data.values
X = np.concatenate((np.ones_like(data[:,0:1]),data[:,:-1]),axis=1)
with open(param[1]) as f:
    parameter = f.readlines()
parameter = [x.strip() for x in parameter]
parameter = [x.split(",") for x in parameter]
print(parameter)
n = NeuralNetwork(X,Y,float(parameter[1][0]),int(parameter[2][0]),int(parameter[3][0]),int(parameter[4][0]))
f_handle = open(param[2], 'a')
np.savetxt(f_handle,n.weights1,delimiter='\n')
np.savetxt(f_handle,n.weights2,delimiter='\n')
f_handle.close()
