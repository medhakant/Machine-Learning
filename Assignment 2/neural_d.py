#Your code goes here
import numpy as np
import pandas as pd
import sys
import time
start_time = time.time()
param = sys.argv[1:]

def sigmoid(x):
    return np.tanh(x)

def softmax(z):
    e = np.exp(z)
    return e/e.sum(axis=1)[:,None]

def sigmoid_derivative(x):
    return np.exp(-x)/((1+np.exp(-x))*(1+np.exp(-x)))

# class NeuralNetwork:
#     def __init__(self, x, y,n,m,o, lr, it,batch_size):
#         self.weights1   = np.random.randn(x.shape[1],n)
#         self.weights2   = np.random.randn(n+1,m)
#         self.weights3   = np.random.randn(m+1,o)
#         self.weights4   = np.random.randn(o+1,10)
#         self.lr         = lr
#         self.num_batches= int((x.shape[0]-1)//batch_size) + 1
#         self.Xs         = np.array_split(x,self.num_batches)
#         self.Ys         = np.array_split(y,self.num_batches)
#         for i in range(it):
#             self.output     = np.zeros(self.Ys[i%self.num_batches].shape)
#             self.input      = self.Xs[i%self.num_batches]
#             self.y          = self.Ys[i%self.num_batches]
#             self.layer1     = np.ones((self.input.shape[0],n+1))
#             self.layer2     = np.ones((self.input.shape[0],m+1))
#             self.layer3     = np.ones((self.input.shape[0],o+1))
#             self.lr         = lr/np.sqrt(it+1)
#             self.feedforward()
#             self.backprop()
        
        

#     def feedforward(self):
#         self.layer1[:,1:] = sigmoid(np.dot(self.input, self.weights1))
#         self.layer2[:,1:] = sigmoid(np.dot(self.layer1, self.weights2))
#         self.layer3[:,1:] = sigmoid(np.dot(self.layer2, self.weights3))
#         self.output       = softmax(np.dot(self.layer3, self.weights4))

#     def backprop(self):
#         error = self.y - self.output
#         d_weights4 = np.dot(self.layer3.T, error)
#         d_weights3 = np.dot(self.layer2.T, (np.dot(error, self.weights4[1:,:].T))*sigmoid_derivative(self.layer3[:,1:]))
#         error = (np.dot(error, self.weights4[1:,:].T))*sigmoid_derivative(self.layer3[:,1:])
#         d_weights2 = np.dot(self.layer1.T, (np.dot(error, self.weights3[1:,:].T))*sigmoid_derivative(self.layer2[:,1:]))
#         error = (np.dot(error, self.weights3[1:,:].T))*sigmoid_derivative(self.layer2[:,1:])
#         d_weights1 = np.dot(self.input.T, (np.dot(error, self.weights2[1:,:].T))*sigmoid_derivative(self.layer1[:,1:]))
        
#         self.weights1 += self.lr*d_weights1/self.y.shape[0]
#         self.weights2 += self.lr*d_weights2/self.y.shape[0]
#         self.weights3 += self.lr*d_weights3/self.y.shape[0]
#         self.weights4 += self.lr*d_weights4/self.y.shape[0]
        
#     def predict(self,x):
#         temp1     = np.ones((x.shape[0],self.layer1.shape[1]))
#         temp1[:,1:] = sigmoid(np.dot(x, self.weights1))
#         temp2     = np.ones((x.shape[0],self.layer2.shape[1]))
#         temp2[:,1:] = sigmoid(np.dot(temp1, self.weights2))
#         temp3     = np.ones((x.shape[0],self.layer3.shape[1]))
#         temp3[:,1:] = sigmoid(np.dot(temp2, self.weights3))
#         return np.argmax(softmax(np.dot(temp3, self.weights4)),axis=1)

# class NeuralNetwork:
#     def __init__(self, x, y, n, lr, it,batch_size):
#         self.weights1   = np.random.randn(x.shape[1],n)
#         self.weights2   = np.random.randn(n+1,10)
#         self.lr         = lr
#         self.num_batches= int((x.shape[0]-1)//batch_size) + 1
#         self.Xs         = np.array_split(x,self.num_batches)
#         self.Ys         = np.array_split(y,self.num_batches)
#         for i in range(it):
#             self.output     = np.zeros(self.Ys[i%self.num_batches].shape)
#             self.input      = self.Xs[i%self.num_batches]
#             self.y          = self.Ys[i%self.num_batches]
#             self.layer1     = np.ones((self.input.shape[0],n+1))
#             self.feedforward()
#             self.backprop()
        
        

#     def feedforward(self):
#         self.layer1[:,1:] = sigmoid(np.dot(self.input, self.weights1))
#         self.output = softmax(np.dot(self.layer1, self.weights2))

#     def backprop(self):
#         d_weights2 = np.dot(self.layer1.T, (self.y - self.output))
#         d_weights1 = np.dot(self.input.T, (np.dot((self.y - self.output), self.weights2[1:,:].T))*sigmoid_derivative(self.layer1[:,1:]))
        
#         self.weights1 += self.lr*d_weights1/self.y.shape[0]
#         self.weights2 += self.lr*d_weights2/self.y.shape[0]
        
#     def predict(self,x):
#         temp     = np.ones((x.shape[0],self.layer1.shape[1]))
#         temp[:,1:] = sigmoid(np.dot(x, self.weights1))
#         return np.argmax(softmax(np.dot(temp, self.weights2)),axis=1)

class NeuralNetwork:
    def __init__(self, x, y,n,m, lr,batch_size):
        self.weights1   = np.random.randn(x.shape[1],n)
        self.weights2   = np.random.randn(n+1,m)
        self.weights3   = np.random.randn(m+1,10)
        self.lr         = lr
        self.num_batches= int((x.shape[0]-1)//batch_size) + 1
        self.Xs         = np.array_split(x,self.num_batches)
        self.Ys         = np.array_split(y,self.num_batches)
        i = 0
        while True:
            self.output     = np.zeros(self.Ys[i%self.num_batches].shape)
            self.input      = self.Xs[i%self.num_batches]
            self.y          = self.Ys[i%self.num_batches]
            self.layer1     = np.ones((self.input.shape[0],n+1))
            self.layer2     = np.ones((self.input.shape[0],m+1))
            self.feedforward()
            self.backprop()
            i = i+1
            elapsed_time = time.time() - start_time
            if elapsed_time>570:
                break
        
        

    def feedforward(self):
        self.layer1[:,1:] = sigmoid(np.dot(self.input, self.weights1))
        self.layer2[:,1:] = sigmoid(np.dot(self.layer1, self.weights2))
        self.output       = softmax(np.dot(self.layer2, self.weights3))

    def backprop(self):
        error = self.y - self.output
        d_weights3 = np.dot(self.layer2.T, error)
        d_weights2 = np.dot(self.layer1.T, (np.dot(error, self.weights3[1:,:].T))*sigmoid_derivative(self.layer2[:,1:]))
        error = (np.dot(error, self.weights3[1:,:].T))*sigmoid_derivative(self.layer2[:,1:])
        d_weights1 = np.dot(self.input.T, (np.dot(error, self.weights2[1:,:].T))*sigmoid_derivative(self.layer1[:,1:]))
        
        self.weights1 += self.lr*d_weights1/self.y.shape[0]
        self.weights2 += self.lr*d_weights2/self.y.shape[0]
        self.weights3 += self.lr*d_weights3/self.y.shape[0]
        
    def predict(self,x):
        temp1     = np.ones((x.shape[0],self.layer1.shape[1]))
        temp1[:,1:] = sigmoid(np.dot(x, self.weights1))
        temp2     = np.ones((x.shape[0],self.layer2.shape[1]))
        temp2[:,1:] = sigmoid(np.dot(temp1, self.weights2))
        return np.argmax(softmax(np.dot(temp2, self.weights3)),axis=1)

data = pd.read_csv(param[0],header=None)
def preprocessY(data):
    col = data[1024].unique().size
    Y = np.ones((data.shape[0],col))
    for i in range(col):
         Y[:,i]= (data[1024]==i).astype(int)
    return Y
    
Y = preprocessY(data)
data = data.values/255
X = np.concatenate((np.ones_like(data[:,0:1]),data[:,:-1]),axis=1)
test = pd.read_csv(param[1],header=None)
test = test.values/255
test = np.concatenate((np.ones_like(test[:,0:1]),test[:,:-1]),axis=1)
predY = test[:,-1]
# n = NeuralNetwork(X,Y,500,100,40,0.005,3000,500)
# n = NeuralNetwork(X,Y,100,0.005,20000,500)  #0.21
# n = NeuralNetwork(X,Y,200,0.05,20000,500)  #0.25
# n = NeuralNetwork(X,Y,100,1,20000,500) #0.27
n = NeuralNetwork(X,Y,1000,100,0.5,500)
pred = n.predict(test)
f_handle = open(param[2], 'a')
np.savetxt(f_handle,pred,delimiter='\n')
f_handle.close()
