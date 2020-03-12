import pandas as pd
import numpy as np
import sys
parameters = sys.argv[1:]



train = pd.read_csv(parameters[0],header=None).values
test = pd.read_csv(parameters[1],header=None).values
_lambda = 1
_iterations = 200000

def fit(X, Y):
    m, n_features = X.shape[0], X.shape[1]
    weights = np.zeros(n_features)
    for i in range(_iterations):
        eta = 1. /(_lambda*(i+1))
        j = np.random.choice(m, 1)[0]
        x, y = X[j], Y[j]
        score = weights.dot(x)
        if y*score < 1:
            weights = (1 - eta*_lambda)*weights + eta*y*x
        else:
            weights = (1 - eta*_lambda)*weights
    return weights

def train_model(train):
    models = {}
    count = 0
    for i in range(10):
        for j in range(i+1,10):
            data = train[(train[:,-1]==i) + (train[:,-1]==j)]
            _X = data[:,:-1]
            X = np.ones(data.shape)
            X[:,:-1]=_X
            Y = data[:,-1]
            Y[Y==i] = -1
            Y[Y==j] = 1
            models[(i,j)] = fit(X,Y)
    return models

def predict(X,models):
    count = np.zeros((X.shape[0],10))
    for i in range(10):
        for j in range(i+1,10):
            pred = X.dot(models[(i,j)])
            count[:,i] += (pred<0).astype(int)
            count[:,j] += (pred>=1).astype(int)
            
    return np.argmax(count,1)

models = train_model(train)
X = np.ones(test.shape)
X[:,:-1] = test[:,:-1]
pred = predict(X,models)
np.savetxt(parameters[2],pred,delimiter='\n')