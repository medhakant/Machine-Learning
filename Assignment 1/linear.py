import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
mode = sys.argv[1]
parameters = sys.argv[2:]

def error_function(W,X,Y):
    return sum((X.dot(W)-Y)*(X.dot(W)-Y))/sum(Y*Y)

if(mode=='a'):
    data = pd.read_csv(parameters[0],header=None)
    data = data.values
    Y = data[:,-1]
    X = np.concatenate((np.ones_like(data[:,0:1]),data[:,:-1]),axis=1)
    XTrans = X.transpose()
    Xinv = np.linalg.inv(XTrans.dot(X))
    W = (Xinv.dot(XTrans)).dot(Y)
    test = pd.read_csv(parameters[1],header=None)
    test = test.values
    testX = np.zeros((test.shape[0],(test.shape[1])+1))
    testX[:,0] = np.ones(test.shape[0])
    testX[:,1:] = test
    output = testX.dot(W)
    np.savetxt(parameters[2],output, delimiter='\n')
    np.savetxt(parameters[3],W,delimiter='\n')

if(mode=='b'):
    data = pd.read_csv(parameters[0],header=None)
    data = data.values
    Y = data[:,-1]
    X = np.concatenate((np.ones_like(data[:,0:1]),data[:,:-1]),axis=1)
    size = int(X.shape[0]/10)
    test = pd.read_csv(parameters[1],header=None)
    test = test.values
    testX = np.zeros((test.shape[0],(test.shape[1])+1))
    testX[:,0] = np.ones(test.shape[0])
    testX[:,1:] = test
    lamb = np.genfromtxt(parameters[2],delimiter=',')
    error = []
    for l in lamb:
        temp = 0
        for i in range(10):
            Xtrain = np.vstack([X[0:i*size,:],X[(i+1)*size:,:] ])
            Xval = X[i*size:(i+1)*size,:]
            Ytrain = np.concatenate([Y[0:size*i],Y[size*(i+1):]])
            Yval = Y[size*i:size*(i+1)]
            XTrans = Xtrain.transpose()
            dim = XTrans.shape[0]
            Xinv = np.linalg.inv(XTrans.dot(Xtrain) + l*np.eye(dim))
            W = (Xinv.dot(XTrans)).dot(Ytrain)
            temp = temp + error_function(W,Xval,Yval)
        error.append(temp)
    l = lamb[error.index(min(error))]
    print(l)
    XTrans = X.transpose()
    dim = XTrans.shape[0]
    Xinv = np.linalg.inv(XTrans.dot(X) + l*np.eye(dim))
    W = (Xinv.dot(XTrans)).dot(Y)
    output = testX.dot(W)
    np.savetxt(parameters[3],output, delimiter='\n')
    np.savetxt(parameters[4],W,delimiter='\n')
    
if(mode =='c'):
    data = pd.read_csv(parameters[0],header=None)
    data = data.values
    Y = data[:,-1]
    X = data[:,:-1]
    Xlog = np.log(1+abs(X))
    Xsq = X**2
    Xsin = np.sin(X)
    Xupon = 1/(1+abs(X))
    Xlogx = np.log(1+abs(X))*X
    X = np.concatenate((np.ones_like(data[:,0:1]),X,Xlog,Xsq,Xsin,Xupon,Xlogx),axis=1)
    size = int(X.shape[0]/4)
    test = pd.read_csv(parameters[1],header=None)
    test = test.values
    testlog = np.log(1+abs(test))
    testsq = test**2
    testsin = np.sin(test)
    testupon = 1/(1+abs(test))
    testlogx = np.log(1+abs(test))*test
    test = np.concatenate((np.ones_like(test[:,0:1]),test,testlog,testsq,testsin,testupon,testlogx),axis=1)    
    lamb = [0.006,0.009,0.01,0.03,0.1,0.3,1,100,1000]
    error = []
    for l in lamb:
        temp = 0
        reg = linear_model.LassoLars(alpha=l)
        for i in range(4):
            Xtrain = np.vstack([X[0:i*size,:],X[(i+1)*size:,:] ])
            Xval = X[i*size:(i+1)*size,:]
            Ytrain = np.concatenate([Y[0:size*i],Y[size*(i+1):]])
            Yval = Y[size*i:size*(i+1)]
            reg.fit(Xtrain,Ytrain)         
            temp = temp + error_function(reg.coef_,Xval,Yval)
        error.append(temp)
    l = lamb[error.index(min(error))]
    print(l)
    reg = linear_model.LassoLars(alpha=l,max_iter=1500,eps=2.7e-19)
    reg.fit(X,Y)
    output = reg.predict(test)
    np.savetxt(parameters[2],output, delimiter='\n')