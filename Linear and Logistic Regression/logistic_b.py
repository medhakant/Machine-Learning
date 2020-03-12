import sys
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
param = sys.argv[1:]
#read the data
data = pd.read_csv(param[0],header=None)
with open(param[2]) as f:
    parameter = f.readlines()
parameter = [x.strip() for x in parameter]
parameter = [x.split(",") for x in parameter]
#the softmax function
def softmax(z):
    e = np.exp(z)
    return e/e.sum(axis=1)[:,None]
#preprocess X with one hot encoding
def preprocessX(data): 
    col = 0
    for i in range(8):
        col +=data[i].unique().size
    X = np.ones((data.shape[0],col+1))
    X[:,1]= (data[0]=='usual').astype(int)
    X[:,2]= (data[0]=='pretentious').astype(int)
    X[:,3]= (data[0]=='great_pret').astype(int)
    X[:,4]= (data[1]=='proper').astype(int)
    X[:,5]= (data[1]=='less_proper').astype(int)
    X[:,6]= (data[1]=='improper').astype(int)
    X[:,7]= (data[1]=='critical').astype(int)
    X[:,8]= (data[1]=='very_crit').astype(int)
    X[:,9]= (data[2]=='complete').astype(int)
    X[:,10]= (data[2]=='completed').astype(int)
    X[:,11]= (data[2]=='incomplete').astype(int)
    X[:,12]= (data[2]=='foster').astype(int)
    X[:,13]= (data[3]=='1').astype(int)
    X[:,14]= (data[3]=='2').astype(int)
    X[:,15]= (data[3]=='3').astype(int)
    X[:,16]= (data[3]=='more').astype(int)
    X[:,17]= (data[4]=='convenient').astype(int)
    X[:,18]= (data[4]=='less_conv').astype(int)
    X[:,19]= (data[4]=='critical').astype(int)
    X[:,20]= (data[5]=='convenient').astype(int)
    X[:,21]= (data[5]=='inconv').astype(int)
    X[:,22]= (data[6]=='nonprob').astype(int)
    X[:,23]= (data[6]=='slightly_prob').astype(int)
    X[:,24]= (data[6]=='problematic').astype(int)
    X[:,25]= (data[7]=='recommended').astype(int)
    X[:,26]= (data[7]=='priority').astype(int)
    X[:,27]= (data[7]=='not_recom').astype(int)
    return X

#preprocess Y with one hot encoding
def preprocessY(data):
    col = data[8].unique().size
    Y = np.ones((data.shape[0],col))
    Y[:,0]= (data[8]=='not_recom').astype(int)
    Y[:,1]= (data[8]=='recommend').astype(int)
    Y[:,2]= (data[8]=='very_recom').astype(int)
    Y[:,3]= (data[8]=='priority').astype(int)
    Y[:,4]= (data[8]=='spec_prior').astype(int)
    return Y

#perform logistic regression
def logistic_regression(X,Y,strategy,lr,max_iter,batch_size):
    W = np.zeros((X.shape[1],Y.shape[1]))
    num_batches = int(X.shape[0]/batch_size)
    Xs = np.split(X,num_batches)
    Ys = np.split(Y,num_batches)
    for i in range(max_iter):
        rate = float(parameter[1][0])
        for j in range(num_batches):
            Z = softmax(np.dot(Xs[j],W))
            gradient = np.dot(Xs[j].T,(Z-Ys[j]))/Ys[j].shape[0]
            if(strategy==1):
                W -= lr*gradient
            elif(strategy==2):
                W -= (lr/np.sqrt(i+1))*gradient
            elif(strategy==3):
                alpha = float(parameter[1][1])
                beta = lr
                initial_loss = -np.sum(np.multiply(Y,np.log(softmax(np.dot(X,W)))))/2*Y.shape[0]
                tempW = W - rate*gradient
                loss = -np.sum(np.multiply(Y,np.log(softmax(np.dot(X,tempW)))))/2*Y.shape[0]
                while(loss > initial_loss + alpha*np.sum(gradient**2)):
                    rate *= beta
                    tempW = W - rate*gradient
                    loss = -np.sum(np.multiply(Y,np.log(softmax(np.dot(X,tempW)))))/2*Y.shape[0]
                W -= rate*gradient
            
    return W

#function to give prediction class
def prediction(X,W):
    Z = np.argmax(softmax(np.dot(X,W)),axis=1)
    output_label = ['not_recom','recommend','very_recom','priority','spec_prior']
    Z = [output_label[i] for i in Z]
    return Z

#perform the operatoins
X = preprocessX(data)
Y = preprocessY(data)
W = logistic_regression(X,Y,int(parameter[0][0]),float(parameter[1][-1]),int(parameter[2][0]),int(parameter[3][0]))
test = preprocessX(pd.read_csv(param[1],header=None))
pred = prediction(test,W)

#save predictions and weights to file
with open(param[3], 'w',newline="") as myfile:
    wr = csv.writer(myfile)
    wr.writerow(pred)
np.savetxt(param[4],W,delimiter=',')