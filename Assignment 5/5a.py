#your code for part a goes here
import numpy as np
import pandas as pd
from math import log
import re
import sys
parameters = sys.argv[1:]

train = pd.read_csv(parameters[0]).values
test = pd.read_csv(parameters[1]).values
train[:,1] = (train[:,1]=='positive').astype(int)

def process_message(m):
    m = m.lower()
    m = m.replace('\'ll','')
    m = m.replace('\'s','')
    regex = re.compile('[^a-z\s]')
    m = regex.sub('', m)
    return m.split(" ")
    
count_pos = dict()
count_neg = dict()
prob_pos = dict()
prob_neg = dict()
num_feed = train.shape[0]
pos_feed, neg_feed = train[:,1].sum(),num_feed-train[:,1].sum()
prob_neg_feed, prob_pos_feed = neg_feed/num_feed, pos_feed/num_feed
neg_words = 0
pos_words = 0
for i in range(num_feed):
    message_processed = process_message(train[i,0])
    for word in message_processed:
        if train[i,1]:
            count_pos[word] = count_pos.get(word, 0) + 1
            pos_words += 1
        else:
            count_neg[word] = count_neg.get(word, 0) + 1
            neg_words += 1
pos_denom = (pos_words+len(list(count_pos.keys())))
neg_denom = (neg_words+len(list(count_neg.keys())))
for word in count_pos:
    prob_pos[word] = ((count_pos.get(word,0)+1)/pos_denom)
for word in count_neg:
    prob_neg[word] = ((count_neg.get(word,0)+1)/neg_denom)
  
  
def classify(processed_message):
    p_pos, p_neg = 0, 0
    for word in processed_message:                
        p_pos += log(prob_pos.get(word,1/pos_denom))
        p_neg += log(prob_neg.get(word,1/neg_denom))
    p_pos += log(prob_pos_feed)
    p_neg += log(prob_neg_feed)
    return p_pos >= p_neg

def predict(testData):
    result = []
    for i in range(testData.shape[0]):
        processed_message = process_message(testData[i,0])
        result.append(int(classify(processed_message)))
    return result
    
trainpred = predict(train)
trainaccu = (train[:,1]==trainpred).astype(int).sum()/train.shape[0]*100
print("Accuracy on training set %.2f" % trainaccu,"%")
pred = predict(test)
np.savetxt(parameters[2],pred,delimiter="\n")
