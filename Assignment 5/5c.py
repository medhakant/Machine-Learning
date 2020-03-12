#your code for part a goes here
import numpy as np
import pandas as pd
from math import log
import re
import sys
from nltk.stem import PorterStemmer 

stop_words = ['yourselves', 'few', 'won', 'between', 'who', 'both', 'because', 
'in', 'some', 'they', 't', 'there', 'ourselves', 'down', 'with', 'y', 'having', 
'just', 'needn', 'theirs', 'be', 'again', "you'll", 'should', 'out', 'these', 
'shan', 'me', 'hadn', 'on', 'how', 'under', "you'd", 'so', 'our', 've',
'himself', 'what', 'below', 'aren', 'why', 'll', 'herself', 'after', "weren't",
'myself', 'which', 'mustn', 'couldn', "won't", 'i', 'a', 'such', "didn't",
'the', 'same', 'when', 'this', "should've", "haven't", 'd', "doesn't", 'if',
"hasn't", 'does', 'over', 'now', 'not', "aren't", 'did', 'too', 'don', 'my',
'about', 'haven', 'as', 'no', "couldn't", 'whom', 'am', 'are', 'during', 'him',
"shouldn't", 'here', 'into', 'o', 'her', 'against', "that'll", 'off', 
'themselves', 'shouldn', 'most', 'those', 'was', "shan't", 'or', 'for',
'until', 'weren', 'all', 'ma', 'it', 'further', "don't", 'doesn', 'to', 
'itself', 'an', 'of', 'will', 'then', 're', 'each', 'ain', 'wouldn', 'had', 
'yourself', 'more', 'own', "mightn't", "you've", "it's", 's', 'once', 'above',
'hasn', "hadn't", "wasn't", 'than', 'were', "she's", 'hers', 'them', 'nor',
'doing', 'up', 'their', 'm', 'we', "you're", 'is', 'do', 'yours', 'before',
'wasn', 'while', 'can', 'has', "isn't", "mustn't", "needn't", 'didn', 'by',
'ours', "wouldn't", 'you', 'his', 'where', 'at', 'have', 'and', 'through', 
'other', 'very', 'isn', 'only', 'been', 'that', 'from', 'your', 'she', 'its',
'being', 'but', 'mightn', 'any', 'he']
parameters = sys.argv[1:]

ps = PorterStemmer()
train = pd.read_csv(parameters[0]).values
test =  pd.read_csv(parameters[1]).values
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
neg_words = 0
pos_words = 0

pos_feed, neg_feed = train[:,1].sum(),num_feed-train[:,1].sum()
prob_neg_feed, prob_pos_feed = neg_feed/num_feed, pos_feed/num_feed

count_pos_bi = dict()
count_neg_bi = dict()
prob_pos_bi = dict()
prob_neg_bi = dict()
neg_bi = 0
pos_bi = 0

for i in range(num_feed):
    message_processed = process_message(train[i,0])
    for word in message_processed:
        if word not in stop_words:
            # word = ps.stem(word)
            if train[i,1]:
                count_pos[word] = count_pos.get(word, 0) + 1
                pos_words += 1
            else:
                count_neg[word] = count_neg.get(word, 0) + 1
                neg_words += 1

# for i in range(num_feed):
#     message_processed = process_message(train[i,0])
#     for i in range(len(message_processed)-1):
#         bi = (message_processed[i],message_processed[i+1]) 
#         if train[i,1]:
#             count_pos_bi[bi] = count_pos_bi.get(bi, 0) + 1
#             pos_bi += 1
#         else:
#             count_neg_bi[bi] = count_neg_bi.get(bi, 0) + 1
#             neg_bi += 1
            
pos_denom = (pos_words+len(list(count_pos.keys())))
neg_denom = (neg_words+len(list(count_neg.keys())))
for word in count_pos:
    prob_pos[word] = ((count_pos.get(word,0)+1)/pos_denom)
for word in count_neg:
    prob_neg[word] = ((count_neg.get(word,0)+1)/neg_denom)
    
# pos_denom_bi = (pos_bi+len(list(count_pos_bi.keys())))
# neg_denom_bi = (neg_bi+len(list(count_neg_bi.keys())))
# for word in count_pos_bi:
#     prob_pos[word] = ((count_pos_bi.get(word,0)+1)/pos_denom_bi)
# for word in count_neg_bi:
#     prob_neg[word] = ((count_neg_bi.get(word,0)+1)/neg_denom_bi)
  
  
def classify(processed_message):
    p_pos, p_neg = 0, 0
    for word in processed_message:
        if word not in stop_words:
            # word = ps.stem(word)
            p_pos += log(prob_pos.get(word,1/pos_denom))
            p_neg += log(prob_neg.get(word,1/neg_denom))
    # for i in range(len(message_processed)-1):
    #     p_pos += log(prob_pos_bi.get((message_processed[i],message_processed[i+1]),1/pos_denom_bi))
    #     p_neg += log(prob_neg_bi.get((message_processed[i],message_processed[i+1]),1/neg_denom_bi))
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
