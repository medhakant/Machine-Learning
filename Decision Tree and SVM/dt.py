import sys
import numpy as np
import pandas as pd
parameters = sys.argv[1:]

train = pd.read_csv(parameters[0])
test = pd.read_csv(parameters[2])
valid = pd.read_csv(parameters[1])

def accuracy_metric(actual, predicted):
    return (actual == predicted).astype(int).sum()/len(actual)

def gini_index(groups):
    n_instances = float(sum([group.shape[0] for group in groups]))
    gini = 0.0
    for group in groups:
        size = group.shape[0]
        if size == 0:
            continue
        score = 0.0
        p = (group.values[:,-1]==1).astype(int).sum() / size
        score += p * p
        p = (group.values[:,-1]==0).astype(int).sum() / size
        score += p * p            
        gini += (1.0 - score) * (size / n_instances)
    return gini

def get_split(data):
    col = data.columns
    index, value, score, groups = 999, 999, 999, None
    for i in range(len(col)-1):
        if data.dtypes[i]=='int64':
            lower = data[col[i]].min()
            upper = data[col[i]].max()
            if lower==upper:
                continue
            v = list(set(np.random.randint(lower,upper,10)))
            for m in v:
                gr = []
                gr.append(data[data[col[i]]<m])
                gr.append(data[data[col[i]]>=m])
                s = gini_index(gr)
                if s < score:
                    score = s
                    index = i
                    groups = gr
                    value = m
                
        else:
            gr = []
            v = []
            for j in data[col[i]].unique():
                gr.append(data[data[col[i]]==j])
                v.append(j)
            s = gini_index(gr)
            if s < score:
                score = s
                index = i
                groups = gr
                value = v
    return {'index':index, 'value':value, 'groups':groups}

class Node:
    def __init__(self,data,depth,max_depth):
        self.col = data.columns
        self.pred = 0
        self.depth = depth
        self.true_data = (data[self.col[-1]]==1).astype(int).sum()
        self.false_data = (data[self.col[-1]]==0).astype(int).sum()
        if self.true_data > self.false_data:
            self.pred = 1
        self.terminal = (len(data[self.col[-1]].unique())==1) or (depth > max_depth) or ((self.true_data/self.false_data) > 0.99) or ((self.false_data/self.true_data) < 0.99) or (data.shape[0] < 5)
        self.index = 0
        self.value = 0
        self.child = 0
        self.dtype = None
        if not self.terminal:
            self.gs = get_split(data)
            self.index = self.gs['index']
            self.value = self.gs['value']
            self.dtype = data.dtypes[self.index]
            self.child = []
            if len(self.gs)>1:
                for group in self.gs['groups']:
                    self.child.append(Node(group,depth+1,max_depth))
            else:
                self.terminal = True
                
    def predict(self,data):
        if self.terminal:
            return self.pred
        elif self.dtype == 'int64':
            if data[self.col[self.index]] < self.value:
                return self.child[0].predict(data)
            else:
                return self.child[1].predict(data)
        else:
            if data[self.index] in self.value:
                return self.child[self.value.index(data[self.index])].predict(data)   
            else:
                return self.pred          


model = Node(train,0,4)


valid_pred = []
for i in range(valid.shape[0]):
    valid_pred.append(model.predict(valid.loc[i]))

pred = []
for i in range(test.shape[0]):
    pred.append(model.predict(test.loc[i]))

np.savetxt(parameters[3],valid_pred, delimiter='\n')
np.savetxt(parameters[4],pred, delimiter='\n')