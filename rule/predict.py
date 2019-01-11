import json
from sklearn.metrics import f1_score,recall_score,precision_score
import numpy as np
import sys

def type2id(data):
    for name in data:
        return name

file_name = sys.argv[1]
data = {}
for i,line in enumerate(open(file_name)):
    temp = json.loads(line)
    node = {}
    for j in range(len(temp['nodes'])):
        node[ tuple(temp['nodes'][j][0]) ] = type2id(temp['nodes'][j][1])
    
    for j in range(len(temp['edges'])):
        a = node[tuple(temp['edges'][j][0])]
        b = node[tuple(temp['edges'][j][1])]
        
        if(a[0]>b[0]):
            c = a
            a = b
            b = c
        
        try:
            data[(a,b)][type2id(temp['edges'][j][2])] += 1
        except:
            if((a,b) in data):
                data[(a,b)][type2id(temp['edges'][j][2])] = 1
            else:
                data[(a,b)] = {}
                data[(a,b)][type2id(temp['edges'][j][2])] = 1

for name in data:
    m = None
    for t in data[name]:
        if(m==None):
            m = t
        else:
            if(data[name][t]>data[name][m]):
                m =t
    data[name] = m

file_name = sys.argv[2]
ans = {'output':[],'predict':[]}
for i,line in enumerate(open(file_name)):
    temp = json.loads(line)
    node = {}
    for j in range(len(temp['nodes'])):
        node[ tuple(temp['nodes'][j][0]) ] = type2id(temp['nodes'][j][1])
    
    for j in range(len(temp['edges'])):
        a = node[tuple(temp['edges'][j][0])]
        b = node[tuple(temp['edges'][j][1])]
        
        if(a[0]>b[0]):
            c = a
            a = b
            b = c
            
        ans['output'].append(type2id(temp['edges'][j][2]))
        ans['predict'].append(data[(a,b)])

print('F1 macro:{0}'.format( f1_score(ans['output'], ans['predict'], average='macro') ))
print('F1 micro:{0}'.format( f1_score(ans['output'], ans['predict'], average='micro') ))
print('precision macro:{0}'.format( precision_score(ans['output'], ans['predict'], average='macro') ))
print('recall macro:{0}'.format( recall_score(ans['output'], ans['predict'], average='macro') ))