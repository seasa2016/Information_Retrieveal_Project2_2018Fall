from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, utils

import numpy as np
import json

import torch
import sys

class itemDataset(Dataset):
    def __init__(self,file_name,mode='train',transform=None):

        if(mode=='test'):
            self.token = {}
            for name in ['nodes','edges','tokens']:
                self.token[name] =  {}
                with open('./data/token/{0}'.format(name)) as f:
                    for i,line in enumerate(f):
                        self.token[name][line.strip()] = i
        elif(mode=='train'):
            self.token = {}
            for name in ['nodes','edges','tokens']:
                self.token[name] =  {}
                self.token[name]['pad'] = 0
                
        self.read_json(file_name)
        
        if(mode=='train'):
            for name in ['nodes','edges','tokens']:
                with open('./data/token/{0}'.format(name),'w') as f:
                    for name in self.token[name]:
                        f.write("{0}\n".format(name))
        self.transform = transform

    def read_json(self,file_name):
        def type2id(data,dtype):
            for name in data:
                try:
                    return self.token[dtype][name]
                except:
                    self.token[dtype][name] = len(self.token[dtype])
                    return self.token[dtype][name]
        
        def word2id(data):
            ans = []
            for word in data:
                word = word.lower()
                try:
                    ans.append(self.token['tokens'][word])
                except:
                    self.token['tokens'][word] = len(self.token['tokens'])
                    ans.append(self.token['tokens'][word])
            return ans

        self.data = []
        self.sent = []
        for i,line in enumerate(open(file_name)):
            temp = json.loads(line)
            for j in range(len(temp['nodes'])):
                temp['nodes'][j] = [temp['nodes'][j][0],type2id(temp['nodes'][j][1],'nodes')]
            for j in range(len(temp['edges'])):
                temp['edges'][j] = [temp['edges'][j][0],temp['edges'][j][1],type2id(temp['edges'][j][2],'edges')]
            
            for j in range(len(temp['edges'])):
                self.data.append( temp['edges'][j] )
                self.data[-1].append(i)
            self.sent.append({'tokens':word2id(temp['tokens']),'nodes':temp['nodes']})
            

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = {}
        sample['edge'] = [self.data[idx][0],self.data[idx][1]]
        sample['sent'] = self.sent[self.data[idx][3]]['tokens']
        sample['nodes'] = self.sent[self.data[idx][3]]['nodes']
        
        sample['label'] = self.data[idx][2]

        if(transforms):
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    def __call__(self,sample):
        sample['sent'] = torch.tensor(sample['sent'],dtype=torch.long)
        sample['sent_len'] = len(sample['sent'])
        sample['label'] = torch.tensor(sample['label'],dtype=torch.long)
        return sample.copy()

def collate_fn(sample):
    data = {}
    
    for name in ['sent_len','label']:
        data[name] = torch.tensor([_[name] for _ in sample],dtype=torch.long)
        
    batch_size,sent_len = len(data['sent_len']),data['sent_len'].max().item()
    
    data['sent'] = torch.stack([ torch.cat([ _['sent'],torch.zeros(sent_len-_['sent_len'],dtype=torch.long) ] ) for _ in sample])
    
    
    data['node'] = torch.zeros(batch_size,sent_len,dtype=torch.long)
    
    for i in range(len(sample)):
        for line in sample[i]['nodes']:
            for num in range(line[0][0],line[0][1]):
                data['node'][i][num] = line[1]
    
    data['edge'] = torch.zeros(batch_size,sent_len,dtype=torch.long) 
    for i in range(len(sample)):
        for j,line in enumerate(sample[i]['edge']):
            for num in range(line[0],line[1]):
                data['edge'][i][num] = j+1
    
    return data

if(__name__ == '__main__'):
	data = itemDataset('./train.json',transform=transforms.Compose([ToTensor()]))

