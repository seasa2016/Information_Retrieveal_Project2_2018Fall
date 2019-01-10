import torch
from data.dataloader import itemDataset,ToTensor,collate_fn,collate_fn1
from torch.utils.data import Dataset,DataLoader

import os
import argparse
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score

from torchvision import transforms, utils
from model.birnn import RNN
from model.birnn_co import RNNC

def test(args,model,test_data,device):
    def convert(data,device):
        for name in data:
            if(isinstance(data[name],torch.Tensor)):
                data[name] = data[name].to(device)
        return data

    print("start testing")

    ans = {'output':[],'label':[]}
    for i,data in enumerate(test_data):
        #first convert the data into cuda
        data = convert(data,device)
        
        with torch.no_grad():
            out = model(data['sent'],data['sent_len'],data['node'],data['edge'])
            
            _,pred = torch.topk(out,1)
            pred = pred.view(-1)

            ans['label'].extend(data['label'].view(-1).cpu().tolist())
            ans['output'].extend(pred.view(-1).cpu().tolist())
    print('F1 macro:{0}'.format( f1_score(ans['label'], ans['output'], average='macro') ))
    print('F1 micro:{0}'.format( f1_score(ans['label'], ans['output'], average='micro') ))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--load_from', required=True, type=str)
    parser.add_argument('--model', required=True, type=str)

    args = parser.parse_args()

    checkpoint = torch.load(args.load_from)

    if(torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("loading data")
    test_data = itemDataset('./data/test.json',mode=args.mode,transform=transforms.Compose([ToTensor()]))
    
    if(args.model == 'birnn'):
        test_loader = DataLoader(test_data, batch_size=args.batch_size,shuffle=True, num_workers=12,collate_fn=collate_fn)
    elif(args.model == 'birnn_co'):
        test_loader = DataLoader(test_data, batch_size=args.batch_size,shuffle=True, num_workers=12,collate_fn=collate_fn1)
        
    print("setting model")
    if(args.model == 'birnn'):
        model = RNN(test_data.token,checkpoint['args'])
    elif(args.model == 'birnn_co'):
        model = RNNC(test_data.token,checkpoint['args'])
    
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    print(model)
    
    test(args,model,test_loader,device)
    


if(__name__ == '__main__'):
    main()
