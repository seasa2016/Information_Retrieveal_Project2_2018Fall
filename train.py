import torch
from data.dataloader import itemDataset,ToTensor,collate_fn
from torch.utils.data import Dataset,DataLoader

import os
import argparse
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import f1_score

from torchvision import transforms, utils
from model.birnn import RNN

def train(args,model,train_data,test_data,criterion,optimizer,device):
    def convert(data,device):
        for name in data:
            if(isinstance(data[name],torch.Tensor)):
                data[name] = data[name].to(device)
        return data

    print("start training")
    for now in range(args.epoch):
        print(now)

        loss_sum = 0
        count = 0
        model.train()
        for i,data in enumerate(train_data):
            #first convert the data into cuda
            data = convert(data,device)
            
            out = model(data['sent'],data['sent_len'],data['node'],data['edge'])
            
            loss = criterion(out,data['label']) 
            _,pred = torch.topk(out,1)
            pred = pred.view(-1)
            count += torch.sum( data['label'] == pred ).item()
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item()

        print('*'*10)
        print('training loss:{0} acc:{1}/{2}'.format(loss_sum,count,len(train_data)*args.batch_size))
        loss_sum = 0
        count = 0
        model.eval()
        
        ans={'label':[],'output':[]}

        for i,data in enumerate(test_data):
            #first convert the data into cuda
            data = convert(data,device)
            
            with torch.no_grad():
                out = model(data['sent'],data['sent_len'],data['node'],data['edge'])
                
                loss = criterion(out,data['label']) 
                _,pred = torch.topk(out,1)
                pred = pred.view(-1)
                count += torch.sum( data['label'] == pred ).item()
                loss_sum += loss.detach().item()

                ans['label'].extend(data['label'].view(-1).cpu().tolist())
                ans['output'].extend(pred.view(-1).cpu().tolist())

        print('testing loss:{0} acc:{1}/{2}'.format(loss_sum,count,len(test_data)*args.batch_size))
        print('F1:{0}'.format( f1_score(ans['label'], ans['output'], average='micro') ))

        torch.save(model.state_dict(), './save_model/step_{0}.pkl'.format(now))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--word_dim', default=100, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--batch_first', default=True, type=bool)
    parser.add_argument('--bidirectional', default=True, type=bool)
    parser.add_argument('--num_layer', default=2, type=int)

    parser.add_argument('--learning_rate', default=0.005, type=float)
    parser.add_argument('--mode', required=True, type=str)

    args = parser.parse_args()

    if(torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("loading data")
    train_data = itemDataset('./data/train.json',mode=args.mode,transform=transforms.Compose([ToTensor()]))
    train_loader = DataLoader(train_data, batch_size=args.batch_size,shuffle=True, num_workers=12,collate_fn=collate_fn)
    
    test_data = itemDataset('./data/test.json',mode=args.mode,transform=transforms.Compose([ToTensor()]))
    test_loader = DataLoader(test_data, batch_size=args.batch_size,shuffle=True, num_workers=12,collate_fn=collate_fn)

    print("setting model")
    model = RNN(train_data.token,args)
    model = model.to(device)
    print(model)
    
    #for name,d in model.named_parameters():
    #    print(name,d.requires_grad)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=args.learning_rate)
    
    criterion = nn.CrossEntropyLoss(reduction='sum')

    train(args,model,train_loader,test_loader,criterion,optimizer,device)
    


if(__name__ == '__main__'):
    main()
