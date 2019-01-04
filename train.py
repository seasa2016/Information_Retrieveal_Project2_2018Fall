import torch
from data.dataloader import itemDataset,ToTensor,collate_fn
from torch.utils.data import Dataset,DataLoader

import os
import argparse
import torch.optim as optim
import torch.nn as nn

from torchvision import transforms, utils
from model.birnn import RNN

batch_size = 128
check = 10
epoch = 500
num_workers=16

def train(model,train_data,criterion,optimizer,device):
    def convert(data,device):
        for name in data:
            if(isinstance(data[name],torch.Tensor)):
                data[name] = data[name].to(device)
        return data

    print("start training")
    count = 0
    for now in range(epoch):
        print(now)

        loss_sum = 0
        model.train()
        for i,data in enumerate(train_data):
            #first convert the data into cuda
            data = convert(data,device)
            
            out = model(data['sent'],data['sent_len'],data['node'],data['edge'])
            
            loss = criterion(out,data['label']) 
            _,pred = torch.topk(out,1)
            pred = pred.view(-1)
            count += torch.sum( data['label'] == pred )
            
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item()

        if(now%check==0):
            print('*'*10)
            print('training loss:{0} acc:{1}/{2}'.format(loss_sum,count.data,25946))
            count = 0
        torch.save(model.state_dict(), './save_model/last.pkl')

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--dropout', default=0, type=float)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--gpu', default=0, type=int)

    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--batch_first', default=True, type=bool)
    parser.add_argument('--bidirectional', default=True, type=bool)
    parser.add_argument('--num_layer', default=2, type=int)

    parser.add_argument('--learning_rate', default=0.005, type=float)

    args = parser.parse_args()

    if(torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print("loading data")
    train_data = itemDataset('./data/train.json',transform=transforms.Compose([ToTensor()]))
    train_loader = DataLoader(train_data, batch_size=args.batch_size,shuffle=True, num_workers=1,collate_fn=collate_fn)

    print("setting model")
    model = RNN(train_data.token,args)
    model = model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(reduction='sum')

    train(model,train_loader,criterion,optimizer,device)
    


if(__name__ == '__main__'):
    main()
