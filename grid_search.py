import argparse
import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing
from dataset import load_datasets
from utils import *
from models.fcn_wang import fcn_wang
from models.resnet1d_wang import resnet1d_wang
from models.lstm import lstm_bidir
from models.xresnet1d101 import xresnet1d101
from models.inceptiontime import inceptiontime
from models.Mynet import ecanet_3, ecanet_5, ecanet_7, marnet, mar_gcn1, mar_gcn2, mar_gcn3

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/ptbxl/', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()

def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train() 
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data) 
        loss = criterion(output, labels) 
        optimizer.zero_grad() 
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10, norm_type=2)
        optimizer.step() 
        running_loss += loss.item() 
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
##    scheduler.step()
    print('Loss: %.4f' % (running_loss/len(dataloader)))

def evaluate(dataloader, net, args, criterion, device):
    print('Validating...')
    net.eval() 
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data) 
        loss = criterion(output, labels) 
        running_loss += loss.item() 
        output = torch.sigmoid(output) 
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % (running_loss/len(dataloader)))
    y_trues = np.vstack(labels_list) 
    y_scores = np.vstack(output_list) 
    AUC, _ = compute_AUC(y_trues, y_scores)
    F1 = compute_F1(y_trues, y_scores)
    print('F1:%.4f, AUC: %.4f' % (F1, AUC))
    if args.phase == 'train' and F1 >= args.best_metric:
        args.best_metric = F1
        torch.save(net.state_dict(), args.model_path)  
    else:
        pass

def test(dataloader, net, args, device):
    print('Testing...')
    net.eval() 
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data) 
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(labels_list) 
    y_scores = np.vstack(output_list) 
    AUC, _ = compute_AUC(y_trues, y_scores)
    TPR = compute_TPR(y_trues, y_scores)
    F1 = compute_F1(y_trues, y_scores)
    ACC = compute_ACC(y_trues, y_scores)
    print('AUC: %.4f, TPR: %.4f, F1: %.4f, ACC: %.4f' % (AUC, TPR, F1, ACC))

    if args.phase == 'test':
        df = pd.DataFrame([[AUC, TPR, F1, ACC]], columns=[ "AUC", "TPR", "F1", "ACCs"])
        df.to_csv(args.result_path)

def train2(dataloader, net, args, criterion1, criterion2, epoch, scheduler, optimizer, device, b):
    print('Training epoch %d:' % epoch)
    net.train() 
    running_loss1 = 0
    running_loss2 = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output1,output2 = net(data) 
        loss1 = criterion1(output1, labels) 
        loss2 = criterion2(output2, labels)
        optimizer.zero_grad() 
        loss = b*loss1 + (1-b)*loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10, norm_type=2)
        optimizer.step()  
        running_loss1 += loss1.item() 
        running_loss2 += loss2.item()  
        output = b*torch.sigmoid(output1) + (1-b)*torch.sigmoid(output2)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
##    scheduler.step()
    print('Loss1: %.4f' % (running_loss1/len(dataloader)))
    print('Loss2: %.4f' % (running_loss2/len(dataloader)))

def evaluate2(dataloader, net, args, criterion1, criterion2, device, b):
    print('Validating...')
    net.eval() 
    running_loss1 = 0
    running_loss2 = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output1,output2 = net(data)  
        loss1 = criterion1(output1, labels) 
        loss2 = criterion2(output2, labels)
        running_loss1 += loss1.item() 
        running_loss2 += loss2.item() 
        output = b*torch.sigmoid(output1) + (1-b)*torch.sigmoid(output2)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss1: %.4f' % (running_loss1/len(dataloader)))
    print('Loss2: %.4f' % (running_loss2/len(dataloader)))
    y_trues = np.vstack(labels_list) 
    y_scores = np.vstack(output_list) 
    AUC, _ = compute_AUC(y_trues, y_scores)
    F1 = compute_F1(y_trues, y_scores)
    print('F1:%.4f, AUC: %.4f' % (F1, AUC))
    if args.phase == 'train' and F1 >= args.best_metric:
        args.best_metric = F1
        torch.save(net.state_dict(), args.model_path) 
    else:
        pass

def test2(dataloader, net, args, device, b):
    print('Testing...')
    net.eval() 
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output1, output2 = net(data) 
        output = b*torch.sigmoid(output1) + (1-b)*torch.sigmoid(output2)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(labels_list) 
    y_scores = np.vstack(output_list) 
    AUC, _ = compute_AUC(y_trues, y_scores)
    TPR = compute_TPR(y_trues, y_scores)
    F1 = compute_F1(y_trues, y_scores)
    ACC = compute_ACC(y_trues, y_scores)
    print('AUC: %.4f, TPR: %.4f, F1: %.4f, ACC: %.4f' % (AUC, TPR, F1, ACC))

    if args.phase == 'test':
        df = pd.DataFrame([[AUC, TPR, F1, ACC]], columns=["AUC", "TPR", "F1", "ACCs"])
        df.to_csv(args.result_path)
        
        df = pd.DataFrame([[args.hyperparameter, round(AUC,4), round(TPR,4), round(F1,4), round(ACC,4)]], columns=["hyperparameter", "AUC", "TPR", "F1", "ACCs"])
        df.to_csv(args.results_path, index=False, mode="a", header=False, encoding="utf-8")


def choose_model(models, nleads, num_classes, device, adj_file=None, inp=None, t=None):
    if models == 'ecanet_3':
        return ecanet_3(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'ecanet_5':
        return ecanet_5(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'ecanet_7':
        return ecanet_7(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'marnet':
        return marnet(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'mar_gcn1':
        return mar_gcn1(input_channels=nleads, num_classes=num_classes, adj_file=adj_file, inp=inp, t=t).to(device)
    elif models == 'mar_gcn2':
        return mar_gcn2(input_channels=nleads, num_classes=num_classes, adj_file=adj_file, inp=inp, t=t).to(device)
    elif models == 'mar_gcn3':
        return mar_gcn3(input_channels=nleads, num_classes=num_classes, adj_file=adj_file, inp=inp, t=t).to(device)
    
    elif models == 'fcn_wang':
        return fcn_wang(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'resnet1d_wang':
        return resnet1d_wang(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'lstm_bidir':
        return lstm_bidir(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'xresnet1d101':
        return xresnet1d101(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'inceptiontime':
        return inceptiontime(input_channels=nleads, num_classes=num_classes).to(device)


if __name__ == "__main__":
    
    args = parse_args()
    data_dir = args.data_dir 
    args.data_dir='data/ptbxl/'
    
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    print(device)

    if args.leads == 'all':
        leads = 'all'  
        nleads = 12  
    else:
        leads = args.leads.split(',')
        nleads = len(leads)
    
    experiments = [
        ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
    ]
    
    models = [
##        'fcn_wang',
##        'resnet1d_wang',
##        'lstm_bidir',
##        'xresnet1d101',
##        'inceptiontime',
##        'mobilenetv3',
    
##        'ecanet_3',
##        'ecanet_5',
##        'ecanet_7',
##        'marnet',
##        'mar_gcn1',
        'mar_gcn2',
##        'mar_gcn3'
        
    ]
            
    for name, task in experiments:
        task_folder_path = f'results/{task}'
        if not os.path.exists(task_folder_path):
            os.makedirs(task_folder_path)

        for model in models:
            set_seed(args)
            model_folder_path = f'results/{task}/{model}'
            if not os.path.exists(model_folder_path):
                os.makedirs(model_folder_path)
                
            # datasets
            train_loader, val_loader, test_loader, num_classes = load_datasets(datafolder=args.data_dir, experiment=name, bs=args.batch_size)
            print(num_classes)

            adj_file = task+"_adj.pkl"
            inp = np.array([[i] for i in range(num_classes)], dtype=np.float32)
            enc = preprocessing.OneHotEncoder(sparse=False)
            inp = enc.fit_transform(inp)
            inp = [list(i) + [0]*(300-num_classes) for i in inp]
            inp = np.array(inp, dtype=np.float32)

            for t in [round(i * 0.1, 1) for i in range(0, 11)]:
                for b in [round(i * 0.1, 1) for i in range(0, 11)]:
                    set_seed(args)
                    print(task, model)
                    args.hyperparameter = f't{t}b{b}'
                    print(args.hyperparameter)
                    args.best_metric = 0 
                  
                    net = choose_model(model, nleads, num_classes, device, adj_file, inp, t)
                    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10,], gamma=0.1)  
                    criterion = nn.BCEWithLogitsLoss()  
                    
                    if t == None:
                        args.model_path = f'results/{task}/{model}/{model}.pth'
                        args.result_path = f'results/{task}/{model}/{model}.csv'
                        args.threshold_path = f'results/{task}/{model}/{model}.pkl'
                    else:
                        if b == None:
                            args.model_path = f'results/{task}/{model}/{model}_t{t}.pth'
                            args.result_path = f'results/{task}/{model}/{model}_t{t}.csv'
                            args.threshold_path = f'results/{task}/{model}/{model}_t{t}.pkl'
                        else:
                            args.model_path = f'results/{task}/{model}/{model}_t{t}b{b}.pth'
                            args.result_path = f'results/{task}/{model}/{model}_t{t}b{b}.csv'
                            args.threshold_path = f'results/{task}/{model}/{model}_t{t}b{b}.pkl'
                            args.results_path = f'results/{task}/{model}/{model}.csv'
                            
                            headers = ["hyperparameter", "AUC", "TPR", "F1", "ACCs"]
                            if not os.path.exists(args.results_path):
                                pd.DataFrame(columns=headers).to_csv(args.results_path, index=False, mode="w", encoding="utf-8")
                    
                    args.phase = 'train'
                    if args.resume:
                        net.load_state_dict(torch.load(args.model_path, map_location=device))
                    else:
                        badnum = 0  
                        epoch = 1  
                        flag = args.best_metric
                        while badnum < 10: 
                            if b == None:
                                train(train_loader, net, args, criterion, epoch, scheduler, optimizer, device)
                                evaluate(val_loader, net, args, criterion, device)
                            else:
                                train2(train_loader, net, args, criterion, criterion, epoch, scheduler, optimizer, device, b)
                                evaluate2(val_loader, net, args, criterion, criterion, device, b)
                            
                            if flag == args.best_metric:
                                badnum += 1
                            else:
                                badnum = 0
                                flag = args.best_metric
                            epoch += 1

                    args.phase = 'test'
                    if os.path.exists(args.model_path):
                        net = choose_model(model, nleads, num_classes, device, adj_file, inp, t)
                        net.load_state_dict(torch.load(args.model_path, map_location=device))  
                        net.eval() 

                        print('Results on test data:')
                        if b == None:
                            test(test_loader, net, args, device)
                        else:
                            test2(test_loader, net, args, device, b)

                        print(args.best_metric)
                        
                    del net
                    torch.cuda.empty_cache()

