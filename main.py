import argparse
import os
import random
import torch
import torch.nn as nn
from tqdm import tqdm  # 显示进度条的模块
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
from models.mobilenet_v3 import mobilenetv3_small
from models.Mynet import eca3_resnet, eca5_resnet, eca7_resnet, ms_eca_resnet, gcn2_ms_eca_resnet, gcnone_ms_eca_resnet, gcnthree_ms_eca_resnet

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
    # 创建解析器
    parser = argparse.ArgumentParser()
    # 添加参数
    parser.add_argument('--data-dir', type=str, default='data/ptbxl/', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()  # 解析参数

# 训练
def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()  # 启用 BatchNormalization 和 Dropout。 在模型测试阶段使用model.train() 让model变成训练模式，此时 dropout和batch normalization的操作在训练q起到防止网络过拟合的问题。
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)  # 输出 前向传播
        loss = criterion(output, labels)  # 损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10, norm_type=2)
        optimizer.step()  # 优化
        running_loss += loss.item()  # 损失值
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
##    scheduler.step()
    print('Loss: %.4f' % (running_loss/len(dataloader)))

# 验证
def evaluate(dataloader, net, args, criterion, device):
    print('Validating...')
    net.eval()  # 不启用 BatchNormalization 和 Dropout。此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)  # 输出
        loss = criterion(output, labels)  # 损失
        running_loss += loss.item()  # 损失值
        output = torch.sigmoid(output)  # sigmoid
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % (running_loss/len(dataloader)))
    y_trues = np.vstack(labels_list)  # 正确标签
    y_scores = np.vstack(output_list)  # 得分
    AUC, _ = compute_AUC(y_trues, y_scores)
    F1 = compute_F1(y_trues, y_scores)
    print('F1:%.4f, AUC: %.4f' % (F1, AUC))
    # 判断是否变好
    if args.phase == 'train' and F1 >= args.best_metric:
        args.best_metric = F1
        torch.save(net.state_dict(), args.model_path)  # 保存当前最佳模型
    else:
        pass

# 测试
def test(dataloader, net, args, device):
    print('Testing...')
    net.eval()  # 不启用 BatchNormalization 和 Dropout。此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)  # 输出
        output = torch.sigmoid(output)  # sigmoid
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(labels_list)  # 正确标签
    y_scores = np.vstack(output_list)  # 得分
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
    net.train()  # 启用 BatchNormalization 和 Dropout。 在模型测试阶段使用model.train() 让model变成训练模式，此时 dropout和batch normalization的操作在训练q起到防止网络过拟合的问题。
    running_loss1 = 0
    running_loss2 = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output1,output2 = net(data)  # 输出 前向传播
        loss1 = criterion1(output1, labels)  # 损失
        loss2 = criterion2(output2, labels)
        optimizer.zero_grad()  # 梯度清零
        loss = b*loss1 + (1-b)*loss2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10, norm_type=2)
        optimizer.step()  # 优化
        running_loss1 += loss1.item()  # 损失值
        running_loss2 += loss2.item()  # 损失值
        output = b*torch.sigmoid(output1) + (1-b)*torch.sigmoid(output2)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
##    scheduler.step()
    print('Loss1: %.4f' % (running_loss1/len(dataloader)))
    print('Loss2: %.4f' % (running_loss2/len(dataloader)))

def evaluate2(dataloader, net, args, criterion1, criterion2, device, b):
    print('Validating...')
    net.eval()  # 不启用 BatchNormalization 和 Dropout。此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
    running_loss1 = 0
    running_loss2 = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output1,output2 = net(data)  # 输出
        loss1 = criterion1(output1, labels)  # 损失
        loss2 = criterion2(output2, labels)
        running_loss1 += loss1.item()  # 损失值
        running_loss2 += loss2.item()  # 损失值
        output = b*torch.sigmoid(output1) + (1-b)*torch.sigmoid(output2)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss1: %.4f' % (running_loss1/len(dataloader)))
    print('Loss2: %.4f' % (running_loss2/len(dataloader)))
    y_trues = np.vstack(labels_list)  # 正确标签
    y_scores = np.vstack(output_list)  # 得分
    AUC, _ = compute_AUC(y_trues, y_scores)
    F1 = compute_F1(y_trues, y_scores)
    print('F1:%.4f, AUC: %.4f' % (F1, AUC))
    # 判断是否变好
    if args.phase == 'train' and F1 >= args.best_metric:
        args.best_metric = F1
        torch.save(net.state_dict(), args.model_path)  # 保存当前最佳模型
    else:
        pass

def test2(dataloader, net, args, device, b):
    print('Testing...')
    net.eval()  # 不启用 BatchNormalization 和 Dropout。此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output1, output2 = net(data)  # 输出
        output = b*torch.sigmoid(output1) + (1-b)*torch.sigmoid(output2)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(labels_list)  # 正确标签
    y_scores = np.vstack(output_list)  # 得分
    AUC, _ = compute_AUC(y_trues, y_scores)
    TPR = compute_TPR(y_trues, y_scores)
    F1 = compute_F1(y_trues, y_scores)
    ACC = compute_ACC(y_trues, y_scores)
    print('AUC: %.4f, TPR: %.4f, F1: %.4f, ACC: %.4f' % (AUC, TPR, F1, ACC))

    if args.phase == 'test':
        df = pd.DataFrame([[AUC, TPR, F1, ACC]], columns=[ "AUC", "TPR", "F1", "ACCs"])
        df.to_csv(args.result_path)

def choose_model(models, nleads, num_classes, device, adj_file=None, inp=None, t=None):
    if models == 'eca3_resnet':
        return eca3_resnet(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'eca5_resnet':
        return eca5_resnet(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'eca7_resnet':
        return eca7_resnet(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'ms_eca_resnet':
        return ms_eca_resnet(input_channels=nleads, num_classes=num_classes).to(device)
    elif models == 'gcn2_ms_eca_resnet':
        return gcn2_ms_eca_resnet(input_channels=nleads, num_classes=num_classes, adj_file=adj_file, inp=inp, t=t).to(device)
    elif models == 'gcnone_ms_eca_resnet':
        return gcnone_ms_eca_resnet(input_channels=nleads, num_classes=num_classes, adj_file=adj_file, inp=inp, t=t).to(device)
    elif models == 'gcnthree_ms_eca_resnet':
        return gcnthree_ms_eca_resnet(input_channels=nleads, num_classes=num_classes, adj_file=adj_file, inp=inp, t=t).to(device)
    
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
    elif models == 'mobilenetv3':
        return mobilenetv3_small(input_channels=nleads, num_classes=num_classes).to(device)


if __name__ == "__main__":
    
    args = parse_args()
    args.data_dir='data/ptbxl/'
    data_dir = args.data_dir  # 规范化指定路径  data/ptbxl/

    # 是否使用GPU
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    print(device)

    # 选择导联数量
    if args.leads == 'all':
        leads = 'all'  # 所有导联
        nleads = 12  # 导联数量
    else:
        leads = args.leads.split(',')
        nleads = len(leads)

    t = None
    b = None

##    t = 0.2
##    b = 0.4
    
    experiments = [
        ('exp0', 'all'),
        ('exp1', 'diagnostic'),
        ('exp1.1', 'subdiagnostic'),
        ('exp1.1.1', 'superdiagnostic'),
        ('exp2', 'form'),
        ('exp3', 'rhythm')
    ]
    
    models = [
        'fcn_wang',
        'resnet1d_wang',
        'lstm_bidir',
        'xresnet1d101',
        'inceptiontime',
        'mobilenetv3',
        
        'eca3_resnet',
        'eca5_resnet',
        'eca7_resnet',
        'ms_eca_resnet',
##        'gcn2_ms_eca_resnet',
##        'gcnone_ms_eca_resnet',
##        'gcnthree_ms_eca_resnet'
        
    ]
            
    for name, task in experiments:
        task_folder_path = f'results/{task}'
        # 判断文件夹是否存在
        if not os.path.exists(task_folder_path):
            # 文件夹不存在，创建文件夹
            os.makedirs(task_folder_path)

        for model in models:
            print(task, model)
            print('t',t,'b',b)
            set_seed(args)

            model_folder_path = f'results/{task}/{model}'
            # 判断文件夹是否存在
            if not os.path.exists(model_folder_path):
                # 文件夹不存在，创建文件夹
                os.makedirs(model_folder_path)
                
            args.best_metric = 0  # 记录最高标准
            # datasets
            train_loader, val_loader, test_loader, num_classes = load_datasets(datafolder=args.data_dir, experiment=name, bs=args.batch_size)
            print(num_classes)

            adj_file = task+"_adj.pkl"
            inp = np.array([[i] for i in range(num_classes)], dtype=np.float32)
            enc = preprocessing.OneHotEncoder(sparse=False)
            inp = enc.fit_transform(inp)
            inp = [list(i) + [0]*(300-num_classes) for i in inp]
            inp = np.array(inp, dtype=np.float32)
        
            # 模型
            net = choose_model(model, nleads, num_classes, device, adj_file, inp, t)
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)  # adam优化器
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10,], gamma=0.1)  # 每隔10个epoch，将学习率衰减（lr*gamma）
            criterion = nn.BCEWithLogitsLoss()  # 损失函数
            
            # 模型路径
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
            
            # 训练
            args.phase = 'train'
            if args.resume:
                # load_state_dict 加载模型参数  load加载模型
                net.load_state_dict(torch.load(args.model_path, map_location=device))
            else:
                badnum = 0  # 记录没变好次数
                epoch = 1  # epoch
                flag = args.best_metric
                while badnum < 10:  # 停止策略
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

            # 预测
            args.phase = 'test'
            if os.path.exists(args.model_path):
                net = choose_model(model, nleads, num_classes, device, adj_file, inp, t)
                net.load_state_dict(torch.load(args.model_path, map_location=device))  # 加载网络
                net.eval()  # 不启用 BatchNormalization 和 Dropout。此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。

                print('Results on test data:')
                if b == None:
                    test(test_loader, net, args, device)
                else:
                    test2(test_loader, net, args, device, b)
                    pass

                print(args.best_metric)
                
            del net
            torch.cuda.empty_cache()

