import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
import math
from utils import *
from torch.nn import Parameter

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=math.ceil((kernel_size - 1) / 2), bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=math.ceil((kernel_size - 1) / 2), bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.downsample = downsample


    def forward(self, x):
        residual = x  # 残差
        # 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        # 2
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual  # 相加
        out = self.relu(out)
        return out

class ECABasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None):
        super(ECABasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=math.ceil((kernel_size - 1) / 2), bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=1, padding=math.ceil((kernel_size - 1) / 2), bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.downsample = downsample
        self.eca = ECA(planes)


    def forward(self, x):
        residual = x  # 残差
        # 1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        # 2
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual  # 相加
        out = self.relu(out)
        return out

class ResNet1d(nn.Module):
    def __init__(self, block, layers, k=3, input_channels=12, inplanes=64, num_classes=9, adj_file=None, inp=None, t=None):
        super(ResNet1d, self).__init__()  #
        self.inplanes = inplanes  # 卷积产生的通道数
        # 一维卷积                 输入通道数        输出通道数        卷积核大小       步幅       填充     向输出添加可学习偏差
        # 卷积后维度 (n - k + 2 * p ) / s + 1  （15000 - 15 + 2*7）// 2  + 1 = 7500
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)  # 64
        self.bn1 = nn.BatchNorm1d(inplanes)  # 归一化  1d
        self.relu = nn.ReLU()  # relu
        
##        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 最大池化  （7500 - 3 + 2*1）// 2 + 1 = 3750

        self.layer = self.mslayer(block, layers, k)  # 3
        
        # 对于任何输入大小的输入，可以将输出尺寸指定为1，但是输入和输出特征的数目不会变化。 1 * 512 * x(数据长度)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化  1d  1维 1 * 512 * 1
##        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Linear(64, num_classes)
        
    def mslayer(self, block, layers, ks):
        self.inplanes = 64
        layer1 = self._make_layer(block, 64, layers[0], ks)  # 3
        layer2 = self._make_layer(block, 64, layers[1], ks, stride=2)  # 4
        layer3 = self._make_layer(block, 64, layers[2], ks, stride=2)  # 6
        layer4 = self._make_layer(block, 64, layers[3], ks, stride=2)  # 3
        return nn.Sequential(layer1, layer2, layer3, layer4)
    
    def _make_layer(self, block, planes, blocks, ks, stride=1):
        downsample = None  # 下采样  使得残差与输出大小一致
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, ks, stride, downsample))  # 1
        self.inplanes = planes * block.expansion  # 改变输入通道数
        for _ in range(1, blocks):  # -1
            layers.append(block(self.inplanes, planes, ks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        ms3 = self.layer(x)
        x = self.adaptiveavgpool(ms3)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        
        return x

class ResNet1d_ms(nn.Module):
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=9, adj_file=None, inp=None, t=None):
        super(ResNet1d_ms, self).__init__()  #
        self.inplanes = inplanes  # 卷积产生的通道数
        # 一维卷积                 输入通道数        输出通道数        卷积核大小       步幅       填充     向输出添加可学习偏差
        # 卷积后维度 (n - k + 2 * p ) / s + 1  （15000 - 15 + 2*7）// 2  + 1 = 7500
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)  # 64
        self.bn1 = nn.BatchNorm1d(inplanes)  # 归一化 1d
        self.relu = nn.ReLU()  # relu
##        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 最大池化  （7500 - 3 + 2*1）// 2 + 1 = 3750

        self.layer3 = self.mslayer(block, layers, 3)  # 3
        self.layer5 = self.mslayer(block, layers, 5)  # 5
        self.layer7 = self.mslayer(block, layers, 7)  # 7
        
        # 对于任何输入大小的输入，可以将输出尺寸指定为1，但是输入和输出特征的数目不会变化。 1 * 512 * x(数据长度)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化  1d  1维 1 * 512 * 1
        self.fc = nn.Linear(64*3, num_classes)
                
    def mslayer(self, block, layers, ks):
        self.inplanes = 64
        layer1 = self._make_layer(block, 64, layers[0], ks)  # 3
        layer2 = self._make_layer(block, 64, layers[1], ks, stride=2)  # 4
        layer3 = self._make_layer(block, 64, layers[2], ks, stride=2)  # 6
        layer4 = self._make_layer(block, 64, layers[3], ks, stride=2)  # 3
        return nn.Sequential(layer1, layer2, layer3, layer4)
    
    def _make_layer(self, block, planes, blocks, ks, stride=1):
        downsample = None  # 下采样  使得残差与输出大小一致
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, ks, stride, downsample))  # 1
        self.inplanes = planes * block.expansion  # 改变输入通道数
        for _ in range(1, blocks):  # -1
            layers.append(block(self.inplanes, planes, ks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
##        x = self.maxpool(x)

        ms3 = self.layer3(x)
        ms5 = self.layer5(x)
        ms7 = self.layer7(x)
        
        ms3 = self.adaptiveavgpool(ms3)
        ms5 = self.adaptiveavgpool(ms5)
        ms7 = self.adaptiveavgpool(ms7)
        
        x = torch.cat((ms3, ms5, ms7), dim=1)
        x = x.view(x.size(0),-1)
        
        x = self.fc(x)
        
        return x

class ResNet1d_gcn2(nn.Module):
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=9, adj_file=None, inp=None, t=None):
        super(ResNet1d_gcn2, self).__init__()  #
        self.inplanes = inplanes  # 卷积产生的通道数
        # 一维卷积                 输入通道数        输出通道数        卷积核大小       步幅       填充     向输出添加可学习偏差
        # 卷积后维度 (n - k + 2 * p ) / s + 1  （15000 - 15 + 2*7）// 2  + 1 = 7500
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)  # 64
        self.bn1 = nn.BatchNorm1d(inplanes)  # 归一化  1d
        self.relu = nn.ReLU()  # relu
##        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 最大池化  （7500 - 3 + 2*1）// 2 + 1 = 3750

        self.layer3 = self.mslayer(block, layers, 3)  # 3
        self.layer5 = self.mslayer(block, layers, 5)  # 5
        self.layer7 = self.mslayer(block, layers, 7)  # 7
        
        # 对于任何输入大小的输入，可以将输出尺寸指定为1，但是输入和输出特征的数目不会变化。 1 * 512 * x(数据长度)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化  1d  1维 1 * 512 * 1
        self.fc = nn.Linear(64 * block.expansion*3, num_classes)

        self.inp = torch.from_numpy(inp).float().to(torch.device("cuda:0"))
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float()).to(torch.device("cuda:0"))
        
        self.gc1 = GraphConvolution(inp.shape[1], 512)
        self.gc2 = GraphConvolution(512, 64*3)
        self.leakyrelu = nn.LeakyReLU(0.3)
        
        
    def mslayer(self, block, layers, ks):
        self.inplanes = 64
        layer1 = self._make_layer(block, 64, layers[0], ks)  # 3
        layer2 = self._make_layer(block, 64, layers[1], ks, stride=2)  # 4
        layer3 = self._make_layer(block, 64, layers[2], ks, stride=2)  # 6
        layer4 = self._make_layer(block, 64, layers[3], ks, stride=2)  # 3
        return nn.Sequential(layer1, layer2, layer3, layer4)
    
    def _make_layer(self, block, planes, blocks, ks, stride=1):
        downsample = None  # 下采样  使得残差与输出大小一致
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, ks, stride, downsample))  # 1
        self.inplanes = planes * block.expansion  # 改变输入通道数
        for _ in range(1, blocks):  # -1
            layers.append(block(self.inplanes, planes, ks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
##        x = self.maxpool(x)
        
        ms3 = self.layer3(x)
        ms5 = self.layer5(x)
        ms7 = self.layer7(x)

        ms3 = self.adaptiveavgpool(ms3)
        ms5 = self.adaptiveavgpool(ms5)
        ms7 = self.adaptiveavgpool(ms7)
        x = torch.cat((ms3, ms5, ms7), dim=1)
        x = x.view(x.size(0),-1)
        out1 = self.fc(x)
        
        adj = gen_adj(self.A).detach()
        gc = self.gc1(self.inp, adj)
        gc = self.leakyrelu(gc)
        gc = self.gc2(gc, adj)
        gc = gc.transpose(1, 0)
        out2 = torch.matmul(x,gc)
        
        return out1, out2

class ResNet1d_gcnone(nn.Module):
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=9, adj_file=None, inp=None, t=None):
        super(ResNet1d_gcnone, self).__init__()  #
        self.inplanes = inplanes  # 卷积产生的通道数
        # 一维卷积                 输入通道数        输出通道数        卷积核大小       步幅       填充     向输出添加可学习偏差
        # 卷积后维度 (n - k + 2 * p ) / s + 1  （15000 - 15 + 2*7）// 2  + 1 = 7500
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)  # 64
        self.bn1 = nn.BatchNorm1d(inplanes)  # 归一化  1d
        self.relu = nn.ReLU()  # relu
##        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 最大池化  （7500 - 3 + 2*1）// 2 + 1 = 3750

        self.layer3 = self.mslayer(block, layers, 3)  # 3
        self.layer5 = self.mslayer(block, layers, 5)  # 5
        self.layer7 = self.mslayer(block, layers, 7)  # 7
        
        # 对于任何输入大小的输入，可以将输出尺寸指定为1，但是输入和输出特征的数目不会变化。 1 * 512 * x(数据长度)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化  1d  1维 1 * 512 * 1
        self.fc = nn.Linear(64 * block.expansion*3, num_classes)

        self.inp = torch.from_numpy(inp).float().to(torch.device("cuda:0"))
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float()).to(torch.device("cuda:0"))
        
        self.gc1 = GraphConvolution(inp.shape[1], 64*3)
##        self.gc2 = GraphConvolution(1024, 64*3)
##        self.leakyrelu = nn.LeakyReLU(0.3)
        
    def mslayer(self, block, layers, ks):
        self.inplanes = 64
        layer1 = self._make_layer(block, 64, layers[0], ks)  # 3
        layer2 = self._make_layer(block, 64, layers[1], ks, stride=2)  # 4
        layer3 = self._make_layer(block, 64, layers[2], ks, stride=2)  # 6
        layer4 = self._make_layer(block, 64, layers[3], ks, stride=2)  # 3
        return nn.Sequential(layer1, layer2, layer3, layer4)
    
    def _make_layer(self, block, planes, blocks, ks, stride=1):
        downsample = None  # 下采样  使得残差与输出大小一致
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, ks, stride, downsample))  # 1
        self.inplanes = planes * block.expansion  # 改变输入通道数
        for _ in range(1, blocks):  # -1
            layers.append(block(self.inplanes, planes, ks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

##        x = self.maxpool(x)
        
        ms3 = self.layer3(x)
        ms5 = self.layer5(x)
        ms7 = self.layer7(x)

        ms3 = self.adaptiveavgpool(ms3)
        ms5 = self.adaptiveavgpool(ms5)
        ms7 = self.adaptiveavgpool(ms7)
        x = torch.cat((ms3, ms5, ms7), dim=1)
        x = x.view(x.size(0),-1)
        out1 = self.fc(x)
        
        adj = gen_adj(self.A).detach()
        gc = self.gc1(self.inp, adj)
        gc = gc.transpose(1, 0)
        out2 = torch.matmul(x,gc)
        
        return out1, out2

class ResNet1d_gcnthree(nn.Module):
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=9, adj_file=None, inp=None, t=None):
        super(ResNet1d_gcnthree, self).__init__()  #
        self.inplanes = inplanes  # 卷积产生的通道数
        # 一维卷积                 输入通道数        输出通道数        卷积核大小       步幅       填充     向输出添加可学习偏差
        # 卷积后维度 (n - k + 2 * p ) / s + 1  （15000 - 15 + 2*7）// 2  + 1 = 7500
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)  # 64
        self.bn1 = nn.BatchNorm1d(inplanes)  # 归一化  1d
        self.relu = nn.ReLU()  # relu
##        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # 最大池化  （7500 - 3 + 2*1）// 2 + 1 = 3750

        self.layer3 = self.mslayer(block, layers, 3)  # 3
        self.layer5 = self.mslayer(block, layers, 5)  # 5
        self.layer7 = self.mslayer(block, layers, 7)  # 7
        
        # 对于任何输入大小的输入，可以将输出尺寸指定为1，但是输入和输出特征的数目不会变化。 1 * 512 * x(数据长度)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)  # 自适应平均池化  1d  1维 1 * 512 * 1
        self.fc = nn.Linear(64 * block.expansion*3, num_classes)

        self.inp = torch.from_numpy(inp).float().to(torch.device("cuda:0"))
        _adj = gen_A(num_classes, t, adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float()).to(torch.device("cuda:0"))
        
        self.gc1 = GraphConvolution(inp.shape[1], 512)
        self.gc2 = GraphConvolution(512, 512)
        self.gc3 = GraphConvolution(512, 64*3)
        self.leakyrelu = nn.LeakyReLU(0.3)
        
    def mslayer(self, block, layers, ks):
        self.inplanes = 64
        layer1 = self._make_layer(block, 64, layers[0], ks)  # 3
        layer2 = self._make_layer(block, 64, layers[1], ks, stride=2)  # 4
        layer3 = self._make_layer(block, 64, layers[2], ks, stride=2)  # 6
        layer4 = self._make_layer(block, 64, layers[3], ks, stride=2)  # 3
        return nn.Sequential(layer1, layer2, layer3, layer4)
    
    def _make_layer(self, block, planes, blocks, ks, stride=1):
        downsample = None  # 下采样  使得残差与输出大小一致
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, ks, stride, downsample))  # 1
        self.inplanes = planes * block.expansion  # 改变输入通道数
        for _ in range(1, blocks):  # -1
            layers.append(block(self.inplanes, planes, ks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

##        x = self.maxpool(x)
        
        ms3 = self.layer3(x)
        ms5 = self.layer5(x)
        ms7 = self.layer7(x)

        ms3 = self.adaptiveavgpool(ms3)
        ms5 = self.adaptiveavgpool(ms5)
        ms7 = self.adaptiveavgpool(ms7)
        x = torch.cat((ms3, ms5, ms7), dim=1)
        x = x.view(x.size(0),-1)
        out1 = self.fc(x)
        
        adj = gen_adj(self.A).detach()
        gc = self.gc1(self.inp, adj)
        gc = self.leakyrelu(gc)
        gc = self.gc2(gc, adj)
        gc = self.leakyrelu(gc)
        gc = self.gc3(gc, adj)
        gc = gc.transpose(1, 0)
        out2 = torch.matmul(x,gc)
        
        return out1, out2

def eca3_resnet(**kwargs):
    model = ResNet1d(ECABasicBlock1d, [1, 1, 1, 1], 3, **kwargs)
    return model

def eca5_resnet(**kwargs):
    model = ResNet1d(ECABasicBlock1d, [1, 1, 1, 1], 5, **kwargs)
    return model

def eca7_resnet(**kwargs):
    model = ResNet1d(ECABasicBlock1d, [1, 1, 1, 1], 7, **kwargs)
    return model

def ms_eca_resnet(**kwargs):
    model = ResNet1d_ms(ECABasicBlock1d, [1, 1, 1, 1], **kwargs)
    return model

def gcn2_ms_eca_resnet(**kwargs):
    model = ResNet1d_gcn2(ECABasicBlock1d, [1, 1, 1, 1], **kwargs)
    return model

def gcnone_ms_eca_resnet(**kwargs):
    model = ResNet1d_gcnone(ECABasicBlock1d, [1, 1, 1, 1], **kwargs)
    return model

def gcnthree_ms_eca_resnet(**kwargs):
    model = ResNet1d_gcnthree(ECABasicBlock1d, [1, 1, 1, 1], **kwargs)
    return model

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
##        self.in_features = in_features  # 输入特征
##        self.out_features = out_features  # 输出特征
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))  # 权重
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class ECA(nn.Module):  # Efficient Channel Attention block
    def __init__(self, channels, gamma=2, b=1):
        super(ECA, self).__init__()
        self.kernel_size = int(abs((math.log(channels, 2) + b) / gamma))  # 计算1d卷积核尺寸
        self.kernel_size = self.kernel_size if self.kernel_size % 2 else self.kernel_size + 1  # 计算1d卷积核尺寸
##        self.kernel_size = 3

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # avgpool
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, stride=1, padding=math.ceil((self.kernel_size - 1) / 2),
                              bias=False)  # 1dconv
        self.sigmoid = nn.Sigmoid()  # sigmoid，将输出压缩到(0,1)

    def forward(self, x):
        weights = self.avg_pool(x)
        weights = self.conv(weights.transpose(-1, -2)).transpose(-1, -2)
        weights = self.sigmoid(weights)
        return x * weights.expand_as(x)  # 将计算得到的weights与输入的feature map相乘

    
    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data = torch.Tensor(1, 12, 72000)
# print(data)
# c = crnn()
# c(data)
# summary函数可以查看网络每层的输出的shape信息和参数信息
# summary(resnet34(), input_size=(1, 12, 1000))
