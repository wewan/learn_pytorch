##   ECUST   ##

##author:veritas xu
##time:2018/3/7

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable
# from tensorboardX import SummaryWriter
#########################################
##      与自带RNN输入输出维度相同       ####
##    seq_len(T)  batch   feature   #####
#########################################

class computeRNN(nn.Module):
    def __init__(self,in_feature,hidden_size,n_class):
        super(computeRNN, self).__init__()
        self.in_feature=in_feature
        self.hidden_size=hidden_size
        self.n_class=n_class
        self.in2hidden=nn.Linear(in_feature+self.hidden_size,self.hidden_size)
        self.hidden2out=nn.Linear(self.hidden_size,self.n_class)
        self.tanh=nn.Tanh()
        self.softmax=nn.Softmax(dim=1)

    ##此处input的尺寸为[seq_len,batch,in_feature]
    def forward(self,input,pre_state):
        T=input.shape[0]
        batch=input.shape[1]
        a=Variable(torch.zeros(T,batch,self.hidden_size))             #a-> [T,hidden_size]
        o=Variable(torch.zeros(T,batch,self.n_class))                 #o ->[T,n_class]
        predict_y=Variable(torch.zeros(T,batch,self.n_class))
        # pre_state = Variable(torch.zeros(batch, self.hidden_size))  # pre_state=[batch,hidden_size]


        if pre_state is None:
            pre_state = Variable(torch.zeros(batch, self.hidden_size))  # hidden ->[batch,hidden_size]

        for t in range(T):
            # input:[T,batch,in_feature]
            tmp = torch.cat((input[t], pre_state), 1)  #  [batch,in_feature]+[batch,hidden_size]-> [batch,hidden_size+in_featue]
            a[t]=self.in2hidden(tmp)                      #  [batch,hidden_size+in_feature]*[hidden_size+in_feature,hidden_size] ->[batch,hidden_size]
            hidden = self.tanh(a[t])

            #这里不赋值的话就没有代表隐层向前传递
            pre_state=hidden

            o[t] = self.hidden2out(hidden)  # [batch,hidden_size]*[hidden_size,n_class]->[batch,n_class]
            #由于此次是一个单分类问题，因此不用softmax函数
            if self.n_class ==1:
                predict_y[t]=F.sigmoid(o[t])
            else:
                predict_y[t] = self.softmax(o[t])


        return predict_y, hidden






class outtwo():
    def __init__(self):
        self.a = [1,2,3,4,5,6,7]

    def __len__(self):
        return len(self.a)

    def __getitem__(self, item):
        return self.a[item],self.a[item]

    # def __call__(self, *args, **kwargs):
    #     return self.__getitem__(*args, **kwargs)



a = outtwo()
print(a[1])
m = lambda x: torch.tensor(x)
print([m(a[item]) for item in [1,2,3]])
b =torch.stack([m(a[item]) for item in [1,2,3]])
print(a[1])

