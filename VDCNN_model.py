
# coding: utf-8

# In[15]:


# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

import os
import pandas as pd
from os.path import exists


# In[16]:


def pad_text(text, pad, fixed_length):
    length = len(text)
    if length < fixed_length:
        return text + [pad]*(fixed_length - length)
    else:
        return text[:fixed_length]

class TextDataset(Dataset):
    
    def __init__(self, texts, dictionary, sort=False, fixed_length=1024):

        PAD_IDX = dictionary.indexer(dictionary.PAD_TOKEN)
        
        self.texts = [([dictionary.indexer(i) for i in list(text.lower())], label) for label, text in texts.values.tolist()]

        if fixed_length:
            self.texts = [(pad_text(text, PAD_IDX, fixed_length), label) for text, label in self.texts]
            
        if sort:
            self.texts = sorted(self.texts, key=lambda x: len(x[0]))
        
    def __getitem__(self, index):
        tokens, label = self.texts[index]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        
    def __len__(self):
        return len(self.texts)

class VDCNNDictionary:

    def __init__(self, args=None):

        self.ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/|_#$%^&*~`+=<>()[]{}" # + space, pad, unknown token
        self.PAD_TOKEN = '<PAD>'
        
        self.build_dictionary()

    def build_dictionary(self):

        self.vocab_chars = [self.PAD_TOKEN, '<UNK>', '<SPACE>'] + list(self.ALPHABET)
        self.char2idx = {char:idx for idx, char in enumerate(self.vocab_chars)}
        self.vocabulary_size = len(self.vocab_chars)

    def indexer(self, char):
        if char.strip() == '':
            char = '<SPACE>'
        try:
            return self.char2idx[char]
        except:
            char = '<UNK>'
            return self.char2idx[char]
            
    


# In[17]:


#yelp full review
# ypf1 = './datasets/yelp_review_full_csv/train.csv'
# ypf2 = './datasets/yelp_review_full_csv/test.csv'

ypf1 = 'E:/study/CS598/project/yelp_review_full_csv/train.csv'
ypf2 = 'E:/study/CS598/project/yelp_review_full_csv/test.csv'
#ag news
ag1 = 'E:/study/CS598/project/ag_news_csv/train.csv'
ag2 = 'E:/study/CS598/project/ag_news_csv/test.csv'
train = pd.read_csv(ypf1, header=None)
#train = pd.read_csv('E:/study/CS598/project/yelp_review_full_csv/train.csv', header=None)
train.columns = ['star', 'review']
#train.columns = ['star', 'title','review']

#test = pd.read_csv('E:/study/CS598/project/yelp_review_full_csv/test.csv', header=None)
test = pd.read_csv(ypf2, header=None)
test.columns = ['star', 'review']
#train.columns = ['star', 'title','review']
        
dictionary = VDCNNDictionary()

trainset = TextDataset(train, dictionary, False, 1024)
testset = TextDataset(test, dictionary, False, 1024)
#train


# In[18]:


trainloader = torch.utils.data.DataLoader(trainset, batch_size= 4,
                                          shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)


# In[19]:


#embed = nn.Embedding(dataset.__len__(),16)
#lookup_tensor = torch.tensor(dataset.__getitem__(0)[0],dtype = torch.long)
#trainset = embed(lookup_tensor)

#embed_test = nn.Embedding(testset.__len__(),16)
#lookup_tensor_test = torch.tensor(testset.__getitem__(0)[0],dtype = torch.long)
#testset = embed(lookup_tensor_test)
# trainset = dataset
# testset = TextDataset(test, dictionary, False, 1024)
#trainset


# Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#trainset.__getitem__(0)
#bigtensor


# In[21]:



def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)


class KMaxPool(nn.Module):

    def __init__(self, k = None):
        super().__init__()
        self.k = k

    def forward(self, x):
        if self.k is None:
            time_steps = x.shape[2]
            self.k = time_steps//2
        kmax, kargmax = x.topk(self.k, dim=2)
        #kmax, kargmax = x.topk(self.k)
        return kmax

    
def downsampling(pool_type, in_channel,out_channel):
    if pool_type == '1':
        pool = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_channel))
    elif pool_type == '2':
        pool = KMaxPool()
    elif pool_type == '3':
        pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    else:
        pool = None
    return pool
    
#initial length
#s = 16
nclasses = 5
vocabulary_size = 1024
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels,kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        
        
    def forward(self, x):
        #residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        #out += residual
        out = self.relu2(out)
        
        return out


class VDCNN(nn.Module):
    def __init__(self):
        
        super(VDCNN, self).__init__()                                    
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=16, padding_idx=0)
        
        self.conv1 = nn.Conv1d(16, 64, kernel_size=3, stride=1, padding=1)
        #between blocks
        self.downsample_1 = downsampling('1',64,128)
        self.downsample_2 = downsampling('1',128,256)
        self.downsample_3 = downsampling('1',256,512)

        self.block1 = nn.Sequential(
            conv_block(64,64,1),
            conv_block(64,64,1))
        
        self.block2 = nn.Sequential(
            conv_block(128,128,1),
            conv_block(128,128,1))
        
        self.block3 = nn.Sequential(
            conv_block(256,256,1),
            conv_block(256,256,1))
        
        self.block4 = nn.Sequential(
            conv_block(512,512,1),
            conv_block(512,512,1))
        
        self.kmaxpool = KMaxPool(8)
        self.linear1 = nn.Sequential(
            nn.Linear(4096,2048) ,nn.ReLU())
        
        self.linear2 = nn.Sequential(
            nn.Linear(2048,2048) ,nn.ReLU())
        
        self.linear3 = nn.Linear(2048,nclasses)
        
    def forward(self, x):
        #print('input',x.size())
        x = self.embedding(x)
        #print('embed',x.size())
        x = x.transpose(1,2)
        x = x.transpose(0,2)
        
        x = self.conv1(x)
        #print('after conv1',x.size())
        x = self.block1(x)
        
        x = self.downsample_1(x)
        #print('1st pool/2',x.size())
        #print(x.size())
        x = self.block2(x)
        #print(x.size())
        x = self.downsample_2(x)
        
        #print(x.size())
        x = self.block3(x)
        #print(x.size())
        x = self.downsample_3(x)
        
        #print(x.size())
        x = self.block4(x)
        #print(x.size())
        x = self.kmaxpool(x)
        #x = kmax_pooling(x, dim=2, k=8)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        #print(x.size())
        return x


net = VDCNN()
net.to(device)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9)


# In[ ]:


# Train

net.train()
for epoch in range(20):  # loop over the dataset multiple times

    net.train()
    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        # get the inputs
        inputs,labels = data
        
        inputs = torch.transpose(inputs, 0, 1).to(device)
        #inputs = inputs.to(device)
        
        labels = torch.tensor(labels).to(device)
        #print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        #torch.cuda.synchronize()
        loss.backward()

        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

        optimizer.step()

        #print statistics
        running_loss += loss.item()
        if i % 1000 == 999 and (i != 1):    # print every 
            print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, i + 1, running_loss / 1000))
            running_loss = 0.0

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = torch.transpose(inputs, 0, 1).to(device)
       
            labels = torch.tensor(labels).to(device)
            

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Epoch',epoch,'Accuracy : %f %%' % (
        100 * correct / total))

# model names: dataset _ downsampleing option _ layers
torch.save(net,'ypf_1_9.model')
print('Finished Training')


# In[ ]:


# # Test

# net.eval()
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)

#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Test Set Accuracy: %f %%' % (
#     100 * correct / total))


# In[ ]:


3//2


# In[9]:


9999%5000 

