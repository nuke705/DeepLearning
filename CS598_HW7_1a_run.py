
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io

#from BOW_model import BOW_model
from CS598_HW7_1a_model import CS598_HW7_1a_model as BOW_model


# In[2]:


#imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
# originally = 8000 vocab
vocab_size = 8000

x_train = []
with io.open('./preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1


# In[3]:


x_test = []
with io.open('./preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_test.append(line)
y_test = np.zeros((25000,))
y_test[0:12500] = 1


# In[4]:


vocab_size += 1

model = BOW_model(vocab_size,750)
model.cuda()


# In[5]:


# opt = 'sgd'
# LR = 0.01
opt = 'adam'
LR = 0.001
no_of_epochs = 10
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

batch_size = 200

L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = []


# In[6]:


for epoch in range(no_of_epochs):

    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input = [x_train[j] for j in I_permutation[i:i+batch_size]]
        y_input = np.asarray([y_train[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(x_input,target)
        loss.backward()

        optimizer.step()   # update weights
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

    # ## test
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):

        x_input = [x_test[j] for j in I_permutation[i:i+batch_size]]
        y_input = np.asarray([y_test[j] for j in I_permutation[i:i+batch_size]],dtype=np.int)
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(x_input,target)
        
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    test_accu.append(epoch_acc)

    time2 = time.time()
    time_elapsed = time2 - time1

    print("  ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)

torch.save(model,'BOW.model')
data = [train_loss,train_accu,test_accu]
data = np.asarray(data)
np.save('data.npy',data)


# output 4  
# 20000 vocab  
# opt = 'adam'  
# LR = 0.001  
# no_of_epochs = 6  
# 0 77.58 0.4700 28.1910
#    83.50 0.3759  
# 1 87.96 0.2920 28.1296
#    84.09 0.3731  
# 2 91.78 0.2122 27.7107
#    87.32 0.3121  
# 3 93.88 0.1588 27.7560
#    87.34 0.3337  
# 4 96.03 0.1139 27.6817
#    85.68 0.4069  
# 5 97.16 0.0827 27.7854
#    86.20 0.4345  

# output3  
# opt = 'adam'  
# LR = 0.0001  
# no_of_epochs = 10  
# 0 98.94 0.0384 41.0255
#    86.30 0.5726  
# 1 99.11 0.0346 40.5721
#    86.24 0.5852  
# 2 98.98 0.0344 53.1788
#    86.16 0.5888  
# 3 99.18 0.0309 51.1363
#    86.08 0.6112  
# 4 99.13 0.0310 69.7932
#    86.16 0.6213  
# 5 99.17 0.0297 55.4612
#    86.06 0.6378  
# 6 99.34 0.0264 54.9475
#    86.04 0.6440  
# 7 99.27 0.0272 20.8155
#    86.05 0.6439  
# 8 99.34 0.0256 25.1047
#    85.93 0.6659  
# 9 99.38 0.0239 24.3463
#    85.92 0.6711  

# output 1  
# opt = 'adam'  
# LR = 0.001  
# no_of_epochs = 6  
# 0 ---77.98 0.4638 17.3765
#    84.16 0.3651  
# 1 ---87.56 0.3031 15.5065
#    83.94 0.3613  
# 2 ---90.26 0.2424 15.7788
#    85.74 0.3346  
# 3 ---92.35 0.1964 15.7848
#    86.10 0.3435  
# 4 ---93.42 0.1685 16.2636
#    87.08 0.3363  
# 5 ---94.60 0.1437 15.7680
#    87.46 0.3435  

# output 2  
# opt = 'adam'  
# LR = 0.001  
# no_of_epochs = 10  
# 0 ---77.75 0.4648 86.6321
#    84.33 0.3598  
# 1 ---87.56 0.3012 57.2938
#    86.59 0.3185  
# 2 90.43 0.2377 48.7824
#    86.40 0.3258  
# 3 92.38 0.1963 32.3619
#    86.92 0.3308  
# 4 93.54 0.1687 24.6993
#    86.68 0.3456  
# 5 94.69 0.1410 58.0564
#    87.26 0.3576  
# 6 95.56 0.1207 86.7761
#    86.94 0.3789  
# 7 95.88 0.1078 68.0074
#    86.72 0.4127  
# 8 96.66 0.0931 33.1653
#    86.51 0.4600  
# 9 97.13 0.0799 36.0467
#    85.55 0.5422  
