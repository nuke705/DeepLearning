
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
from CS598_HW7_1b_model import CS598_HW7_1b_model as BOW_model


# In[2]:


glove_embeddings = np.load('./preprocessed_data/glove_embeddings.npy')
vocab_size = 100000
#100000
x_train = []
with io.open('./preprocessed_data/imdb_train_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0
    line = line[line!=0]

    line = np.mean(glove_embeddings[line],axis=0)

    x_train.append(line)
x_train = np.asarray(x_train)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('./preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0
    line = line[line!=0]
    
    line = np.mean(glove_embeddings[line],axis=0)

    x_test.append(line)
x_test = np.asarray(x_test)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = BOW_model(300) # try 300 as well

model.cuda()


# In[3]:


# opt = 'sgd'
# LR = 0.01
opt = 'adam'
LR = 0.001
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

batch_size = 200
no_of_epochs = 15
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = []


# In[4]:


for epoch in range(no_of_epochs):

    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()
    
    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input = x_train[I_permutation[i:i+batch_size]]
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data,target)
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

        x_input = x_train[I_permutation[i:i+batch_size]]
        y_input = y_train[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(data,target)
        
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

    print("  test: ", "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss)


torch.save(model,'BOW1b.model')
data = [train_loss,train_accu,test_accu]
data = np.asarray(data)
np.save('data1b.npy',data)


# opt = 'adam'  
# LR = 0.001  
# hiddensize 300  
# 0 87.34 0.3016 0.7081
#    82.91 0.3805   
# 1 87.54 0.2951 0.6288
#    88.98 0.2739  
# 2 87.97 0.2912 0.6359
#    88.14 0.2855  
# 3 87.94 0.2911 0.6291
#    89.12 0.2676  
# 4 88.38 0.2843 0.6463
#    80.69 0.4170  
# 5 88.52 0.2798 0.6232
#    89.60 0.2576  
# 6 88.62 0.2780 0.6815
#    89.76 0.2551  
# 7 88.51 0.2770 0.6077
#    88.07 0.2758  
# 8 88.80 0.2737 0.6415
#    88.85 0.2688  
# 9 89.12 0.2671 0.6122
#    82.76 0.3764  

# hidden = 300  
# 0 82.15 0.3987 1.3100
#   test:  85.65 0.3436  
# 1 85.70 0.3355 0.6297
#   test:  85.70 0.3330  
# 2 86.42 0.3211 0.5989
#   test:  86.38 0.3219
# 3 86.78 0.3132 0.6475
#   test:  84.90 0.3424
# 4 86.91 0.3109 0.6447
#   test:  87.78 0.2931
# 5 86.98 0.3060 0.6393
#   test:  87.99 0.2884
# 6 87.48 0.3022 0.6181
#   test:  88.42 0.2807
# 7 87.46 0.2961 0.6388
#   test:  85.20 0.3369
# 8 87.72 0.2933 0.6122
#   test:  86.62 0.3084
# 9 88.26 0.2867 0.6067
#   test:  88.66 0.2743
# 10 88.15 0.2845 0.6153
#   test:  87.06 0.3008
# 11 88.44 0.2805 0.6125
#   test:  88.01 0.2837
# 12 88.56 0.2788 0.6074
#   test:  89.26 0.2612
# 13 88.64 0.2748 0.5928
#   test:  89.34 0.2617
# 14 88.98 0.2728 0.6038
#   test:  88.26 0.2750
# 15 88.89 0.2696 0.6117
#   test:  90.04 0.2489  
# 16 88.97 0.2666 0.6089
#   test:  89.18 0.2595  
# 17 89.49 0.2601 0.6082
#   test:  89.99 0.2445  
# 18 89.30 0.2627 0.6071
#   test:  86.48 0.3044  
# 19 89.43 0.2554 0.6204
#   test:  90.69 0.2328  

# 500 hidden  
# 0 82.96 0.3883 1.4362
#   test:  76.35 0.5018  
# 1 85.80 0.3332 0.6762
#   test:  77.08 0.5168  
# 2 86.52 0.3179 0.6807
#   test:  87.29 0.3002  
# 3 86.88 0.3105 0.6488  
#   test:  82.31 0.3881  
# 4 87.08 0.3063 0.6513
#   test:  87.26 0.3024    
# 5 87.45 0.3022 0.7016
#   test:  87.78 0.2912    
# 6 87.96 0.2935 0.7071
#   test:  81.74 0.3942  
# 7 88.12 0.2887 0.6740
#   test:  86.13 0.3202  
# 8 88.16 0.2857 0.6169
#   test:  87.96 0.2844  
# 9 88.25 0.2815 0.6135   
#   test:  80.61 0.4303  
# 10 88.63 0.2786 0.6025
#   test:  89.77 0.2535  
# 11 88.95 0.2715 0.6264
#   test:  89.58 0.2525  
# 12 88.70 0.2721 0.6213
#   test:  90.41 0.2434  
# 13 89.13 0.2626 0.6139
#   test:  90.42 0.2422  
# 14 89.43 0.2594 0.6101
#   test:  90.02 0.2448  
# 15 89.58 0.2546 0.6136
#   test:  90.54 0.2369  
# 16 89.74 0.2545 0.6208
#   test:  88.80 0.2662  
# 17 89.96 0.2448 0.6280
#   test:  91.52 0.2184  
# 18 89.88 0.2456 0.6090
#   test:  89.49 0.2495  
# 19 90.22 0.2409 0.6104
#   test:  85.99 0.3135  
