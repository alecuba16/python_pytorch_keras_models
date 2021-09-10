# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

#Configs
train_file = '../datasets/Escamb_wtdata_train_alarm_86400.csv.gz'
exclude_columns = ['alarm_block_code', 'alarm_all', 'alarm_all_block_code', 'ot', 'ot_block_code', 'ot_all', 'ot_all_block_code']
include_columns = ['VelViento_avg','Pot_avg','VelRotor_avg','TempAceiteMultip_avg','TempAmb_avg','TempRodamMultip_avg'] #Escamb multi
target_name = 'alarm'
datetime_name = 'date_time'
train_per = 80

# Importing the dataset
#movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')
#ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
wtdata_train = pd.read_csv(train_file,sep=',', compression='gzip',parse_dates=[datetime_name])
to_drop = set(wtdata_train.columns).intersection(set(exclude_columns).difference([target_name]))
wtdata_train = wtdata_train.drop(to_drop, axis=1)
#Include columns?
if include_columns is not None:
    if not datetime_name in include_columns:
        include_columns.append(datetime_name)
    if not target_name in include_columns:
        include_columns.append(target_name)
    if not 'ld_id' in include_columns:
        include_columns.append('ld_id')
    if not 'ot' in include_columns:
        include_columns.append('ot')
    if not 'ot_all' in include_columns:
        include_columns.append('ot_all')
    wtdata_train = wtdata_train[list(set(wtdata_train.columns).intersection(include_columns))] 
    
# Identify columns all NA
idx_NA_columns_train = pd.isnull(wtdata_train).sum() > 0.9 * wtdata_train.shape[0]
if any(idx_NA_columns_train):
    wtdata_train = wtdata_train.drop(idx_NA_columns_train[idx_NA_columns_train == True].index, axis=1)
    # wtdata_test = wtdata_test.drop(idx_NA_columns_train[idx_NA_columns_train == True].index, axis=1)

#Drop columns ld_id,etc
wtdata_train_df = wtdata_train
#wtdata_train=wtdata_train_df
wtdata_train = wtdata_train.drop(list(set(wtdata_train.columns).intersection([target_name,'ld_id','ot','ot_all',datetime_name])), axis=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
sc = StandardScaler()
# X_train = sc.fit_transform(X_train.as_matrix())
wtdata_train = Imputer(missing_values='NaN', strategy='mean', axis=0).fit_transform(wtdata_train.as_matrix())
wtdata_train = sc.fit_transform(wtdata_train)

#Divide train-test
trainpos = int(np.floor(wtdata_train.shape[0]*train_per/100))
training_set = wtdata_train[1:(trainpos+1),:]
#training_set = np.array(training_set, dtype = 'int')
test_set = wtdata_train[(trainpos+1):,:]
#test_set = np.array(test_set, dtype = 'int')

# Getting the number of users and movies
#nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_users = int(training_set.shape[0])
#nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
nb_movies = int(training_set.shape[1])

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)# 20 son las columnas features, nº movies, 20 es el numero de neuronas en la primera
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for row in range(nb_users):
        input = Variable(training_set[row]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # 1e-10 is for Avoid 0/0 inf division
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(test_set.size()[0]):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10) # 1e-10 is for Avoid 0/0 inf division
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))