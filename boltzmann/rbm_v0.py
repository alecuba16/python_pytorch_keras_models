# Boltzmann Machines

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
train_file = '../datasets/juancho_train.csv.gz'
test_file = '../datasets/juancho_test.csv.gz'
exclude_columns = ['alarm_block_code', 'alarm_all', 'alarm_all_block_code', 'ot', 'ot_block_code', 'ot_all', 'ot_all_block_code']
include_columns = ['VelViento_avg','Pot_avg','VelRotor_avg','TempAceiteMultip_avg','TempAmb_avg','TempRodamMultip_avg'] #Escamb multi
target_name = 'alarm'
datetime_name = 'date_time'
train_per = 80

# Preparing the training set and the test set
wtdata_train = pd.read_csv(train_file,sep=',', compression='gzip',parse_dates=[datetime_name])
to_drop = set(wtdata_train.columns).intersection(set(exclude_columns).difference([target_name]))
wtdata_train = wtdata_train.drop(to_drop, axis=1)

if test_file is not None:
    wtdata_test = pd.read_csv(test_file,sep=',', compression='gzip',parse_dates=[datetime_name])
    to_drop = set(wtdata_test.columns).intersection(set(exclude_columns).difference([target_name]))
    wtdata_test = wtdata_test.drop(to_drop, axis=1)
    
if test_file is not None:
    #Drop columns ld_id,etc
    wtdata_train_df = wtdata_train
    #wtdata_train=wtdata_train_df
    wtdata_train = wtdata_train.drop(list(set(wtdata_train.columns).intersection([target_name,'ld_id','ot','ot_all',datetime_name])), axis=1)
    #Drop columns ld_id,etc
    wtdata_test_df = wtdata_test
    #wtdata_train=wtdata_train_df
    wtdata_test = wtdata_test.drop(list(set(wtdata_test.columns).intersection([target_name,'ld_id','ot','ot_all',datetime_name])), axis=1)
else:
    #Divide train-test
    trainpos = int(np.floor(wtdata_train.shape[0]*train_per/100))
    #Drop columns ld_id,etc
    wtdata_train_df = wtdata_train[1:(trainpos+1),:]
    wtdata_test_df =  wtdata_train[(trainpos+1):,:]
    #wtdata_train=wtdata_train_df
    wtdata_train = wtdata_train_df.drop(list(set(wtdata_train.columns).intersection([target_name,'ld_id','ot','ot_all',datetime_name])), axis=1)
    wtdata_test = wtdata_test_df.drop(list(set(wtdata_train.columns).intersection([target_name,'ld_id','ot','ot_all',datetime_name])), axis=1)

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
    wtdata_test = wtdata_test[list(set(wtdata_test.columns).intersection(include_columns))] 
    
# Identify columns all NA
idx_NA_columns_train = pd.isnull(wtdata_train).sum() > 0.9 * wtdata_train.shape[0]
if any(idx_NA_columns_train):
    wtdata_train = wtdata_train.drop(idx_NA_columns_train[idx_NA_columns_train == True].index, axis=1)
    wtdata_test = wtdata_test.drop(idx_NA_columns_train[idx_NA_columns_train == True].index, axis=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
sc = StandardScaler()
# X_train = sc.fit_transform(X_train.as_matrix())
wtdata_train = Imputer(missing_values='NaN', strategy='mean', axis=0).fit_transform(wtdata_train.as_matrix())
training_set = sc.fit_transform(wtdata_train)

wtdata_test = Imputer(missing_values='NaN', strategy='mean', axis=0).fit_transform(wtdata_test.as_matrix())
test_set = sc.fit_transform(wtdata_test)

# Getting the number of users and movies
#nb_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_users = int(training_set.shape[0])
#nb_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
nb_movies = int(training_set.shape[1])

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Creating the architecture of the Neural Network
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))

# Testing the RBM
test_loss = 0
s = 0.
testrows=test_set.size()[0]
error=np.zeros(testrows)
for id_user in range(testrows):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h)
        current_error = torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        test_loss += current_error
        error[id_user] = current_error
        s += 1.
    
wtdata_out = wtdata_test_df
wtdata_out['error']=error
import plotly.offline as plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go

lds_ids=wtdata_out['ld_id'].unique()
for ld_id in lds_ids:
    plot_file = str(ld_id)+'_error_plot.html'
    selected_wtdata=wtdata_out[wtdata_out['ld_id']==ld_id]
    error_moving_avg=np.convolve(selected_wtdata['error'], np.ones((15,)) / 15, mode='valid')
    min_dist=np.amin(error_moving_avg)
    max_dist = np.amax(error_moving_avg)
    error_moving_avg=(error_moving_avg-min_dist)/(max_dist-min_dist)
    data = [go.Scatter(x=selected_wtdata['date_time'], y=error_moving_avg, name='Normalized Error plot', line=dict( color=('rgb(22, 96, 167)'), width=4))]
    alarms=np.zeros((0,0))
    #alarms = selected_wtdata[selected_wtdata[target_name] == 1][datetime_name]
    number_of_alarms=alarms.shape[0]
    if number_of_alarms>0:
        lines=[None]*number_of_alarms
        for i in range(number_of_alarms):
            lines[i]={'type': 'line','x0': alarms.iloc[i],'y0': 0,'x1':alarms.iloc[i],'y1': 1,'line': {'color': 'rgb(244, 66, 72)','width': 3,}}
        layout = dict(shapes=lines,title='Normalized Error distance to BMU plot', xaxis=dict(title='Date time'), yaxis=dict(range=[0, 1],title='Error'))
    else:
        layout = dict(title='Normalized Error distance to BMU plot ld_id '+str(ld_id)+' for '+target_name, xaxis=dict(title='Date time'), yaxis=dict(range=[0, 1],title='Error'))
    fig = dict(data=data, layout=layout)
    plotly.plot(fig, filename=plot_file, auto_open=False)
