import pandas as pd
import numpy as np
from functions.load_wtdata import load_wtdata
from pathlib import Path
import gc
import tempfile
import os
#Configs
db_config = {'table_cast_park_dic':'1_cast_park_table_dic','host':"127.0.0.1",'user':"itestit",'password':"itestit2014",'db':"SCHistorical_DB"}
exclude_columns = ['alarm_block_code','alarm_all','alarm_all_block_code','alarm','ot','ot_block_code','ot_all','ot_all_block_code']
datetime_name = 'date_time'
result_folder = 'results'
if not os.path.exists(result_folder):
        os.makedirs(result_folder)
batch_size = 500
Marging=15 #15 dias antes los datos son malos.
# 2014-2015
# 'unix_timestamp_ini':1388534400,
# 'unix_timestamp_end':1420070399,
# 2015-2016
# 'unix_timestamp_ini':1420070499,
# 'unix_timestamp_end':1451606399,
# 2014-2016
#'unix_timestamp_ini':1388534400,
#'unix_timestamp_end':1451606399,
# 2016->
#'unix_timestamp_ini_test':1451606400,
#'unix_timestamp_end_test':1498236799,

#wt_query = {'timesteps':100,'epochs':50,'class_weight':{0: 1.,1: 10.},'ld_id':194,'ld_code':"B211",'wp_id':20,'wp_code':"izco",'seconds_to_aggregate':600,'array_id_walm':"607,608,613,627,631,659",'array_ot':"10067,10068",'freq_dat_med_min':10,'fault':'Gbox','type':"phealtdeep",'filter':"",'power_condition':"",'include_variables':"",'exclude_variables':"regex:model|fake_data|^SPCosPhi|^FrecRed|^Estado",'target_name':"alarm",'unix_timestamp_ini':1388534400,'unix_timestamp_end':1420070399,'unix_timestamp_ini_test':1420070499,'unix_timestamp_end_test':1500000000}
#wt_query = {'timesteps':100,'epochs':50,'class_weight':{0: 1.,1: 500.},'ld_id':212,'ld_code':"B312",'wp_id':20,'wp_code':"izco",'seconds_to_aggregate':600,'array_id_walm':"614,615,616,636,639,641",'array_ot':"10004",'freq_dat_med_min':10,'fault':'Gen','type':"phealtdeep",'filter':"",'power_condition':"",'include_variables':"",'exclude_variables':"regex:model|fake_data|^SPCosPhi|^FrecRed|^Estado",'target_name':"alarm",'unix_timestamp_ini':1388534400,'unix_timestamp_end':1420070399,'unix_timestamp_ini_test':1420070499,'unix_timestamp_end_test':1500000000}
#wt_query = {'timesteps':100,'epochs':50,'class_weight':{0: 1.,1: 100.},'ld_id':211,'ld_code':"B311",'wp_id':20,'wp_code':"izco",'seconds_to_aggregate':600,'array_id_walm':"614,615,616,636,639,641",'array_ot':"10004",'freq_dat_med_min':10,'fault':'Gen','type':"phealtdeep",'filter':"",'power_condition':"",'include_variables':"",'exclude_variables':"regex:model|fake_data|^SPCosPhi|^FrecRed|^Estado",'target_name':"alarm",'unix_timestamp_ini':1388534400,'unix_timestamp_end':1451606399,'unix_timestamp_ini_test':1420070499,'unix_timestamp_end_test':1500000000}
#wt_query = {'timesteps':100,'epochs':50,'class_weight':{0: 1.,1: 500.},'ld_id':189,'ld_code':"B206",'wp_id':20,'wp_code':"izco",'seconds_to_aggregate':600,'array_id_walm':"614,615,616,636,639,641",'array_ot':"10004",'freq_dat_med_min':10,'fault':'Gen','type':"phealtdeep",'filter':"",'power_condition':"",'include_variables':"",'exclude_variables':"regex:model|fake_data|^SPCosPhi|^FrecRed|^Estado",'target_name':"alarm",'unix_timestamp_ini':1388534400,'unix_timestamp_end':1420070399,'unix_timestamp_ini_test':1420070499,'unix_timestamp_end_test':1500000000}
#wt_query = {'timesteps':100,'epochs':50,'class_weight':{0: 1.,1: 500.},'ld_id':179,'ld_code':"B113",'wp_id':20,'wp_code':"izco",'seconds_to_aggregate':600,'array_id_walm':"614,615,616,636,639,641",'array_ot':"10004",'freq_dat_med_min':10,'fault':'Gen','type':"phealtdeep",'filter':"",'power_condition':"",'include_variables':"",'exclude_variables':"regex:model|fake_data|^SPCosPhi|^FrecRed|^Estado",'target_name':"alarm",'unix_timestamp_ini':1388534400,'unix_timestamp_end':1420070399,'unix_timestamp_ini_test':1420070499,'unix_timestamp_end_test':1500000000}
wt_query = {'timesteps':100,'epochs':50,'class_weight':{0: 1.,1: 500.},'ld_id':201,'ld_code':"B301",'wp_id':20,'wp_code':"izco",'seconds_to_aggregate':600,'array_id_walm':"614,615,616,636,639,641",'array_ot':"10004",'freq_dat_med_min':10,'fault':'Gen','type':"phealtdeep",'filter':"",'power_condition':"",'include_variables':"",'exclude_variables':"regex:model|fake_data|^SPCosPhi|^FrecRed|^Estado",'target_name':"alarm",'unix_timestamp_ini':1388534400,'unix_timestamp_end':1420070399,'unix_timestamp_ini_test':1420070499,'unix_timestamp_end_test':1500000000}
#Fuhrlander
#wt_query = {'timesteps':50,'epochs':10,'class_weight':{0: 1.,1: 500.},'ld_id':80,'ld_code':"FL701",'wp_id':13,'wp_code':"sant",'seconds_to_aggregate':300,'array_id_walm':"1271,1329,964,1306,2302,2304,2306,1369,1370",'array_ot':"",'freq_dat_med_min':5,'fault':'Gbox','type':"phealtdeep",'filter':"",'power_condition':"",'include_variables':"",'exclude_variables':"regex:model|fake_data|^SPCosPhi|^FrecRed|^Estado",'target_name':"alarm",'unix_timestamp_ini':1325376000,'unix_timestamp_end':1356998399,'unix_timestamp_ini_test':1388534400,'unix_timestamp_end_test':1420070399}

timesteps=wt_query['timesteps']
filename=str(result_folder+'/'+wt_query['ld_code'])+'_wtdata_train_'+wt_query['fault']+'_'+wt_query['target_name']+'_'+str(wt_query['unix_timestamp_ini'])+'_'+str(wt_query['unix_timestamp_end'])+'.csv.gz'
if not Path(filename).is_file():
    print(filename+" not found...Downloading train data...")
    wtdata_train=load_wtdata(wt_query=wt_query,db_config=db_config)
    wtdata_train.to_csv(filename, sep=',',index =False,compression='gzip')
else:
    print("Loading disk train data...")
    wtdata_train = pd.read_csv(filename, sep=',', compression='gzip',low_memory=False)

#Format date_time
wtdata_train[datetime_name]=pd.to_datetime(wtdata_train[datetime_name],format='%Y-%m-%d %H:%M:%S')

if wt_query['target_name']=='alarm' and 'ot_all' in wtdata_train.columns:
    #wtdata_train.loc[wtdata_train['ot_all'] == 1, 'alarm'] = 0
    wtdata_train = wtdata_train[wtdata_train['ot_all'] != 1]
if wt_query['target_name']=='alarm' and 'ot' in wtdata_train.columns:
    wtdata_train=wtdata_train[wtdata_train['ot'] != 1]

#Modify alarm to do pre_alarm
#from datetime import datetime, timedelta
#Anticipation = 14
#Marging=14
#dates_prealarm=[]
#active_alarms=wtdata_train[wtdata_train[wt_query['target_name']]==1][datetime_name].values
#for alarm in active_alarms:
#    for m in range(0,Marging):
#        dates_prealarm.append(alarm - np.timedelta64(Anticipation+m, 'D'))
#wtdata_train.loc[wtdata_train[datetime_name].isin(active_alarms),wt_query['target_name']]=0
#wtdata_train.loc[wtdata_train[datetime_name].isin(dates_prealarm),wt_query['target_name']]=1
from datetime import datetime, timedelta

dates_prealarm=[]
active_alarms=wtdata_train[wtdata_train[wt_query['target_name']]==1][datetime_name].values
for alarm in active_alarms:
    for m in range(0,Marging):
        dates_prealarm.append(alarm - np.timedelta64(m, 'D'))
wtdata_train.loc[wtdata_train[datetime_name].isin(active_alarms),wt_query['target_name']]=0
wtdata_train.loc[wtdata_train[datetime_name].isin(dates_prealarm),wt_query['target_name']]=1

del dates_prealarm, active_alarms
a=set(wtdata_train.columns)
a.difference
to_drop = set(wtdata_train.columns).intersection(exclude_columns).difference([wt_query['target_name']])
if any(to_drop):
    wtdata_train = wtdata_train.drop(to_drop, axis=1)

#Identify columns all NA
idx_NA_columns_train = pd.isnull(wtdata_train).sum()>0.9*wtdata_train.shape[0]
if any(idx_NA_columns_train):
    wtdata_train=wtdata_train.drop(idx_NA_columns_train[idx_NA_columns_train==True].index,axis=1)

wtdata_train = wtdata_train.dropna(axis=0,how='any',subset=set(wtdata_train.columns).difference(['date_time']))

y_train = wtdata_train.loc[:, wt_query['target_name']]
y_train = y_train.as_matrix()
X_train = wtdata_train.drop([wt_query['target_name']], axis=1)
del wtdata_train
gc.collect()

## Splitting the dataset into the Training set and Test set
#def non_shuffling_train_test_split(X, y, test_size=0.2):
#    import numpy as np
#    i = int((1 - test_size) * X.shape[0]) + 1
#    X_train, X_test = np.split(X, [i])
#    y_train, y_test = np.split(y, [i])
#    return X_train, X_test, y_train, y_test
#
#X_train, X_test, y_train, y_test = non_shuffling_train_test_split(X, y, test_size = 0.1)

#Copy and Drop date_time
X_train_df=X_train[datetime_name]
to_drop = set(X_train.columns).intersection([datetime_name,wt_query['target_name']])
X_train=X_train.drop(to_drop, axis=1)

num_features = X_train.shape[1]
num_rows = X_train.shape[0]
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train.as_matrix())

# Reshaping
#X_train = np.reshape(X_train, (X_train.shape[0], 1,X_train.shape[1]))

# Creating a data structure with timesteps and t+1 output
#Save in disk to get free memory
temp_train = tempfile.NamedTemporaryFile(prefix='temp_train')
X_temp_timestepped=np.memmap(temp_train, dtype='float64', mode='w+', shape=((X_train.shape[0]-timesteps),timesteps,X_train.shape[1]))
#X_temp_timestepped=np.empty(shape=((num_rows-timesteps)*timesteps,num_features))
#X_temp_timestepped=np.memmap('temp_matrix.tmp', dtype='float64', mode='w+', shape=((num_rows-timesteps)*timesteps,num_features))
for i in range(timesteps,X_train.shape[0]):
    X_temp_timestepped[i-timesteps,:]=np.reshape(X_train[i-timesteps:i, :],(timesteps,X_train.shape[1]))

X_train=X_temp_timestepped
del X_temp_timestepped
y_train=y_train[timesteps:]
gc.collect()

#Disable GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Seed
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

filename_model=str(result_folder+'/'+wt_query['ld_code'])+'_wtdata_train_'+wt_query['fault']+'_'+wt_query['target_name']+'_'+str(wt_query['unix_timestamp_ini'])+'_'+str(wt_query['unix_timestamp_end'])+'_model'

if not Path(filename_model+'.json').is_file():
    def build_classifier2(input_dim):
        classifier = Sequential()
        classifier.add(LSTM(units = 10, return_sequences=True,input_shape = (timesteps,input_dim[1])))
        classifier.add(LSTM(units = 10, return_sequences=True))
        classifier.add(LSTM(units = 10))
        classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        return classifier
    classifier2 = build_classifier2([X_train.shape[0],X_train.shape[2]])
    # Fitting the ANN to the Training set
    classifier2.fit(np.array(X_train), np.array(y_train), batch_size = batch_size, epochs = wt_query['epochs'],class_weight = wt_query['class_weight'])

    #Save model
    # serialize model to JSON
    model_json = classifier2.to_json()
    filename_model=str(result_folder+'/'+wt_query['ld_code'])+'_wtdata_train_'+wt_query['fault']+'_'+wt_query['target_name']+'_'+str(wt_query['unix_timestamp_ini'])+'_'+str(wt_query['unix_timestamp_end'])+'_model'
    with open(filename_model+'.json', "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    classifier2.save_weights(filename_model+'.h5')
    print("Saved model to disk")
else:
    json_file = open(filename_model + '.json', 'r')
    classifier2 = json_file.read()
    json_file.close()
    from keras.models import model_from_json
    classifier2 = model_from_json(classifier2)
    # load weights into new model
    classifier2.load_weights(filename_model+'.h5')
    print("Loaded model from disk")
# # load json and create model
# json_file = open(filename_model+'.json', 'r')
# classifier2 = json_file.read()
# json_file.close()
# from keras.models import model_from_json
# classifier2 = model_from_json(classifier2)
# # load weights into new model
# classifier2.load_weights(filename_model+'.h5')
# print("Loaded model from disk")

## Load test data
bk_ini=wt_query['unix_timestamp_ini']
bk_end=wt_query['unix_timestamp_end']
wt_query['unix_timestamp_ini']=wt_query['unix_timestamp_ini_test']
wt_query['unix_timestamp_end']=wt_query['unix_timestamp_end_test']
filename=str(result_folder+'/'+wt_query['ld_code'])+'_wtdata_test_'+wt_query['fault']+'_'+wt_query['target_name']+'_'+str(wt_query['unix_timestamp_ini'])+'_'+str(wt_query['unix_timestamp_end'])+'.csv.gz'
if not Path(filename).is_file():
    print(filename + " not found...Downloading test data...")
    wtdata_test=load_wtdata(wt_query=wt_query,db_config=db_config)
    wtdata_test.to_csv(filename, sep=',',index =False,compression='gzip')
else:
    print("Loading disk test data...")
    wtdata_test = pd.read_csv(filename, sep=',', compression='gzip',low_memory=False)
wt_query['unix_timestamp_ini']=bk_ini
wt_query['unix_timestamp_end']=bk_end

wtdata_test[datetime_name]=pd.to_datetime(wtdata_test[datetime_name],format='%Y-%m-%d %H:%M:%S')

if wt_query['target_name']=='alarm' and 'ot_all' in wtdata_test.columns:
    wtdata_test.loc[wtdata_test['ot_all'] == 1, 'alarm'] = 0
if wt_query['target_name']=='alarm' and 'ot' in wtdata_test.columns:
    wtdata_test.loc[wtdata_test['ot'] == 1, 'alarm'] = 0

to_drop = set(wtdata_test.columns).intersection(exclude_columns).difference([wt_query['target_name']])
if any(to_drop):
    wtdata_test = wtdata_test.drop(to_drop, axis=1)

dates_prealarm=[]
active_alarms=wtdata_test[wtdata_test[wt_query['target_name']]==1][datetime_name].values
for alarm in active_alarms:
    for m in range(0,Marging):
        dates_prealarm.append(alarm - np.timedelta64(m, 'D'))
wtdata_test.loc[wtdata_test[datetime_name].isin(active_alarms),wt_query['target_name']]=0
wtdata_test.loc[wtdata_test[datetime_name].isin(dates_prealarm),wt_query['target_name']]=1
if any(idx_NA_columns_train):
    wtdata_test=wtdata_test.drop(idx_NA_columns_train[idx_NA_columns_train==True].index,axis=1)

wtdata_test = wtdata_test.dropna(axis=0,how='any',subset=set(wtdata_test.columns).difference(['date_time']))
y_test = wtdata_test.loc[:, wt_query['target_name']]
y_test = y_test.as_matrix()
X_test = wtdata_test.drop([wt_query['target_name']], axis=1)
del wtdata_test

X_test_df=X_test[datetime_name]
to_drop = set(X_test.columns).intersection([datetime_name,wt_query['target_name']])
X_test=X_test.drop(to_drop, axis=1)
X_test = sc.transform(X_test.as_matrix())
temp_test = tempfile.NamedTemporaryFile(prefix='temp_train')
X_temp_timestepped=np.memmap(temp_test, dtype='float64', mode='w+', shape=((X_test.shape[0]-timesteps),timesteps,X_test.shape[1]))
#X_temp_timestepped=np.empty(shape=((num_rows-timesteps)*timesteps,num_features))
#X_temp_timestepped=np.memmap('temp_matrix.tmp', dtype='float64', mode='w+', shape=((num_rows-timesteps)*timesteps,num_features))
for i in range(timesteps,X_test.shape[0]):
    X_temp_timestepped[i-timesteps,:]=np.reshape(X_test[i-timesteps:i, :],(timesteps,X_test.shape[1]))

X_test=X_temp_timestepped
del X_temp_timestepped
y_test=y_test[timesteps:]
gc.collect()
## End prepare test data

# Predicting the Test set results
y_pred = classifier2.predict(X_test)

y_pred_df = pd.DataFrame(y_pred)
y_pred_bin = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_bin)

from sklearn.metrics import cohen_kappa_score
kap = cohen_kappa_score(y_pred_bin,y_test)

accuracy='NA'
if(np.unique(y_test).size>1 and np.unique(y_pred_bin).size>1):
    numerator=(cm[0,0]+cm[1,1])
    denominator=sum(sum(cm))
    if(denominator!=0 and (numerator!=denominator or denominator!=0)) :
        accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)
print(kap)
#print(y_pred_df)
pre_alarm_dates=pd.DataFrame({'datetime':X_test_df[timesteps:].as_matrix(), 'predict':y_pred[:,0]})
rest_filename=str(result_folder+'/'+wt_query['ld_code'])+'_'+wt_query['fault']+'_result_test_'+str(wt_query['unix_timestamp_ini'])+'_'+str(wt_query['unix_timestamp_end'])+'.csv'
pre_alarm_dates.to_csv(rest_filename, sep=',',index =False)

import matplotlib.pyplot as plt
from datetime import datetime

date_time = pd.to_datetime(X_test_df[timesteps:],format='%Y-%m-%d %H:%M:%S')
plt.plot(date_time,y_test, color = 'red', label = 'Real alarm')
plt.plot(date_time,y_pred, color = 'green', label = 'Prediction probability')
#plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')

#plt.plot(date_time,y_pred_bin, color = 'blue', label = 'Prediction alarm')

plt.title('Alarm Prediction test '+str(Marging)+' Marging')
plt.xlabel('Time')
plt.ylabel('Alarm probability')
plt.legend()

plot_filename=str(result_folder+'/'+wt_query['ld_code'])+'_'+wt_query['fault']+'_prediction_test_'+str(wt_query['unix_timestamp_ini_test'])+'_'+str(wt_query['unix_timestamp_end_test'])+'.png'
fig = plt.gcf()
fig.set_size_inches(50, 30)
plt.savefig(plot_filename,dpi=100)
plt.close()
#plt.show()


#Predict train
y_pred = classifier2.predict(X_train)

y_pred_df = pd.DataFrame(y_pred)
y_pred_bin = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred_bin)

from sklearn.metrics import cohen_kappa_score
kap = cohen_kappa_score(y_pred_bin,y_train)

accuracy='NA'
if(np.unique(y_train).size>1 and np.unique(y_pred_bin).size>1):
    numerator=(cm[0,0]+cm[1,1])
    denominator=sum(sum(cm))
    if(denominator!=0 and (numerator!=denominator or denominator!=0)) :
        accuracy=(cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)
print(kap)
#print(y_pred_df)
pre_alarm_dates=pd.DataFrame({'datetime':X_train_df[timesteps:].as_matrix(), 'predict':y_pred[:,0]})
rest_filename=str(result_folder+'/'+wt_query['ld_code'])+'_'+wt_query['fault']+'_result_train_'+str(wt_query['unix_timestamp_ini'])+'_'+str(wt_query['unix_timestamp_end'])+'.csv'
pre_alarm_dates.to_csv(rest_filename, sep=',',index =False)

import matplotlib.pyplot as plt
from datetime import datetime

date_time = pd.to_datetime(X_train_df[timesteps:],format='%Y-%m-%d %H:%M:%S')
plt.plot(date_time,y_train, color = 'red', label = 'Real alarm')
plt.plot(date_time,y_pred, color = 'green', label = 'Prediction probability')
#plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
#plt.plot(date_time,y_pred_bin, color = 'blue', label = 'Prediction alarm')

plt.title('Alarm Prediction train '+str(Marging)+' Marging')
plt.xlabel('Time')
plt.ylabel('Alarm probability')
plt.legend()

plot_filename=str(result_folder+'/'+wt_query['ld_code'])+'_'+wt_query['fault']+'_prediction_train_'+str(wt_query['unix_timestamp_ini'])+'_'+str(wt_query['unix_timestamp_end'])+'.png'
fig = plt.gcf()
fig.set_size_inches(50, 30)
plt.savefig(plot_filename,dpi=100)
plt.close()
#pre_alarm_dates.to_csv('results.csv', sep=',',index =False)
#print(pre_alarm_dates)
