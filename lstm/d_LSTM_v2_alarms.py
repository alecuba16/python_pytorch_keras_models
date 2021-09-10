import pandas as pd
import numpy as np
from functions.load_wtdata import load_wtdata
from pathlib import Path
import gc
import tempfile
import os
#Configs
result_folder = 'results'
if not os.path.exists(result_folder):
        os.makedirs(result_folder)
batch_size = 500
timesteps=4000
epochs=20
class_weight={0: 1.,1: 500.}
datetime_name='date_time'
wtdata_train = pd.read_csv('train.csv', sep=',',low_memory=False)

#Format date_time
wtdata_train[datetime_name]=pd.to_datetime(wtdata_train[datetime_name],format='%Y-%m-%d %H:%M:%S')

y_train = wtdata_train.loc[:,'pre_alarm']
y_train = y_train.as_matrix()
X_train = wtdata_train.drop(['pre_alarm'], axis=1)
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
X_train_df=X_train['date_time']
to_drop = set(X_train.columns).intersection(['date_time','pre_alarm'])
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

filename_model=result_folder+'/'+'alarm_model'

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
    classifier2.fit(np.array(X_train), np.array(y_train), batch_size = batch_size, epochs = epochs,class_weight = class_weight)

    #Save model
    # serialize model to JSON
    model_json = classifier2.to_json()
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

## Load test data
print("Loading disk test data...")
wtdata_test = pd.read_csv('test.csv', sep=',',low_memory=False)

#Format date_time
wtdata_test['date_time']=pd.to_datetime(wtdata_test['date_time'],format='%Y-%m-%d %H:%M:%S')

y_test = wtdata_test.loc[:, 'pre_alarm']
y_test = y_test.as_matrix()
X_test = wtdata_test.drop(['pre_alarm'], axis=1)
del wtdata_test

X_test_df=X_test['date_time']
to_drop = set(X_test.columns).intersection(['date_time','pre_alarm'])
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
#pre_alarm_dates=pd.DataFrame({'datetime':X_test_df[timesteps:].as_matrix(), 'predict':y_pred[:,0]})
#rest_filename=result_folder+'/result_test_.csv'
#pre_alarm_dates.to_csv(rest_filename, sep=',',index =False)

import matplotlib.pyplot as plt
plt.use('Agg')
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

plot_filename=result_folder+'/prediction_test_.png'
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
rest_filename=result_folder+'/result_train.csv'
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

plot_filename=result_folder+'/prediction_train.png'
fig = plt.gcf()
fig.set_size_inches(50, 30)
plt.savefig(plot_filename,dpi=100)
plt.close()
#pre_alarm_dates.to_csv('results.csv', sep=',',index =False)
#print(pre_alarm_dates)
