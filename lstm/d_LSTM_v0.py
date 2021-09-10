import pandas as pd
import numpy as np

#Configs
train_file = 'A801_wtdata_train_alarm.csv.gz'
test_file = 'A801_wtdata_test_alarm.csv.gz'
exclude_columns = ['alarm','alarm_block_code','alarm_all','alarm_all_block_code','ot','ot_block_code','ot_all','ot_all_block_code']
target_name = 'pre_alarm'
#target_name = 'alarm'
datetime_name = 'date_time'
timesteps = 50
batch_size = 500
epochs = 50

wtdata_train = pd.read_csv(train_file,sep=',',compression='gzip')
wtdata_test = pd.read_csv(test_file,sep=',',compression='gzip')

to_drop = set(wtdata_train.columns).intersection(exclude_columns)
wtdata_train = wtdata_train.drop(to_drop, axis=1)

to_drop = set(wtdata_test.columns).intersection(exclude_columns)
wtdata_test = wtdata_test.drop(to_drop, axis=1)

#Identify columns all NA
idx_NA_columns_train = pd.isnull(wtdata_train).sum()==wtdata_train.shape[0]
idx_NA_columns_test = pd.isnull(wtdata_test).sum()==wtdata_test.shape[0]
idx_NA_columns_traintest =idx_NA_columns_train[(idx_NA_columns_train | idx_NA_columns_test)==True].index
wtdata_train=wtdata_train.drop(idx_NA_columns_traintest,axis=1)
wtdata_test=wtdata_test.drop(idx_NA_columns_traintest,axis=1)
                                               
wtdata_train = wtdata_train.dropna()
wtdata_test = wtdata_test.dropna()

y_train = wtdata_train.ix[:, target_name]
y_train = y_train.as_matrix()
X_train = wtdata_train.drop([target_name], axis=1)

y_test = wtdata_test.ix[:, target_name]
y_test = y_test.as_matrix()
X_test = wtdata_test.drop([target_name], axis=1)

num_features = X_train.shape[1]

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
X_train_df=X_train
to_drop = set(X_train.columns).intersection([datetime_name,target_name])
X_train=X_train.drop(to_drop, axis=1)

X_test_df=X_test
to_drop = set(X_test.columns).intersection([datetime_name,target_name])
X_test=X_test.drop(to_drop, axis=1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
training_set_scaled = sc.fit_transform(X_train.as_matrix())
test_set_scaled = sc.transform(X_test.as_matrix())
# Reshaping
#X_train = np.reshape(X_train, (X_train.shape[0], 1,X_train.shape[1]))

# Creating a data structure with 20 timesteps and t+1 output
X_temp_timestepped = []
y_temp_timestepped = []
for i in range(timesteps, training_set_scaled.shape[0]):
    X_temp_timestepped.append(training_set_scaled[i-timesteps:i, :])
    y_temp_timestepped.append(y_train[i])
X_train, y_train = np.array(X_temp_timestepped), np.array(y_temp_timestepped)

X_temp_timestepped = []
y_temp_timestepped = []
for i in range(timesteps, test_set_scaled.shape[0]):
    X_temp_timestepped.append(test_set_scaled[i-timesteps:i, :])
    y_temp_timestepped.append(y_train[i])
X_test, y_test = np.array(X_temp_timestepped), np.array(y_temp_timestepped)

#Disable GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#Seed
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def build_classifier2(input_dim):
    classifier = Sequential()
    classifier.add(LSTM(units = 3, return_sequences=True,input_shape = (timesteps,input_dim[1])))
    classifier.add(LSTM(units = 3, return_sequences=True))
    classifier.add(LSTM(units=3))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier2 = build_classifier2([X_train.shape[0],X_train.shape[2]])
# Fitting the ANN to the Training set
classifier2.fit(X_train, y_train, batch_size = batch_size, epochs = epochs)

#Save model
# serialize model to JSON
model_json = classifier2.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
classifier2.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
classifier2 = json_file.read()
json_file.close()
from keras.models import model_from_json
classifier2 = model_from_json(classifier2)
# load weights into new model
classifier2.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
classifier2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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
pre_alarm_dates=X_test_df.ix[(timesteps+1):,[datetime_name]][y_pred_bin==True]
pre_alarm_dates.to_csv('results.csv', sep=',',index =False)
#print(pre_alarm_dates)
