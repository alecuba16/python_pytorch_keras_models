import pandas as pd
import numpy as np

#Configs
train_file = 'juanchodataset_train.csv.gz'
test_file = 'juanchodataset_test.csv.gz'
exclude_columns = ['alarm_block_code','alarm_all','alarm_all_block_code','ot','ot_block_code','ot_all','ot_all_block_code']
#target_name = 'pre_alarm'
target_name = 'alarm'
datetime_name = 'date_time'

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

X_train = wtdata_train.drop([target_name], axis=1)
y_train = wtdata_train[target_name]

X_test = wtdata_test.drop([target_name], axis=1)
y_test = wtdata_test[target_name]

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
X_train = sc.fit_transform(X_train.as_matrix())
X_test = sc.transform(X_test.as_matrix())

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#import keras.backend.tensorflow_backend as K

#with K.tf.device('/cpu:0'):
#    K.set_session(K.tf.Session(config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=True, device_count = {'GPU':-1})))
    # Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
#def build_classifier(optimizer):
#    classifier = Sequential()
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 153))
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return classifier
#classifier = KerasClassifier(build_fn = build_classifier)
#parameters = {'batch_size': [25,30],
#            'epochs': [400,500],
#            'optimizer': ['rmsprop']}
#grid_search = GridSearchCV(estimator = classifier,
#                           param_grid = parameters,
#                           scoring = 'accuracy',
#                           cv = 10)
#grid_search = grid_search.fit(X_train,y_train)
#best_parameters = grid_search.best_params_
#best_accuracy = grid_search.best_score_

def build_classifier2(input_dim):
    classifier = Sequential()
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu', input_dim = input_dim))
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier2 = build_classifier2(X_train.shape[1])
# Fitting the ANN to the Training set
classifier2.fit(X_train, y_train, batch_size = 10, epochs = 300)

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
pre_alarm_dates=X_test_df.ix[:,[datetime_name]][y_pred_bin==True]

#print(pre_alarm_dates)