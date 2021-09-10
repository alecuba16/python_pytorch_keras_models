import gzip
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
import os
import re
import plotly.offline as plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
from functions.tensorsom import SOM

# Configs
#train_file = '../datasets/Escamb_wtdata_train_alarm_600.csv.gz'
#train_file = '../datasets/Escamb_wtdata_train_alarm_86400.csv.gz'
#train_file = '../datasets/Izco_wtdata_train_alarm_Gbox1_2014-2017_86400.csv.gz'
train_file = '../datasets/Moncay_wtdata_train_ot_Gen1_2014-2017_86400.csv.gz'
#train_file = '../datasets/STA_wtdata_train_alarm_86400.csv.gz'
#test_file = '../datasets/juancho_test.csv.gz'
exclude_columns = ['alarm_block_code', 'alarm_all', 'alarm_all_block_code', 'alarm', 'ot_block_code', 'ot_all', 'ot_all_block_code']
#include_columns = ['VelViento_avg','Pot_avg','VelRotor_avg','TempAceiteMultip_avg','TempAmb_avg','TempRodamMultip_avg'] #Escamb multi
#include_columns = ['VelViento_avg','Pot_avg'] #Escamb multi
#include_columns = ['VelViento_avg','Pot_avg','VelRotor_avg','TempAceiteMultip_avg','TempAmb_avg','TempMultip_avg'] #Izco multi
include_columns = ['VelViento_avg','Pot_avg','VelRotor_avg','TempAceiteMultip_avg','TempAmb_avg','TempRodamMultip_avg'] #Moncay multi


# target_name = 'pre_alarm'
Marging=5
#target_name = 'alarm'
target_name = 'ot'
datetime_name = 'date_time'
som_size=25
threshold = 0.9
nums = re.compile(r"\_(\d+).csv.gz")
seconds=nums.search(train_file).group(1)
som_folder='results_fusion_'+str(seconds)
if not os.path.exists(som_folder):
    os.makedirs(som_folder)

print("Reading "+train_file+" file ...")
wtdata_train = pd.read_csv(train_file, sep=',', compression='gzip',usecols=['ld_id'])

filename=som_folder+'/fusion_som.pydata.gz'
if not Path(filename).is_file():
    print("Create model...")
    wtdata_train = pd.read_csv(train_file, sep=',', compression='gzip',parse_dates=[datetime_name],low_memory=False)
    #wtdata_test = pd.read_csv(test_file, sep=',', compression='gzip')

    to_drop = set(wtdata_train.columns).intersection(set(exclude_columns).difference([target_name]))
    wtdata_train = wtdata_train.drop(to_drop, axis=1)

    # to_drop = set(wtdata_test.columns).intersection(set(exclude_columns).difference([target_name]))
    # wtdata_test = wtdata_test.drop(to_drop, axis=1)

    if Marging > 0:
        dates_prealarm = []
        active_alarms = wtdata_train[wtdata_train[target_name] == 1][datetime_name].values
        for alarm in active_alarms:
            for m in range(0, Marging):
                dates_prealarm.append(alarm - np.timedelta64(m, 'D'))
        wtdata_train.loc[wtdata_train[datetime_name].isin(active_alarms), target_name] = 0
        wtdata_train.loc[wtdata_train[datetime_name].isin(dates_prealarm), target_name] = 1

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

    X_train = wtdata_train.drop([target_name], axis=1)
    y_train = wtdata_train[target_name]

    print("Numer of rows and columns of dataset:" + str(wtdata_train.shape[0]) + " x " + str(wtdata_train.shape[1]))
    # X_test = wtdata_test.drop([target_name], axis=1)
    # y_test = wtdata_test[target_name]

    #Add row id for mapping
    #X_train['row_id']=np.arange(X_train.shape[0])
    # X_test['row_id']=np.arange(X_test.shape[0])

    X_train_df = X_train
    to_drop = set(X_train.columns).intersection([datetime_name, target_name, 'ld_id'])
    X_train = X_train.drop(to_drop, axis=1)

    # X_test_df = X_test
    # to_drop = set(X_test.columns).intersection([datetime_name, target_name, 'ld_id'])
    # X_test = X_test.drop(to_drop, axis=1)

    # Feature Scaling
    sc = StandardScaler()
    # X_train = sc.fit_transform(X_train.as_matrix())
    X_train = Imputer(missing_values='NaN', strategy='mean', axis=0).fit_transform(X_train.as_matrix())
    X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test.as_matrix())
    # X_test = Imputer(missing_values='NaN', strategy='mean', axis=0).fit_transform(X_test.as_matrix())
    # X_test = sc.transform(X_test)

    # Training the SOM
    #som_size = int((X_train.shape[0]**0.54321)) # slower
    #som_size = int((X_train.shape[0]**0.45)) # Faster
    #som_size = int((X_train.shape[0] ** 0.35))  # Faster
    som_size = 71 #Prueba para izco
    #sigma = som_size/2 #Slower
    sigma = round(som_size/3,1) #Faster
    learningrate=0.5
    print("SOM size:" + str(som_size)+" sigma(initial radius):"+str(sigma)+" alpha(learning rate):"+str(learningrate))
    som = SOM()
    print("Initializing SOM map randomly..")
    som.initialize(m=som_size, n=som_size, dim=X_train.shape[1], sigma=sigma, alpha=learningrate,n_iterations=50,random_seed=1)
    print("Training: ",end='')
    som.train(X_train)
    to_save=({'X_train':X_train,'y_train':y_train,'X_train_df':X_train_df,'scaler':sc,'trained_model':som.get_trained_model()})
    fd=gzip.open(filename,'wb')
    pickle.dump(to_save,fd,pickle.HIGHEST_PROTOCOL)
    fd.close()
else:
    #load
    print("Load from "+filename+"...")
    fd = gzip.open(filename, 'rb')
    dump = pickle.load(fd)
    fd.close()
    X_train=dump['X_train']
    X_train_df = dump['X_train_df']
    y_train=dump['y_train']
    som=SOM()
    som.set_trained_model(dump['trained_model'])
    som_size=som.get_map_size()[0]
    sc=dump['scaler']

# print("Creating som plot file...")
# #Filter by ld_id
# lds_ids=X_train_df['ld_id'].unique()
# for ld_id in lds_ids:
#     plot_file=som_folder+'/'+str(ld_id)+'_som_plot.html'
#     if not Path(plot_file).is_file():
#         current_X_train=X_train[X_train_df['ld_id']==ld_id]
#         current_Y_train=y_train[X_train_df['ld_id']==ld_id]
#         mapped = som.map_vects(current_X_train)
#         mapped = np.column_stack((mapped,current_Y_train))
#         df2 = pd.DataFrame(mapped, columns=['x', 'y','a'])
#         matrix_xy=np.chararray((som_size,som_size))
#         #-1=no info,0=no alarmas,1=todas alarmas,2=alguna alarma gana no alarmas,3=alguna alarma gana alarmas
#         for r in range(som_size):
#             for c in range(som_size):
#                 alarms=df2.loc[(df2.x == r) & (df2.y==c),'a']
#                 positive=np.count_nonzero(alarms)
#                 negative=np.size(alarms)-positive
#                 matrix_xy[r, c] = '?'
#                 if positive == 0 and negative > 0:
#                     matrix_xy[r, c] = 'H'
#                 if positive>0 and negative == 0:
#                     matrix_xy[r, c] = 'A'
#                 if positive>0 and positive<negative:
#                     matrix_xy[r,c] = 'a'
#                 if positive>0 and positive>=negative:
#                     matrix_xy[r,c] = 'h'
#
#         z=som.distance_map()
#         z_text = matrix_xy
#         x=y=list(range(som_size))
#         fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text,colorscale='Greys', hoverinfo='z')
#         #Make text size smaller
#         colors = {'?':'#00ccff','H': '#00ff00', 'A': '#FF0000', 'a': '#ff9933','h':' #ffff00'}
#         for i in range(len(fig.layout.annotations)):
#             key=fig.layout.annotations[i]['text']
#             if 'b\'' in key:
#                 key=key[2:-1]
#                 fig.layout.annotations[i]['text']=key
#             fig.layout.annotations[i]['font']['color']=colors[key]
#             #fig.layout.annotations[i].font.size = 8
#         plotly.plot(fig, filename=plot_file,auto_open=False)
#     else:
#         print(str(ld_id) + ":som Plot file exists.")

#Only for export map for cluster analysis:
lds_ids=X_train_df['ld_id'].unique()
for ld_id in lds_ids:
    map_csv='xy'+str(ld_id)+'.csv'
    if not Path(map_csv).is_file():
        current_X_train=X_train[X_train_df['ld_id']==ld_id]
        mapped = som.map_vects(current_X_train)
        df2 = pd.DataFrame(mapped, columns=['x', 'y'])
        df2.to_csv(map_csv,header=True,sep=',',index=False)
    else:
        print("File '"+map_csv+"' exists.")


for var in include_columns:
    #var='TempRodamMultip_avg'
    #var='TempAceiteMultip_avg'
    #var='VelViento_avg'
    #var='Pot_avg'
    print("Creating "+var+" file...")
    #Filter by ld_id
    lds_ids=X_train_df['ld_id'].unique()
    min_val=np.amin(X_train_df[var])
    max_val=np.amax(X_train_df[var])
    for ld_id in lds_ids:
        plot_file=som_folder+'/'+str(ld_id)+'_var_'+var+'_plot.html'
        if not Path(plot_file).is_file():
            current_X_train=X_train[X_train_df['ld_id']==ld_id]
            current_Y_train=y_train[X_train_df['ld_id']==ld_id]
            mapped = som.map_vects(current_X_train)
            mapped = np.column_stack((mapped,current_Y_train,X_train_df[X_train_df['ld_id']==ld_id][var]))
            df2 = pd.DataFrame(mapped, columns=['x', 'y','a',var])
            matrix_xy=np.chararray((som_size,som_size))
            map_matrix=np.zeros((som_size,som_size))
            #-1=no info,0=no alarmas,1=todas alarmas,2=alguna alarma gana no alarmas,3=alguna alarma gana alarmas
            for r in range(som_size):
                for c in range(som_size):
                    alarms=df2.loc[(df2.x == r) & (df2.y==c),'a']
                    current=df2.loc[(df2.x == r) & (df2.y==c),var].values
                    current = current[~np.isnan(current)]
                    if current.size>0:
                        #map_matrix[r, c] = np.median(current)
                        map_matrix[r, c] = np.mean(current)
                    else:
                        map_matrix[r, c]= np.nan
                    positive=np.count_nonzero(alarms)
                    negative=np.size(alarms)-positive
                    matrix_xy[r, c] = ' '
                    if positive == 0 and negative > 0:
                        matrix_xy[r, c] = 'H'
                    if positive>0 and negative == 0:
                        matrix_xy[r, c] = 'A'
                    if positive>0 and positive<negative:
                        matrix_xy[r,c] = 'a'
                    if positive>0 and positive>=negative:
                        matrix_xy[r,c] = 'h'
            z=map_matrix
            z_text = matrix_xy
            x=y=list(range(som_size))

            colorscale = [[min_val, '#FFFFFF'], [max_val, '#cc0000']]
            #fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text,colorscale='Greys', hoverinfo='z')
            fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, hoverinfo='z', zmin=min_val, zmax=max_val,showscale=True)
            #Make text size smaller
            colors = {'?':'#00ccff','H': '#00ff00', 'A': '#FF0000', 'a': '#ff9933','h':' #ffff00'}
            for i in range(len(fig.layout.annotations)):
                key=fig.layout.annotations[i]['text']
                if 'b\'' in key:
                    key=key[2:-1]
                    fig.layout.annotations[i]['text']=key
                    fig.layout.annotations[i]['font']['color']=colors[key]
                #fig.layout.annotations[i].font.size = 8
            plotly.plot(fig, filename=plot_file,auto_open=False)
        else:
            print(str(ld_id) + ":"+var+" Plot file exists.")

# for ld_id in lds_ids:
#     plot_file = som_folder + '/' + str(ld_id) + '_distance_plot.html'
#     if not Path(plot_file).is_file():
#         print(str(ld_id) + ":Creating distance plot file...")
#         current_X_train=X_train[X_train_df['ld_id']==ld_id]
#         current_X_train_df=X_train_df[X_train_df['ld_id']==ld_id]
#         current_Y_train=y_train[X_train_df['ld_id']==ld_id]
#         mapped = som.map_vects(current_X_train)
#         centroids=som.get_centroids()
#
#         distances=np.zeros(current_X_train.shape[0])
#         for i in range(current_X_train.shape[0]-1):
#             center=centroids[mapped[i][0]][mapped[i][1]]
#             distances[i]=np.sqrt(np.sum((current_X_train[i] - center) ** 2))
#
#         distances=np.convolve(distances, np.ones((15,)) / 15, mode='valid')
#         min_dist=np.amin(distances)
#         max_dist = np.amax(distances)
#         distances=(distances-min_dist)/(max_dist-min_dist)
#         data = [go.Scatter(x=X_train_df['date_time'], y=distances, name='Normalized Distance plot', line=dict( color=('rgb(22, 96, 167)'), width=4))]
#         ots=np.zeros((0,0))
#         if target_name == 'ot_all':
#             ots = current_X_train_df[current_Y_train == 1][datetime_name]
#         elif 'ot_all' in current_X_train_df.columns:
#             ots=current_X_train_df[current_X_train_df['ot_all']==1][datetime_name]
#         if ots.shape[0]>0:
#             lines=np.empty(ots.shape[0])
#             for i in range(ots.shape[0]):
#                 np.append(lines,{'type': 'line','x0': ots.iloc[i],'y0': 0,'x1':ots.iloc[i],'y1': 1,'line': {'color': 'rgb(55, 128, 191)','width': 3,}})
#             layout = dict(shapes=lines,title='Normalized Error distance to BMU plot', xaxis=dict(title='Date time'), yaxis=dict(range=[0, 1],title='Error'))
#         else:
#             layout = dict(title='Normalized Error distance to BMU plot', xaxis=dict(title='Date time'), yaxis=dict(range=[0, 1],title='Error'))
#         fig = dict(data=data, layout=layout)
#         plotly.plot(fig, filename=plot_file, auto_open=False)
#     else:
#         print(str(ld_id) + ":distance Plot file exists.")

# for ld_id in lds_ids:
#     result_file = som_folder+'/'+str(ld_id)+'_results_som_'+str(threshold)+'.csv'
#     if not Path(result_file).is_file():
#         current_X_train=X_train[X_train_df['ld_id']==ld_id]
#         current_X_train_df=X_train_df[X_train_df['ld_id']==ld_id]
#         current_Y_train=y_train[X_train_df['ld_id']==ld_id]
#         # Finding the outliers
#         print(str(ld_id) + ":Creating Result file...")
#         print(str(ld_id)+":Finding outliers date_time with threshold >"+str(threshold)+" ...")
#         mappings = np.array(som.map_vects(current_X_train))
#         distance_map=som.distance_map()
#         out=np.array(np.where(distance_map>threshold)).T
#         df = pd.DataFrame()
#         for i in range(out.shape[0]):
#             out_rows=current_X_train_df[(np.equal([out[i,0],out[i,1]],mappings).all(axis=1))]
#             if out_rows.shape[0] > 0:
#                 newdf=pd.DataFrame({'ld_id':out_rows['ld_id'],'date_time':out_rows[datetime_name],'val':distance_map[out[i,0],out[i,1]]})
#                 df = df.append(newdf, ignore_index = True)
#         if df.size >0:
#             df = df.sort_values([datetime_name],ascending=[True])
#         df.to_csv(som_folder+'/'+str(ld_id)+'_results_som_'+str(threshold)+'.csv', sep=',',index =False)
#     else:
#         print(str(ld_id) + ":Result file exists.")
#     print(str(ld_id) + ":Finish!")


# lds_ids=X_train_df['ld_id'].unique()
# for ld_id in lds_ids:
#     plot_file=som_folder+'/'+str(ld_id)+'_som_plot.html'
#     if not Path(plot_file).is_file():
#         current_X_train=X_train[X_train_df['ld_id']==ld_id]
#         current_X_time=X_train_df[X_train_df['ld_id']==ld_id][datetime_name]
#         mapped = pd.DataFrame(som.map_vects(current_X_train))
#         mapped[datetime_name] = current_X_time.values
#         mapped.columns=['x', 'y',datetime_name]
#         #Check registers by square block
#         #to_check_rc=[[[39,15],[44,16]],[[25,28],[29,34]]]
#         #to_check_rc = [[[25, 28], [29, 34]]]
#         #to_check_rc = [[[15, 35], [18, 38]]]
#         to_check_rc = [[[4, 52], [7, 54]]]
#         mark_dates=np.empty((0,0), dtype=np.datetime64, order='C')
#         for rc in to_check_rc:
#             for r in range(rc[0][0],rc[1][0]):
#                 for c in range(rc[0][1],rc[1][1]):
#                     mark_dates = np.append(mark_dates,mapped.loc[(mapped.x == r) & (mapped.y == c),datetime_name])
#         mark_dates.sort()
#         mark_dates


#Find x,y in mapped
#ld_id=216
#xy=[30,3]
#current_X_train=X_train[X_train_df['ld_id']==ld_id]
#current_X_train_df=X_train_df[X_train_df['ld_id']==ld_id]
#mapped = som.map_vects(current_X_train)
#whereInd = []
#for i,row in enumerate(mapped):
#    if all(row == xy):
#        whereInd.append((i))
#current_X_train_df.iloc[whereInd][datetime_name]