
from joblib import Parallel, delayed
from minisom import MiniSom
import gzip

def parallel_win(x, sw,xy,input_len,y):
    som = MiniSom(x=xy, y=xy, input_len=input_len, sigma=1.0, learning_rate=0.5)
    som.weights = sw
    w = som.winner(x)
    return [w[0], w[1],y]

if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Imputer
    from pathlib import Path
    import pickle
    import pandas as pd
    import numpy as np
    #from ggplot import *
    import sys
    import os
    import re

    # Set the interpreter bool
    try:
        if sys.ps1: interpreter = True
    except AttributeError:
        interpreter = False
        if sys.flags.interactive: interpreter = True

    # Configs
    train_file = 'Escamb_wtdata_train_alarm_600.csv.gz'
    #train_file = 'Escamb_wtdata_train_alarm_86400.csv.gz'
    #train_file = 'STA_wtdata_train_alarm_86400.csv.gz'
    #test_file = 'juancho_test.csv.gz'
    exclude_columns = ['alarm_block_code', 'alarm_all', 'alarm_all_block_code', 'ot', 'ot_block_code', 'ot_all',
                   'ot_all_block_code']
    include_columns = ['VelViento_avg','Pot_avg','VelRotor_avg','TempAceiteMultip_avg','TempAmb_avg','TempRodamMultip_avg'] #Escamb multi
    #include_columns = ['VelViento_avg','Pot_avg','VelRotor_avg','TempAceiteMultip_avg','TempAmb_avg','TempMultip_avg'] #Izco multi
 
    # target_name = 'pre_alarm'
    Marging=5
    target_name = 'ot_all'
    datetime_name = 'date_time'
    som_size=25
    threshold = 0.9
    nums = re.compile(r"\_(\d+).csv.gz")
    seconds=nums.search(train_file).group(1)
    som_folder='results_'+str(seconds)
    if not os.path.exists(som_folder):
        os.makedirs(som_folder)

    print("Reading "+train_file+" file ...")
    wtdata_train = pd.read_csv(train_file, sep=',', compression='gzip',usecols=['ld_id'])

    #Filter by ld_id
    lds_ids=wtdata_train['ld_id'].unique()

    for ld_id in lds_ids:
        print("Wind turbine " + str(ld_id) + " ...")
        filename=som_folder+'/'+str(ld_id)+'_som.pydata.gz'
        if not Path(filename).is_file():
            print(str(ld_id)+":Create model...")
            wtdata_train = pd.read_csv(train_file, sep=',', compression='gzip',parse_dates=[datetime_name])
            wtdata_train=wtdata_train[wtdata_train['ld_id']==ld_id]
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
            if include_columns is not None and not include_columns:
                wtdata_train = wtdata_train.loc[:,include_columns]

            # Identify columns all NA
            idx_NA_columns_train = pd.isnull(wtdata_train).sum() > 0.9 * wtdata_train.shape[0]
            if any(idx_NA_columns_train):
                wtdata_train = wtdata_train.drop(idx_NA_columns_train[idx_NA_columns_train == True].index, axis=1)
                # wtdata_test = wtdata_test.drop(idx_NA_columns_train[idx_NA_columns_train == True].index, axis=1)
			
            X_train = wtdata_train.drop([target_name], axis=1)
            y_train = wtdata_train[target_name]

            # X_test = wtdata_test.drop([target_name], axis=1)
            # y_test = wtdata_test[target_name]

            #Add row id for mapping
            X_train['row_id']=np.arange(X_train.shape[0])
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
            som_size = int(4*(wtdata_train.shape[0]**0.54321))
            print("Som size:"+str(som_size))
            if som_size < 100:
                sigma = 5
            if som_size < 500:
                sigma = 10
            if som_size < 1000:
                sigma = 20
            if som_size < 5000:
                sigma = 50
            if som_size < 10000:
                sigma = 100
            sigma = som_size/10
            learningrate=0.3
            som = MiniSom(x = som_size, y = som_size, input_len = X_train.shape[1], sigma = sigma, learning_rate = learningrate)
            som.random_weights_init(X_train)
            som.train_random(data = X_train, num_iteration = 3000)

            to_save=({'som_weights':som.weights,'X_train':X_train,'y_train':y_train,'X_train_df':X_train_df,'scaler':sc,'som_size':som_size,'ld_id':ld_id,'sigma':sigma,'learningrate':learningrate})
            fd=gzip.open(filename,'wb')
            pickle.dump(to_save,fd,pickle.HIGHEST_PROTOCOL)
            fd.close()
        else:
            #load
            print(str(ld_id)+":Load from "+filename+"...")
            fd = gzip.open(filename, 'rb')
            dump = pickle.load(fd)
            fd.close()
            X_train=dump['X_train']
            X_train_df = dump['X_train_df']
            y_train=dump['y_train']
            ld_id=dump['ld_id']
            som_size=dump['som_size']
            sigma=dump['sigma']
            learningrate=dump['learningrate']
            som = MiniSom(x = som_size, y = som_size, input_len = X_train.shape[1], sigma = sigma, learning_rate = learningrate)
            som.weights=dump['som_weights']
            sc=dump['scaler']

        sommap=som.distance_map().T
        #Ggplot
        # x=np.int64(np.repeat(range(sommap.shape[1]),repeats=sommap.shape[0]))
        # y=np.int64(np.reshape(np.repeat(range(sommap.shape[0]),axis=0,repeats=sommap.shape[0]),(sommap.shape[0],sommap.shape[0]),order='F').flatten())
        # fill=sommap.flatten()
        # df = pd.DataFrame({
        #     'x':x+0.5,
        #     'y':y+0.5,
        #     'fill':fill,
        # })
        # p=ggplot(aes(x='x',y='y',color='fill',fill='fill'),data=df)+geom_point(shape='s',size=500)
        # #
        # markers = ['o', 's']
        # colors = ['r', 'g']
        # df2 = pd.DataFrame(columns=('x', 'y', 'color','marker'))
        # for i, x in enumerate(X_train):
        #     w = som.winner(x)
        #     df2.loc[i]=[w[0]+0.5,w[1] + 0.5,colors[y_train[i]],markers[y_train[i]]]
        plot_file=som_folder+'/'+str(ld_id)+'_som_plot.html'
        if not Path(plot_file).is_file():
            print(str(ld_id) + ":Creating plot file...")
            if not interpreter:
                print(str(ld_id)+":Processing data map parallel...")
                rows=Parallel(n_jobs=10)(delayed(parallel_win)(x,som.weights,som_size,X_train.shape[1],y_train.iloc[i]) for i,x in enumerate(X_train))
            else:
                print(str(ld_id)+":Processing data map sequential...")
                rows=[]
                for i, x in enumerate(X_train):
                    w = som.winner(x)
                    if w is not None:
                        rows=np.append(rows,[w[0],w[1],y_train.iloc[i]],axis=0)
                rows = np.reshape(rows, (y_train.shape[0], (rows.shape[0]/y_train.shape[0]).__int__()))
            df2 = pd.DataFrame(rows, columns=['x', 'y','a'])
            #pickle.dump(df2, open(str(ld_id)+'_df2.pydata', 'wb'), pickle.HIGHEST_PROTOCOL)
            #df2 = pickle.load(open(str(ld_id)+'_df2.pydata', 'rb'))
            matrix_xy=np.chararray((som_size,som_size))
            #-1=no info,0=no alarmas,1=todas alarmas,2=alguna alarma gana no alarmas,3=alguna alarma gana alarmas
            for r in range(som_size):
                for c in range(som_size):
                    alarms=df2.loc[(df2.x == r) & (df2.y==c),'a']
                    positive=np.count_nonzero(alarms)
                    negative=np.size(alarms)-positive
                    matrix_xy[r, c] = '?'
                    if positive == 0 and negative > 0:
                        matrix_xy[r, c] = 'H'
                    if positive>0 and negative == 0:
                        matrix_xy[r, c] = 'A'
                    if positive>0 and positive<negative:
                        matrix_xy[r,c] = 'a'
                    if positive>0 and positive>=negative:
                        matrix_xy[r,c] = 'h'

            import plotly
            import plotly.figure_factory as ff
            z=sommap
            z_text = matrix_xy
            x=y=list(range(som_size))
            fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text,colorscale='Greys', hoverinfo='z')
            #Make text size smaller
            colors = {'?':'#00ccff','H': '#00ff00', 'A': '#FF0000', 'a': '#ff9933','h':' #ffff00'}
            for i in range(len(fig.layout.annotations)):
                key=fig.layout.annotations[i]['text']
                if 'b\'' in key:
                    key=key[2:-1]
                    fig.layout.annotations[i]['text']=key
                fig.layout.annotations[i]['font']['color']=colors[key]
                #fig.layout.annotations[i].font.size = 8
            plotly.offline.plot(fig, filename=plot_file,auto_open=False)
        else:
            print(str(ld_id) + ":Plot file exists.")
        #
        # from pylab import bone, pcolor, colorbar, plot, show
        # bone()
        # pcolor(som.distance_map().T)
        # colorbar()
        # for i, x in enumerate(X_train):
        #     geom_point()
        #     plot(df2['x'][i] + 0.5,
        #          df2['y'][i] + 0.5,
        #          markers[y_train[i]],
        #          markeredgecolor = colors[y_train[i]],
        #          markerfacecolor = 'None',
        #          markersize = 5,
        #          markeredgewidth = 2)
        # show()
        #
        #
        # # Visualizing the results
        # from pylab import bone, pcolor, colorbar, plot, show
        # bone()
        # pcolor(som.distance_map().T)
        # colorbar()
        # markers = ['o', 's']
        # colors = ['r', 'g']
        # for i, x in enumerate(X_train):
        #     w = som.winner(x)
        #     geom_point()
        #     plot(w[0] + 0.5,
        #          w[1] + 0.5,
        #          markers[y_train[i]],
        #          markeredgecolor = colors[y_train[i]],
        #          markerfacecolor = 'None',
        #          markersize = 5,
        #          markeredgewidth = 2)
        # show()
        #
        #
        #
        result_file = som_folder+'/'+str(ld_id)+'_results_som_'+str(threshold)+'.csv'
        if not Path(result_file).is_file():
            # Finding the outliers
            print(str(ld_id) + ":Creating Result file...")
            print(str(ld_id)+":Finding outliers date_time with threshold >"+str(threshold)+" ...")
            mappings = som.win_map(X_train)
            out=np.where(som.distance_map().T>threshold)
            df = pd.DataFrame()
            for i in range(out[0].shape[0]-1):
                map=mappings[(out[0][i],out[1][i])]
                if map.__len__() > 0:
                    outliers_rows=np.matrix(sc.inverse_transform(map))
                    rows_id=np.array(outliers_rows[:,-1]).flatten()
                    if rows_id is not None and rows_id.size>0:
                        newdf=pd.DataFrame({'ld_id':X_train_df.iloc[rows_id]['ld_id'],'date_time':X_train_df.iloc[rows_id][datetime_name],'val':som.distance_map().T[out[0][i],out[1][i]]})
                        df = df.append(newdf, ignore_index = True)
            if df.size >0:
                df = df.sort_values([datetime_name],ascending=[True])
            df.to_csv(som_folder+'/'+str(ld_id)+'_results_som_'+str(threshold)+'.csv', sep=',',index =False)
        else:
            print(str(ld_id) + ":Result file exists.")
        print(str(ld_id) + ":Finish!")