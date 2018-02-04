#title           :FE_1dcnn.py
#description     :Essential functions for 1D-cnn for farmersedge project
#author          :SFL_scientific
#date            :2/1/2018

def windows(sample_data, window_size):
    start = 0
    while start < sample_data.shape[1]:
        yield start, start + window_size
        start += (window_size)
        
def reshape_data(data, nb_features, num_features):
    
    import numpy as np
    
    X_r = np.zeros((len(data), nb_features, num_features))

    for i,(start,end) in zip(range(num_features),windows(data,nb_features)):

        X_r[:, :, i] = data[:, start:end]
    return(X_r)

def data_merge (data1, data2, colname1, colname2, processing = True):
    
    if processing == True:
        data1[colname2] = data1[colname1].apply(lambda x: x.split('_')[0])
        
    data1[colname2] = data1[colname2].astype(int)
    data2[colname2] = data2[colname2].astype(int)

    MergeDat=data1.merge(data2,how="left")
    
    return(MergeDat)

def feat_drop_col (feat_data, dropby):
    print('Processing: Starting drop features by keywords:', dropby)
    
    full_col_list = feat_data.columns
    
    feats_drops = []
    
    for keyword in dropby:
    
        feats_drop = list(feat_data.filter(regex=(keyword)).columns)
        feats_drops.append(feats_drop)
    
    flat_list_drop = [item for sublist in feats_drops for item in sublist]
    flat_list_drop = list(set(flat_list_drop))
    print('Processing:',len(flat_list_drop), 'of features have been dropped...')

    feats_reserved = full_col_list.drop(flat_list_drop)
    feat_data = feat_data[feats_reserved]
    
    return(feat_data, feats_reserved)

def get_na_feats(X_sample):
    
    feats_na = X_sample.columns[X_sample.isna().any()].tolist()
    fes = []

    for col in feats_na:
        fe = feats_na[2].split('::')[2]
        fes.append(fe)

    fes = list(set(fes))
    
    temps = []

    for i in fes:
        temp = list(X_sample.filter(regex=(i)).columns)
        temps.append(temp)
    
    flat_list = [item for sublist in temps for item in sublist]

    return(flat_list)

def train_test_split(sample_data, valsize = 0.05, random_state = 42):
    
    from sklearn.model_selection import train_test_split

    _, _, train_idx, val_idx = train_test_split(sample_data, sample_data.index, test_size=valsize, random_state=random_state)
    
    sample_data.ix[train_idx,'split'] = 'Train'
    sample_data.ix[val_idx,'split'] = 'Validation'
    
    print(sample_data['split'].value_counts())
    
    return(sample_data)

def stdScaler(X_sample_nona):

    from sklearn.preprocessing import StandardScaler
    x_values = X_sample_nona.values

    scaler = StandardScaler().fit(x_values)
    scaled_x = scaler.transform(x_values)
    
    return(scaled_x)


def data_process(feature_data, y_data, ts_len, keycol1, keycol2, ycol, ylabel, drop_col = False, dropby = None):
    
    import pandas as pd
    import numpy as np
    
    # Read data
    train = feature_data
    yield_df = y_data
    
    sample_y = pd.concat([yield_df[keycol2],yield_df[ycol],yield_df[ylabel]],axis=1)

        
    
    if drop_col == True:
        if dropby != None:
            
            train, feats_reserved = feat_drop_col(train, dropby)
            
        else:
            print('Processing: Feature dropping is triggered but no keywords assigned for dropping')
            
    
    # merge data
    MergeDat = data_merge(train, sample_y, keycol1, keycol2)
    
    # drop na values in ycol
    df1 = MergeDat.dropna(subset=[ycol])
    
    # drop columns
    col_to_drop = [keycol1, keycol2, ycol, ylabel]
    X_sample = df1.drop(col_to_drop,axis = 1)
    
    fes = get_na_feats(X_sample)

    X_sample_nona = X_sample.drop(fes,axis = 1)
        
    df1 = train_test_split(df1, valsize = 0.25, random_state = 10)
        
    scaled_x = stdScaler(X_sample_nona)
    
    y = df1[ycol]

    X_train = scaled_x[df1.split=='Train']
    X_val = scaled_x[df1.split=='Validation']

    Y_train = y[df1.split=='Train']
    Y_val = y[df1.split=='Validation']
    
    X_full_r = reshape_data(scaled_x, ts_len, int(scaled_x.shape[1]/ts_len))
    print(X_full_r.shape)
    X_train_r = reshape_data(X_train, ts_len, int(X_train.shape[1]/ts_len))
    print(X_train_r.shape)
    X_val_r = reshape_data(X_val,ts_len, int(X_val.shape[1]/ts_len))
    print(X_val_r.shape)
    
    return(X_full_r, y, X_train_r, Y_train, X_val_r, Y_val, df1)

def fit_1d_cnn (X_train_r, Y_train, X_val_r, Y_val, lr = 0.0005, nb_epoch = 500, batch_size = 50):
    
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
    from keras.optimizers import SGD
    from keras.callbacks import ModelCheckpoint
    
    model = Sequential()
    model.add(Convolution1D(nb_filter=20, filter_length=2, input_shape=(X_train_r.shape[1], X_train_r.shape[2])))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1,kernel_initializer='normal'))
    model.add(Activation('linear'))

    sgd = SGD(lr= lr, nesterov=True, decay=1e-6, momentum=0.9)
    model.compile(loss='mean_absolute_error',optimizer=sgd)
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
    
    model.fit(X_train_r, Y_train, nb_epoch=500, validation_data=(X_val_r, Y_val), batch_size=50,callbacks=[checkpointer])

    model.load_weights('weights.hdf5')
    
    return(model)


def obtain_dummies(data, col):
    
    import pandas as pd

    full_label_dummy = pd.get_dummies(data[col])
    label_train = full_label_dummy[data.split == "Train"]
    label_val = full_label_dummy[data.split == "Validation"]
    
    return(full_label_dummy, label_train, label_val)

def abline(slope, intercept):
    '''
    DESC: Plot a line from slope and intercept
    INPUT: slope(float), intercept(float)
    -----
    OUTPUT: matplotlib plot with plotted line of desired slope and intercept
    '''
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', c='b', label='Perfect Predictions')

def line_of_best_fit(x, y,ci, label,c):
    '''
    DESC: Plot a line of best fit from scatter plot
    INPUT: x-coordinates(list/array), y-coordinates(list/array), confidence-interval(float), label(str)
    -----
    OUTPUT: seaborn plot with plotted line of best fit with confidence interval and equation for line of best fit
    '''
    import seaborn as sns
    import numpy as np
    sns.regplot(x, y, fit_reg=True, scatter=True, label=label, ci = ci,color=c )
    return np.polynomial.polynomial.polyfit(x, y, 1)

def actual_v_predictions_plot(actual, preds, title, metric, ci, pos3=None, color='orange',label_dummies=None, save=False, savename = None):
    '''
    DESC: Creates and acutal v.s. predictions plot to evaluate regressions
    INPUT: actual(list/array), preds(list/array), title(str), ci(float), pos1(tuple), pos2(tuple), save(bool)
    -----
    OUTPUT: matplotlib plot with prefect fit, line of best fit equation and plot, scatter plot of actual vs predicted values and MAPE
    '''
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    plt.xlim(0,10000)
    plt.ylim(0,10000)
    best_fit_eq= line_of_best_fit(actual, preds, ci = ci, label='Line of Best Fit with {}% CI'.format(ci), c=color)
    if isinstance(label_dummies, pd.DataFrame):
        labels = label_dummies.idxmax(axis=1)
        df = pd.DataFrame({'Actual':actual.tolist(), 'Predictions':list(preds),'Labels':labels})
        scatter_plot2d(df, 'Actual', 'Predictions', by='Labels')
    if isinstance(label_dummies, pd.Series):
        df = pd.DataFrame({'Actual':actual.tolist(), 'Predictions':list(preds),'Labels':label_dummies})
        scatter_plot2d(df, 'Actual', 'Predictions', by='Labels')
    abline(1,0)
    # MAE, RMSE, MAPE = regression_evaluation(df['Actual'].values, df['Predictions'].values)
    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    plt.title(title)
    plt.legend()
    plt.text(1000,8300,s='y = {}x + {}'.format(round(best_fit_eq[1],2), round(best_fit_eq[0],2)))
    if metric:
        plt.text(1000,7800, s='{}={}'.format(metric[0], round(metric[1],2)))
    if pos3:
        plt.text(pos3[0],pos3[1],'Canola = Red\nLentil = Green\nDurum = Blue\nHard Wheat = Purple')
    plt.rcParams.update({'font.size': 10, "figure.figsize":[6,6]})
    plt.tight_layout()
    if save:
        plt.savefig('./plot/'+savename+'.jpg')
    plt.show()
    print('Line of Best Fit: \t\t y = {}x + {}'. format(best_fit_eq[1], best_fit_eq[0]))
    
def histogram2d(actual, predictions):
    
    import matplotlib.pyplot as plt
    
    plt.hist2d(actual, predictions)
    plt.colorbar()
    plt.title('2D Histogram Actual v.s. Predicted Yield', fontsize=16)
    plt.xlabel('Predicted Yield', fontsize=16)
    plt.ylabel('Actual Yield', fontsize=16)
    plt.tight_layout()
    plt.savefig('hist2d')
    plt.show()
    
def predicted_actual_dist(actual, predictions, bins, save=False):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=(7,7))
    bins = np.linspace(0, 9000, bins)
    plt.hist(predictions, bins=bins, alpha=0.5, label='Predicted Yield')
    plt.hist(actual, bins=bins, color='g', alpha =0.5, label='Actual Yield')
    plt.xlabel('Yield', fontsize=16)
    plt.ylabel('Counts', fontsize=16)
    plt.legend(loc='best')
    plt.title('Distribution of Yield Predictions', fontsize=16)
    plt.tight_layout()
    if save:
        plt.savefig('Distribution of Yield Predictions.jpg')
    plt.show()
    
def scatter_plot2d(df,col1,col2,by=False,figsize=(8,6),label=['Canola','Durum','Lentil','Hard Wheat'],vmin=0,vmax=10000,xlabel=None,
                  ylabel=None,title=None,save=False):

    '''
    DESC:
            Plot 2d histogram colored by group column
    INPUT:
            df(pd.DataFrame):           Target dataframe
            co11(str):                  First target column
            col2(str):                  Second target column
            by(str):                    group column
            label(list):                legend labels
            vmin(int):                  min value for xlim/ylim
            vmax(int):                  max value for xlim/ymin

    -----
    OUTPUT: matplotlib 2d scatter plot with perfect matching line
    '''
    
    import matplotlib.pyplot as plt

    
    if by:
        num_unique = df[by].nunique()
        unique_value = sorted(df[by].unique())
        cmap = plt.cm.get_cmap('hsv',num_unique+1)

        colors=[]
        for i in range(num_unique):
            colors.append(cmap(i))

        for value,c in zip(unique_value,colors):
            print (c,value)
            plt.scatter(df.loc[df[by]==value][col1].values,df.loc[df[by]==value][col2].values,
                c=c,alpha=0.8,edgecolors = 'black')

        plt.xlim(vmin,vmax)
        plt.ylim(vmin,vmax)


        #plt.xticks(np.arange(1000,6000,1000),fontsize=12)
        #plt.yticks(np.arange(1000,6000,1000),fontsize=12)

        plt.title(title,fontsize=16)
        plt.xlabel(xlabel,fontsize=16)
        plt.ylabel(ylabel,fontsize=16)

        if save:
            plt.savefig(save)


def eval_metric(true_y, pred_y):

    
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # rmse
    rmse = sqrt(mean_squared_error(true_y, pred_y))

    # MAE
    mae = mean_absolute_error(true_y, pred_y)

    # MAPE
    mape = mean_absolute_percentage_error(true_y, pred_y)
    
    return(rmse, mae, mape)



def val_1dcnn(data, model, X_full_r, y, X_train_r, Y_train, X_val_r, Y_val, label_col,plot = True):
    
    import matplotlib.pyplot as plt
    
    # for all data
    ypred_all = model.predict(X_full_r)
    ypred_all = ypred_all.flatten()
    
    rmse_all, mae_all, mape_all = eval_metric(y, ypred_all)
    
    print("all data rmse = ", rmse_all)
    
    print('all data mae =', mae_all)
    
    print('all data mape =', mape_all)

    # for train data
    
    ypred_train = model.predict(X_train_r)
    ypred_train = ypred_train.flatten()
    
    rmse_train, mae_train, mape_train = eval_metric(Y_train, ypred_train)
    
    print("train data rmse = ", rmse_train)
    
    print('train data mae =', mae_train)
    
    print('train data mape =', mape_train)    

    # for test data
    
    ypred_val = model.predict(X_val_r)
    ypred_val = ypred_val.flatten()
    
    rmse_val, mae_val, mape_val = eval_metric(Y_val, ypred_val)
    
    print("test data rmse = ", rmse_val)
    
    print('test data mae =', mae_val)
    
    print('test data mape =', mape_val)    
    
    full_label_dummy, label_train, label_val = obtain_dummies(data, label_col)
    
    if plot == True:
            
        plt.style.use('default')
        actual_v_predictions_plot(actual = y, preds=ypred_all, title = "1D_CNN: Unmask Tiff Yield", pos3= [500,5000], metric = None, ci = 80, color='orange',label_dummies=full_label_dummy)
        predicted_actual_dist(y, ypred_all.flatten(), bins = 20, save = False)
        histogram2d(y, ypred_all)
        
        plt.style.use('default')
        actual_v_predictions_plot(actual = Y_val, preds=ypred_val, title = "1D_CNN: Unmask Tiff Yield testing data ONLY", metric = None, ci = 80, pos3=None, color='orange',label_dummies=label_val)
        predicted_actual_dist(Y_val, ypred_val.flatten(), bins = 20, save = False)
        histogram2d(Y_val, ypred_val)
        
        plt.style.use('default')
        actual_v_predictions_plot(actual = Y_train, preds=ypred_train, title = "1D_CNN: Unmask Tiff Yield training data ONLY", metric = None, ci = 80, pos3=None, color='orange',label_dummies=label_train)
        predicted_actual_dist(Y_train, ypred_train.flatten(), bins = 20, save = False)
        histogram2d(Y_train, ypred_train)
            
    return(ypred_all, ypred_val, ypred_train)