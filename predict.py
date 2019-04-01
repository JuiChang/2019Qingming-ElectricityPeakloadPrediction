import numpy as np
import pandas as pd
import datetime as dt
import xgboost as xgb

def pred2(data):
    data_2 = data.loc[:, ['weekday', '尖峰負載(MW)', '台中溫度']] 

    # leave only weekday
    mask_weekday = data_2['weekday'] <= 4
    data_2 = data_2.loc[mask_weekday]

    associ_data = data.loc[mask_weekday]

    data_2.drop(data_2.head(5).index, inplace=True) # drop fisrt 5 rows

    data_2['-1尖峰負載(MW)'] = (associ_data['尖峰負載(MW)'].tolist())[4:-1]
    data_2['-1台中溫度'] = (associ_data['台中溫度'].tolist())[4:-1]

    # leave only Tue
    mask = data_2['weekday'] == 1
    data_2 = data_2.loc[mask]

    data_2.drop('weekday', axis=1, inplace=True)

    # mask: certain month
    start_date = dt.datetime(2017,3,1)
    end_date = dt.datetime(2017,3,31)
    mask17 = (data_2.index > start_date) & (data_2.index <= end_date)
    start_date = dt.datetime(2018,3,1)
    end_date = dt.datetime(2018,3,31)
    mask18 = (data_2.index > start_date) & (data_2.index <= end_date)
    start_date = dt.datetime(2019,3,1)
    end_date = dt.datetime(2019,3,31)
    mask19 = (data_2.index > start_date) & (data_2.index <= end_date)
    mask = mask17 | mask18 | mask19
    data_2 = data_2.loc[mask]


    X = data_2.drop('尖峰負載(MW)', axis=1)
    Y = data_2.loc[:, '尖峰負載(MW)']
    dtrain = xgb.DMatrix(X, label=Y.tolist())

    param = {'max_depth': 5}
    param['nthread'] = 4
    param['eval_metric'] = 'rmse'

    num_round = 40
    model2 = xgb.train(param, dtrain, num_round)

    # test df
    columns = data_2.columns[1:]
    index = ['2019-04-02']
    test = pd.DataFrame(index=index, columns=columns)

    # TBD
    test['台中溫度'] = 26
    test['-1尖峰負載(MW)'] = data.loc['2019-04-01', '尖峰負載(MW)']
    test['-1台中溫度'] = data.loc['2019-04-01', '台中溫度']
    dtest = xgb.DMatrix(test)
    ypred = model2.predict(dtest)

    return ypred[0]


def pred3(data):
    data_2 = data.loc[:, ['weekday', '尖峰負載(MW)', '台中溫度']] 

    # leave only weekday
    mask_weekday = data_2['weekday'] <= 4
    data_2 = data_2.loc[mask_weekday]

    associ_data = data.loc[mask_weekday]

    data_2.drop(data_2.head(3).index, inplace=True) # drop fisrt 3 rows

    data_2['-2尖峰負載(MW)'] = (associ_data['尖峰負載(MW)'].tolist())[1:-2]
    data_2['-2台中溫度'] = (associ_data['台中溫度'].tolist())[1:-2]

    data_2['-3尖峰負載(MW)'] = (associ_data['尖峰負載(MW)'].tolist())[0:-3]
    data_2['-3台中溫度'] = (associ_data['台中溫度'].tolist())[0:-3]


    # leave only Wed
    mask = data_2['weekday'] == 2
    data_2 = data_2.loc[mask]

    data_2.drop('weekday', axis=1, inplace=True)



    start_date = dt.datetime(2017,2,1)
    end_date = dt.datetime(2017,3,31)
    mask17 = (data_2.index > start_date) & (data_2.index <= end_date)
    start_date = dt.datetime(2018,3,1)
    end_date = dt.datetime(2018,3,31)
    mask18 = (data_2.index > start_date) & (data_2.index <= end_date)
    start_date = dt.datetime(2019,2,15)
    end_date = dt.datetime(2019,3,31)
    mask19 = (data_2.index > start_date) & (data_2.index <= end_date)
    mask = mask17 | mask18 | mask19

    data_2 = data_2.loc[mask]


    # data_2.head(5)

    X = data_2.drop('尖峰負載(MW)', axis=1)
    Y = data_2.loc[:, '尖峰負載(MW)']
    dtrain = xgb.DMatrix(X, label=Y.tolist())

    param = {'max_depth': 5}
    param['nthread'] = 4
    param['eval_metric'] = 'rmse'

    num_round = 40

    model3 = xgb.train(param, dtrain, num_round)

    columns = data_2.columns[1:]
    index = ['2019-04-03']
    test = pd.DataFrame(index=index, columns=columns)

    # TBD
    test['台中溫度'] = 26
    test['-2尖峰負載(MW)'] = data.loc['2019-04-01', '尖峰負載(MW)']
    test['-2台中溫度'] = data.loc['2019-04-01', '台中溫度']
    test['-3尖峰負載(MW)'] = data.loc['2019-03-29', '尖峰負載(MW)']
    test['-3台中溫度'] = data.loc['2019-03-29', '台中溫度']
    dtest = xgb.DMatrix(test)
    ypred = model3.predict(dtest)

    return ypred[0]


def pred8(data):

    data_2 = data.loc[:, ['weekday', '尖峰負載(MW)', '台中溫度']] 

    # leave only weekday
    mask_weekday = data_2['weekday'] <= 4
    data_2 = data_2.loc[mask_weekday]

    associ_data = data.loc[mask_weekday]
    #     associ_data = data.copy()

    data_2.drop(data_2.head(7).index, inplace=True) # drop fisrt 9 rows

    data_2['-5尖峰負載(MW)'] = (associ_data['尖峰負載(MW)'].tolist())[2:-5]
    data_2['-5台中溫度'] = (associ_data['台中溫度'].tolist())[2:-5]

    data_2['-6尖峰負載(MW)'] = (associ_data['尖峰負載(MW)'].tolist())[1:-6]
    data_2['-6備轉容量(MW)'] = (associ_data['備轉容量(MW)'].tolist())[1:-6]
    data_2['-6台中溫度'] = (associ_data['台中溫度'].tolist())[1:-6]

    data_2['-7尖峰負載(MW)'] = (associ_data['尖峰負載(MW)'].tolist())[:-7]
    data_2['-7備轉容量(MW)'] = (associ_data['備轉容量(MW)'].tolist())[:-7]
    data_2['-7台中溫度'] = (associ_data['台中溫度'].tolist())[:-7]


    # leave only Mon
    mask = data_2['weekday'] == 0
    data_2 = data_2.loc[mask]

    data_2.drop('weekday', axis=1, inplace=True)


    start_date = dt.datetime(2017,3,1)
    end_date = dt.datetime(2017,3,31)
    mask17 = (data_2.index > start_date) & (data_2.index <= end_date)
    start_date = dt.datetime(2018,3,1)
    end_date = dt.datetime(2018,3,31)
    mask18 = (data_2.index > start_date) & (data_2.index <= end_date)
    start_date = dt.datetime(2019,3,1)
    end_date = dt.datetime(2019,3,31)
    mask19 = (data_2.index > start_date) & (data_2.index <= end_date)
    mask = mask17 | mask18 | mask19

    data_2 = data_2.loc[mask]



    X = data_2.drop('尖峰負載(MW)', axis=1)
    Y = data_2.loc[:, '尖峰負載(MW)']
    dtrain = xgb.DMatrix(X, label=Y.tolist())

    param = {'max_depth': 5}
    param['nthread'] = 4
    param['eval_metric'] = 'rmse'

    num_round = 40

    model8 = xgb.train(param, dtrain, num_round)


    columns = data_2.columns[1:]
    index = ['2019-04-08']
    test = pd.DataFrame(index=index, columns=columns)

    # TBD
    test['台中溫度'] = 26
    test['-5尖峰負載(MW)'] = data.loc['2019-04-01', '尖峰負載(MW)']
    test['-5台中溫度'] = data.loc['2019-04-01', '台中溫度']
    test['-6尖峰負載(MW)'] = data.loc['2019-03-29', '尖峰負載(MW)']
    test['-6備轉容量(MW)'] = data.loc['2019-03-29', '備轉容量(MW)']
    test['-6台中溫度'] = data.loc['2019-03-29', '台中溫度']
    test['-7尖峰負載(MW)'] = data.loc['2019-03-28', '尖峰負載(MW)']
    test['-7備轉容量(MW)'] = data.loc['2019-03-28', '備轉容量(MW)']
    test['-7台中溫度'] = data.loc['2019-03-28', '台中溫度']
    dtest = xgb.DMatrix(test)
    ypred = model8.predict(dtest)

    return ypred[0]


def pred4567(data):
    ret = list()

    # LR for 2019/4/4 by 2017/4/1 and 2018/4/4
    from sklearn.linear_model import LinearRegression

    # TBD: 4/4 ~ 4/7 temparature
    tempe = [27, 27, 28, 28]

    for i in range(4):
    
        X = [[data['台中溫度'][90 + i], data['尖峰負載(MW)'][85:90].mean()], 
            [data['台中溫度'][458 + i], (data['尖峰負載(MW)'][451:454].sum() + data['尖峰負載(MW)'][456:458].sum())/5]]
        y = [data['尖峰負載(MW)'][90 + i], data['尖峰負載(MW)'][458 + i]]

        reg = LinearRegression().fit(X, y)

        #print(reg.score(X, y), reg.coef_, reg.intercept_)

        # TBD
        ret.append(reg.predict([[tempe[i], (28535+28756+29140+30093+29673)/5]])[0])

    return ret
