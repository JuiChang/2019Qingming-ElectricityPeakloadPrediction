import numpy as np
import pandas as pd
import predict


# Create output dataframe
index = pd.date_range(start='4/2/2019', periods=7, freq='D').format(formatter=lambda x: x.strftime('%Y%m%d'))
columns = ['peak_load(MW)']
submission = pd.DataFrame(index=index, columns=columns)
submission.index.names = ['date']

# load data
data = pd.read_csv('2017_19326_t.csv')

# change to common datetime format, and set as index
data['日期'] = pd.to_datetime(data['日期'], format='%Y%m%d')
data.set_index('日期', inplace=True)

# add weekday column and move to front
data['weekday'] = data.index.dayofweek
cols = data.columns.tolist()
cols = cols[-1:] + cols[:-1]
data = data[cols]

# pass data to each predicting function, get the prediction and save to the dataframe
submission.loc['20190402'] = round(predict.pred2(data))
submission.loc['20190403'] = round(predict.pred3(data))
qm = np.array(predict.pred4567(data))
submission.loc[['20190404', '20190405', '20190406', '20190407']] = np.expand_dims(np.round_(qm), axis=1)
submission.loc['20190408'] = round(predict.pred8(data))

# output the dataframe to csv
submission.to_csv('submission.csv')