import numpy as np
import pandas as pd
import os
import easygui as eg 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from tqdm import tqdm


os.chdir('D:\Softpack\Software_top\Python\Python_Project\Project_script\KK_box_Music_Recommendation')
print('Loading data...')
data_path = '.\\Input\\'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')
songs = pd.read_csv(data_path + 'songs.csv')
members = pd.read_csv(data_path + 'members.csv')
songexinfo = pd.read_csv(data_path + 'song_extra_info.csv')

# cross validation testset: the last 3 million of the training set
train2 = train.iloc[-int(3e6):-1,:]
#
print('Data preprocessing...')
song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language', 'composer']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

songexinfo['record_cnt'] = songexinfo['isrc'].apply(lambda x: str(x)[0:2])
songexinfo['record_genre'] = songexinfo['isrc'].apply(lambda x: str(x)[2:5])
songexinfo['record_yr'] = songexinfo['isrc'].apply(lambda x: str(x)[5:7])

songex_cols = list(songexinfo.columns)
songex_cols.remove('isrc')
train = train.merge(songexinfo[songex_cols], on='song_id', how='left')
test = test.merge(songexinfo[songex_cols], on='song_id', how='left')

members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
members = members.drop(['registration_init_time'], axis=1)

members_cols = members.columns
train = train.merge(members[members_cols], on='msno', how='left')
test = test.merge(members[members_cols], on='msno', how='left')


" Checking section "
# check the NA ratio
def check_NA(df):
    import pandas as pd
    print pd.isnull(df).sum()/df.shape[0] * 100
    print df.shape[0]
#    print df.head(5)
check_NA(train) 
check_NA(test)   
check_NA(members) 
check_NA(songs)
check_NA(songexinfo)

tt =songexinfo.loc[pd.isnull(songexinfo['isrc']),'record_yr']
type(tt.iloc[0]) # str

" ==================================="

# dealing with NAN
train = train.fillna(-1)
test = test.fillna(-1)

#train = train.dropna(axis = 0, inplace = True)
#test = test.dropna(axis = 0, inplace = True)
#train.drop('gender', axis = 1, inplace = True)
#test.drop('gender', axis = 1, inplace = True)

import gc
del members, songs; gc.collect();

cols = list(train.columns)
cols.remove('target')

for col in tqdm(cols):
    if train[col].dtype == 'object':
        train[col] = train[col].apply(str)
        test[col] = test[col].apply(str)

        le = LabelEncoder()
        train_vals = list(train[col].unique())
        test_vals = list(test[col].unique())
        le.fit(train_vals + test_vals)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

X = np.array(train.drop(['target'], axis=1))
y = train['target'].values

X_test = np.array(test.drop(['id'], axis=1))
ids = test['id'].values

del train, test; gc.collect();

X_train, X_valid, y_train, y_valid = train_test_split(X, y, \
    test_size=0.1, random_state = 12)
    
del X, y; gc.collect();


d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_valid, label=y_valid) 

watchlist = [d_train, d_valid]


print('Training LGBM model...')
params = {}
#params['learning_rate'] = 0.4
params['application'] = 'binary'
params['max_depth'] = 20
params['num_leaves'] = 2**10
params['verbosity'] = 0
params['metric'] = 'binary_logloss'#'auc'

model = lgb.train(params, train_set=d_train, num_boost_round=200, valid_sets=watchlist, \
early_stopping_rounds=10, verbose_eval=10)

print('Making predictions and saving them...')
p_test = model.predict(X_test)

subm = pd.DataFrame()
subm['id'] = ids
subm['target'] = p_test
subm.to_csv('submission.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')