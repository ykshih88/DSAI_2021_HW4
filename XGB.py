import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import accuracy_score, mean_squared_error

def round_by_threshold(data,low,up,thr,vb=False):
    #data區間小於閾值的設為low (low=1,up=2,thr=0.9 , 1.7->1 , 1.91->2)
    if(vb):
        print('lower than threshold:',(np.array(data>low) & np.array(data<low+thr)).sum())
        print('greater than threshold:',(np.array(data>low+thr) & np.array(data<up)).sum())
        print('threshold:',thr)
    data[np.array(data>low) & np.array(data<low+thr)] = low
    data[np.array(data>low+thr) & np.array(data<up)] = up
    
    
    return data
    
def load():
    train_df = pd.read_csv('data/sales_train.csv')
    test_ori_df = pd.read_csv('data/test.csv')
    shops_df = pd.read_csv('data/shops.csv')
    items_df = pd.read_csv('data/items.csv')
    item_categories_df = pd.read_csv('data/item_categories.csv')
    return train_df,test_ori_df

def preprocess(train_df,test_ori_df):
    train_df = train_df.drop_duplicates()#drop duplicates
    train_df['date'] = pd.to_datetime(train_df['date'], dayfirst=True)
    train_df['date'] = train_df['date'].apply(lambda x: x.strftime('%Y-%m'))

    df = train_df.groupby(['date','shop_id','item_id']).sum()
    df = df.pivot_table(index=['shop_id','item_id'], columns='date', values='item_cnt_day', fill_value=0)
    df.reset_index(inplace=True)
    df[df<0] = 0


    test_df = pd.merge(test_ori_df, df, on=['shop_id','item_id'], how='left')
    test_df.drop(['ID'], axis=1, inplace=True)
    test_df = test_df.fillna(0)

    #drop_shop in test not in train
    drop_shop = []
    for i in range(60):
        if i not in test_df.shop_id.unique():
            drop_shop.append(i)
            df.drop(df[df.shop_id == i].index,inplace=True)
    # print('shop should drop:',drop_shop)
    # print('after drop shop:',df.shop_id.unique())

    #drop_item in test not in train
    drop_item = np.setdiff1d(df.item_id.unique(), test_df.item_id.unique())
    drop_mask = df.item_id == drop_item[0]#第0個初始化mask
    for i in range(1,len(drop_item)):
        drop_mask = drop_mask|(df.item_id == drop_item[i])
    df.drop(df[drop_mask].index,inplace=True)
    # print('item should drop:',len(drop_item))
    # print('after drop item:',len(df.item_id.unique()))

    #build one hot of shop_id
    shop_id_onehot = pd.get_dummies(test_df['shop_id'])
    test_df = test_df.join(shop_id_onehot)

    shop_id_onehot = pd.get_dummies(df['shop_id'])
    df = df.join(shop_id_onehot)


    #drop date data
    Y_train = df['2014-11'].values
    X_train = df.drop(['2015-10'], axis = 1)
    X_train = X_train.drop(['2015-09'], axis = 1)
    X_train = X_train.drop(['2015-08'], axis = 1)
    X_train = X_train.drop(['2015-07'], axis = 1)
    X_train = X_train.drop(['2015-06'], axis = 1)
    X_train = X_train.drop(['2015-05'], axis = 1)
    X_train = X_train.drop(['2015-04'], axis = 1)
    X_train = X_train.drop(['2015-03'], axis = 1)
    X_train = X_train.drop(['2015-02'], axis = 1)
    X_train = X_train.drop(['2015-01'], axis = 1)
    X_train = X_train.drop(['2014-12'], axis = 1)
    X_train = X_train.drop(['2014-11'], axis = 1)
        
    X_test = test_df
    for i in range(1,13):
        n = '0'+str(i) if i<10 else str(i)
        X_test = X_test.drop(['2013-'+n], axis = 1)

    #drop item_id and shop_id
    X_train.drop(['item_id'], axis=1, inplace=True)
    X_train.drop(['shop_id'], axis=1, inplace=True)
    X_train[X_train<=0]=1e-3

    X_test.drop(['item_id'], axis=1, inplace=True)
    X_test.drop(['shop_id'], axis=1, inplace=True)
    

    print(X_train.shape, Y_train.shape)
    print(X_test.shape)
    return X_train,Y_train,X_test

def postprocess(preds):
    thr_list= [0.75]
    preds = round_by_threshold(preds,0,1,thr_list[0])
    for i in range(4,14):
        print(i,i+1,(np.array(preds>i) & np.array(preds<i+1)).sum(),preds[np.array(preds>i) & np.array(preds<i+1)].mean())
        preds = round_by_threshold(preds,i,i+1,1-(preds[np.array(preds>i) & np.array(preds<i+1)].mean()-i))
    #     preds = round_by_threshold(preds,i,i+1,0.3,True)
        
    # preds = preds.round()
    print('preds<0:',(preds<0).sum())
    print('preds>=19:',(preds>=19).sum())
    preds[preds<0]=0
    preds[preds>=19]=10

def train_and_test(X_train,Y_train,X_test):
    clf = XGBRegressor(random_state = 41
                   ,max_depth=15
                   ,n_estimators=300
                   ,min_child_weight=1
                   ,objective = 'reg:squaredlogerror'
                   ,learning_rate=0.0500000012
                   ,tree_method='hist')
    clf.fit(X_train, Y_train)
    preds=pd.Series(clf.predict(X_test))
    return preds

def output_result(X_test,preds):
    submission = pd.DataFrame({
    "ID": X_test.index, 
    "item_cnt_month": preds
    })
    submission.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    train_df,test_ori_df= load()
    X_train,Y_train,X_test = preprocess(train_df,test_ori_df)
    preds = train_and_test(X_train,Y_train,X_test)
    result = postprocess(preds)
    output_result(X_test,preds)

