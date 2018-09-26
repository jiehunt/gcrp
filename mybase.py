import os
import gc
import json
import datetime
import time
import warnings

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from pandas.io.json import json_normalize

import matplotlib.pyplot as plt

from contextlib import contextmanager

import lightgbm as lgb
import xgboost as xgb

""""""""""""""""""""""""""""""
# system setting
""""""""""""""""""""""""""""""
warnings.filterwarnings('ignore')
os.environ["OMP_NUM_THREADS"] = "4"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

t0 = datetime.datetime.now()
log_name = "aaa" + '.log'
# log_file = open(log_name, 'a')


""""""""""""""""""""""""""""""
# Help Function
""""""""""""""""""""""""""""""
@contextmanager
def timer(name):
    """
    Taken from Konstantin Lopuhin https://www.kaggle.com/lopuhin
    in script named : Mercari Golf: 0.3875 CV in 75 LOC, 1900 s
    https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
    """
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

start_time = time.time()

""""""""""""""""""""""""""""""
# Feature
""""""""""""""""""""""""""""""
def load_df(csv_path='./input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv(csv_path,
        converters={column: json.loads for column in JSON_COLUMNS},
        dtype={'fullVisitorId': 'str'}, # Important!!
        nrows=nrows
        )

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


""""""""""""""""""""""""""""""
# Feature
""""""""""""""""""""""""""""""
def f_get_srk_feature():

    with timer("goto open train"):
        train_df = load_df()

    test_file = './input/test.csv'
    with timer("goto open test"):
        test_df =  load_df(test_file)

    train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')

    ### drop unused columns
    const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1 ]
    cols_to_drop = const_cols + ['sessionId']

    train_df = train_df.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
    test_df = test_df.drop(cols_to_drop, axis=1)

    # Impute 0 for missing target values
    train_df["totals.transactionRevenue"].fillna(0, inplace=True)
    # train_y = train_df["totals.transactionRevenue"].values

    # label encode the categorical variables and convert the numerical variables to float
    cat_cols = ["channelGrouping", "device.browser",
                "device.deviceCategory", "device.operatingSystem",
                "geoNetwork.city", "geoNetwork.continent",
                "geoNetwork.country", "geoNetwork.metro",
                "geoNetwork.networkDomain", "geoNetwork.region",
                "geoNetwork.subContinent", "trafficSource.adContent",
                "trafficSource.adwordsClickInfo.adNetworkType",
                "trafficSource.adwordsClickInfo.gclId",
                "trafficSource.adwordsClickInfo.page",
                "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
                "trafficSource.keyword", "trafficSource.medium",
                "trafficSource.referralPath", "trafficSource.source",
                'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']
    for col in cat_cols:
        print(col)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
        train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
        test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

    num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']
    for col in num_cols:
        train_df[col] = train_df[col].astype(float)
        test_df[col] = test_df[col].astype(float)

    return train_df, test_df, cat_cols, num_cols


# custom function to run light gbm model
# def run_lgb(train_X, train_y, val_X, val_y, test_X):
#     params = {
#         "objective" : "regression",
#         "metric" : "rmse",
#         "num_leaves" : 30,
#         "min_child_samples" : 100,
#         "learning_rate" : 0.1,
#         "bagging_fraction" : 0.7,
#         "feature_fraction" : 0.5,
#         "bagging_frequency" : 5,
#         "bagging_seed" : 2018,
#         'device' : 'gpu',
#         'gpu_platform_id' : 0,
#         'gpu_device_id': 0,
#         "verbosity" : -1
#     }
#
#     lgtrain = lgb.Dataset(train_X, label=train_y)
#     lgval = lgb.Dataset(val_X, label=val_y)
#     model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
#
#     pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
#     return pred_test_y, model

""""""""""""""""""""""""""""""
# Model
""""""""""""""""""""""""""""""
def m_lgb_model(train, test, cat_cols = None, num_cols = None):

    target = ["totals.transactionRevenue"]
    splits = 5

    params = {
        'boosting_type': 'gbdt',
        "objective" : "regression",
        'metric':'rmse',

        'learning_rate': 0.1,
        # 'learning_rate': 0.1,
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 4,  # -1 means no limit

        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)

        "bagging_fraction" : 0.7, # Subsample ratio of the training instance.
        "bagging_frequency" : 5, # frequence of subsample, <=0 means no enable
        "bagging_seed" : 2018,

        'feature_fraction': 0.7,  # Subsample ratio of columns when constructing each tree.

        'nthread': 4,
        'verbose': 0,

        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }

    # Split the train dataset into development and valid based on time
    # dev_df = train_df[train_df['date']<=datetime.date(2017,5,31)]
    # val_df = train_df[train_df['date']>datetime.date(2017,5,31)]
    # dev_df = train_df[train_df['date']<=20170531]
    # val_df = train_df[train_df['date']>20170531]
    # dev_y = np.log1p(dev_df["totals.transactionRevenue"].values)
    # val_y = np.log1p(val_df["totals.transactionRevenue"].values)
    #
    # dev_X = dev_df[cat_cols + num_cols]
    # val_X = val_df[cat_cols + num_cols]
    # test_X = test_df[cat_cols + num_cols]

    pred = np.zeros( shape=(len(test), 1) )
    folds = StratifiedKFold(n_splits = splits, random_state = 1982)
    predictors = cat_cols+num_cols

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train[predictors], np.log1p(train["totals.transactionRevenue"].values))):
        print ("goto %d fold :" % n_fold)

        X_train_n = train[predictors].iloc[trn_idx].values
        Y_train_n = np.log1p(train["totals.transactionRevenue"].iloc[trn_idx].values)

        X_valid_n = train[predictors].iloc[val_idx].values
        Y_valid_n = np.log1p(train["totals.transactionRevenue"].iloc[val_idx].values)

        dtrain = lgb.Dataset(X_train_n, label=Y_train_n,
                          feature_name=predictors,
                          )

        dvalid = lgb.Dataset(X_valid_n, label=Y_valid_n,
                          feature_name=predictors,
                          )

        evals_results = {}
        file_path = './model/'+'lgb_'+str(n_fold) +'.hdf5'

        if os.path.exists(file_path):
            my_model = file_path
        else:
            my_model = None

        model = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid], valid_names=['train','valid'],
                     evals_result=evals_results, num_boost_round=1000, early_stopping_rounds=50,
                     init_model = my_model,
                     verbose_eval=True, feval=None)

        model.save_model(file_path)

        if n_fold > 0:
            pred = model.predict(test[predictors], num_iteration=model.best_iteration) + pred
        else:
            pred = model.predict(test[predictors], num_iteration=model.best_iteration)


    # class_pred = pd.DataFrame(class_pred)
    # oof_names = ['is_attributed_oof']
    # class_pred.columns = oof_names
    # print("Full roc auc scores : %.6f" % roc_auc_score(train['is_attributed'], class_pred[oof_names]))

    # Save OOF predictions - may be interesting for stacking...
    # file_name = 'oof/'+str(model_type) + '_' + str(feature_type) +'_' + str(data_type) + '_oof.csv'
    # class_pred.to_csv(file_name, index=False, float_format="%.6f")


    pred = pred / splits
    pred =pd.DataFrame(pred)
    pred.columns = target
    return pred

""""""""""""""""""""""""""""""
# Ganerate Result
""""""""""""""""""""""""""""""
def g_make_single_submission(outfile, pred):
    submit = pd.read_csv('./input/test.csv', dtype={'fullVisitorId': 'str'}, usecols=['fullVisitorId'])

    pred[pred<0] = 0
    submit['PredictedLogRevenue'] =np.expm1(pred)
    submit = submit.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
    submit["PredictedLogRevenue"] = np.log1p(submit["PredictedLogRevenue"])
    submit.to_csv(outfile,float_format='%.3f', index=False)
    return

if __name__ == '__main__':

    use_pse = False
    model_type = 'lgb'

    ##################################
    # traing for nn
    ##################################
    with timer ("get train, test data ..."):
        train, test, cat_cols, num_cols = f_get_srk_feature()

    # if model_type == 'xgb' or model_type == 'lgb':
    #     print ("goto train ", str(model_type) )
    #     pred =  app_train(train, test, model_type,feature_type, data_set,use_pse, pseudo)
    # elif model_type == 'nn':
    #     pred = app_train_nn(train, test, model_type, feature_type, data_set)
    pred = m_lgb_model(train, test, cat_cols, num_cols)

    outfile = 'output/' + str(model_type) + '.csv'
    g_make_single_submission(outfile, pred)

    print('[{}] All Done!!!'.format(time.time() - start_time))
