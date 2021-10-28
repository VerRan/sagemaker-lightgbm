# flake8: noqa: F401
import os
import sys

import argparse
import joblib
# # import training function
# from training import parse_args, train_fn

# # import deployment functions
# from explaining import model_fn, predict_fn, input_fn, output_fn

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

def parse_args(sys_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tree_num_leaves",
        type=int,
        default=31
    )
    parser.add_argument(
        "--num_round",
        type=int,
        default=10
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
    )

    args, _ = parser.parse_known_args(sys_args)
    return args

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    
##local data 
#     X_data = pd.read_csv('/home/sagemaker-user/yeahmobi-bj-lightgbm/datasets/CR_train_x.csv')
#     Y_data = pd.read_csv('/home/sagemaker-user/yeahmobi-bj-lightgbm/datasets/CR_train_y.csv')

#     X_test_data = pd.read_csv('/home/sagemaker-user/yeahmobi-bj-lightgbm/datasets/CR_test_x.csv')
#     Y_test_data = pd.read_csv('/home/sagemaker-user/yeahmobi-bj-lightgbm/datasets/CR_test_y.csv')

    # prepare datasets for model training with sagemaker ec2 local path
    X_data = pd.read_csv('/opt/ml/input/data/x_train/CR_train_x.csv')
    Y_data = pd.read_csv('/opt/ml/input/data/y_train/CR_train_y.csv')
    X_test_data = pd.read_csv('/opt/ml/input/data/x_test/CR_test_x.csv')
    Y_test_data = pd.read_csv('/opt/ml/input/data/y_test/CR_test_y.csv')
 
    cf = ['pkg_name','pkg_size','category','sub_category','country','platform','device_brand_name','lang','osv']
#   cf = [0,1,2,3,4,5,6,7,8]
## 预处理训练数据，针对离散数据转换为category类型
    obj_feat = list(X_data.loc[:, X_data.dtypes == 'object'].columns.values)
    for feature in obj_feat:
        X_data[feature] = pd.Series(X_data[feature], dtype="category")
    train_data = lgb.Dataset(X_data, label=Y_data,categorical_feature=cf)
## 预处理测试数据，针对离散数据转换为category类型    
    obj_feat_test = list(X_test_data.loc[:, X_test_data.dtypes == 'object'].columns.values)
    for feature in obj_feat_test:
        X_test_data[feature] = pd.Series(X_test_data[feature], dtype="category")
    test_data = lgb.Dataset(X_test_data, label=Y_test_data,categorical_feature=cf)

# 模型训练
    param = {'num_leaves': args.tree_num_leaves, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = args.num_round
    gbm = lgb.train(param, train_data, num_round, valid_sets=[test_data],categorical_feature=cf)
    joblib.dump(gbm, args.model_dir + "/classifier.pkl")