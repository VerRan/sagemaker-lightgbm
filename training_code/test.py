from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
from numpy import genfromtxt
import pandas as pd
import lightgbm as lgb

if __name__ == "__main__":
    
    X_data = pd.read_csv('/home/sagemaker-user/yeahmobi-bj-lightgbm/datasets/CR_train_x.csv')
    Y_data = pd.read_csv('/home/sagemaker-user/yeahmobi-bj-lightgbm/datasets/CR_train_y.csv')
    
    X_test_data = pd.read_csv('/home/sagemaker-user/yeahmobi-bj-lightgbm/datasets/CR_test_x.csv')
    Y_test_data = pd.read_csv('/home/sagemaker-user/yeahmobi-bj-lightgbm/datasets/CR_test_y.csv')
 
    cf = ['pkg_name','pkg_size','category','sub_category','country','platform','device_brand_name']
    
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

    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data],categorical_feature=cf)

# 模型存储
joblib.dump(bst, 'test.pkl')
# # # # 模型加载
lgb = joblib.load('test.pkl')

# 模型预测
test_data_str = 'COM.NEXT.INNOVATION.TAKATAK,OTHERS,APPLICATIONS,SOCIAL,IN,ANDROID,VIVO,OTHERS,11,3.135494,0,7,0,1109,157,3,0.01910828,0,-20.72326584,-20.72326584,-20.72326584,-20.72326584,-3.792754,-3.032013'
y_pred = bst.predict(test_data_str)
y_pred

# # 模型评估
# print('The accuracy of prediction is:', accuracy_score(y_test, y_pred))

# # 特征重要度
# print('Feature importances:', list(lgb.feature_importances_))
    
    
