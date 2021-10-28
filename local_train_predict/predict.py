from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib
from numpy import genfromtxt
import pandas as pd
import lightgbm as lgb

X_test_data = pd.read_csv('/home/sagemaker-user/sagemaker-lightgbm/datasets/CR_test_x.csv')
## 预处理测试数据，针对离散数据转换为category类型    
obj_feat_test = list(X_test_data.loc[:, X_test_data.dtypes == 'object'].columns.values)
for feature in obj_feat_test:
    X_test_data[feature] = pd.Series(X_test_data[feature], dtype="category")
# # # # 模型加载
lgb = joblib.load('test.pkl')

# 模型预测
y_pred = lgb.predict(X_test_data)
print(y_pred)