import sys
import os
os.system('pip install joblib pathlib lightgbm numpy==1.20.1 pandas==1.3.4')

import numpy as np
from pathlib import Path
import json
import joblib
import warnings
import pandas as pd
# from sagemaker_inference import content_types, decoder
## https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html 函数说明

def model_fn(model_dir):
    model_dir = Path(model_dir)
    # load regressor
    lgb = joblib.load(Path(model_dir, "classifier.pkl"))
    
    return lgb

def input_fn(request_body_str, request_content_type):
    print(str(request_body_str))
    data_lst = request_body_str.split('\n')
    request = [list(i.split(',')) for i in data_lst[:-1]]
    df  = pd.DataFrame(request)
    for idx, feature in enumerate(range(len(df.columns))):
        if idx < 9:
            df[feature] = pd.Series(df[feature], dtype="category")
        else:
            df[feature] = pd.Series(df[feature], dtype="float")
    return df

def predict_fn(request, model):
    response = model.predict(request)
    return response

## 输出与输入必须对齐，否则transoformer的过滤就无法对齐
def output_fn(response, response_content_type):
    response = [str(i) for i in response]
    response = "\n".join(response)
    print('response:' + str(response))
    return response