import boto3
import sagemaker
import sagemaker.amazon.common as smac
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.transformer import Transformer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import CSVSerializer
import pandas as pd
import numpy as np
import scipy.sparse
import os
import json
import sagemaker
from time import strftime, gmtime

sess = sagemaker.Session()

## 预处理推理数据，针对离散数据转换为category类型    
# X_test_data = pd.read_csv('/home/sagemaker-user/sagemaker-lightgbm/datasets/CR_test_x.csv')
# obj_feat_test = list(X_test_data.loc[:, X_test_data.dtypes == 'object'].columns.values)
# for feature in obj_feat_test:
#     X_test_data[feature] = pd.Series(X_test_data[feature], dtype="category")
# X_test_data.to_csv(r'/home/sagemaker-user/sagemaker-lightgbm/datasets/v_data.csv')
##---- 如上代码后续剥离到预处理部分----


#  创建模型---start--- 此部分针对需要自带模型到Sagemaker进行推理的场景
# sm = boto3.client("sagemaker")
# # role = 'arn:aws:iam::517141035927:role/service-role/AmazonSageMaker-ExecutionRole-20190819T155749'
# # container='517141035927.dkr.ecr.us-west-2.amazonaws.com/training-lightgbm:latest'
# # model_data = 's3://sagemaker-us-west-2-517141035927/output/lightgbm-model-training-2021-10-26-12-17-52-220/output/model.tar.gz'
# model_name = 'my-lightgbm'
# # primary_container = {"Image": container, "ModelDataUrl": model_data}

# # lightgbm_model = sm.create_model(
# #     ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer=primary_container
# # )

# # print(lightgbm_model["ModelArn"])
#  创建模型---end---  


## 根据模型创建批量转换任务  ，也可以根据模型名称加载模型
model_name = 'lightgbm-inference-model'
lightgbm_test_transformer = Transformer(
    model_name,
    1,
    "ml.m4.xlarge",
    output_path="s3://sagemaker-us-west-2-517141035927/output",
    sagemaker_session=sess,
    strategy="MultiRecord",
    assemble_with="Line",
)

## 输入必须是s3的地址 
v_data = 's3://sagemaker-us-west-2-517141035927/dataset/v_data.csv'
lightgbm_test_transformer.transform(v_data, content_type="text/csv", split_type="Line")
lightgbm_test_transformer.wait()