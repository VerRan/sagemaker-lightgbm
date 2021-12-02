# LightGBM 算法框架运行在Amazon Sagemaker POC

## 背景

在机器学习普遍之前，人们常常想到的方法就是基于规则来解决业务问题，比如在营销场景通过经验来抽象出规则，用于提高投放精准度，比如我们需要将某个物品的广告投放给能够获取高转换率的客户，这里就可以考虑哪些客户可能对要投放的广告感兴趣，比如通过SQL方式圈出对应人群，针对大数据集和复杂特征的场景可以通过大数据框架进行用户画像的构建，然后进行客户选取。但是在当前机器学习应用越来越广泛的背景下，更多的公司选择或者尝试使用机器学习来优化基于原有规则的模式，一方面随着时间的推移基于规则的系统越发复杂难以维护，另一方面基于规则的系统效果已经达到瓶颈，通过机器学习的创新方法来提升效果。在此背景下本文将介绍如何使用lightgbm以及如何借助Amazon Sagemaker来提高使用lightgbm的机器学习效率。

## LightGBM介绍

LightGBM 是一个梯度提升框架，使用的是基于树的学习算法。LightGBM的优势是旨在分布式和高效，具有以下优点：

* 训练速度更快，效率更高。
* 较低的内存使用率。
* 更好的准确性。
* 支持并行、分布式和 GPU 学习。
* 能够处理大规模数据

介绍来源：LightGBM站点（https://lightgbm.readthedocs.io/en/latest/）
同样基于树的算法框架如XGBoost，针对XGboost如果需要在Amazon Sagemaker上运行可以直接使用 Amazon Sagemaker提供的托管算法（https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html），针对LightGBM当前Sagemaker还没有托管的算法，本文将重点介绍如何通过BYOC（Bring Your Own Container）的方式也就是自定义镜像的方式来实现在Sagemaker上运行LightGBM。

## Sagemaker介绍

Amazon SageMaker 是一项完全托管的机器学习服务。借助 SageMaker，数据科学家和开发人员可以快速轻松地构建和训练机器学习模型，然后将它们直接部署到生产就绪的托管环境中。它提供了一个集成的 Jupyter 创作笔记本实例，可轻松访问您的数据源以进行探索和分析，因此您无需管理服务器。它还提供了常见的机器学习算法，这些算法经过优化，可以在分布式环境中针对超大数据高效运行。凭借对自带算法和框架的原生支持，SageMaker 提供灵活的分布式训练选项，可根据您的特定工作流程进行调整。通过从 SageMaker Studio 或 SageMaker 控制台点击几下启动模型，将模型部署到安全且可扩展的环境中。训练和推理按使用分钟数计费，没有最低费用和提前预存费用。Sagemaker详细介绍参见（https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html）

## Lightgbm本地测试

* 环境搭建

Mac下的安装方法

brew install lightgbm

其他环境安装参见：https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html

本文将使用LightGBM Python SDK的方式进行介绍，首先需要安装lightgbm的python依赖

pip install lightgbm

* 输入格式

当前LightGBM Python 模块可以从以下位置加载数据：

* LibSVM（索引从零开始）/TSV/CSV 格式的文本文件
* NumPy 2D 数组、pandas DataFrame、H2O DataTable 的 Frame、SciPy 稀疏矩阵
* LightGBM 二进制文件
* LightGBM 序列对象

本文将使用CSV作为输入进行介绍，参数设置训练推理部分的详细使用和说明参见：https://lightgbm.readthedocs.io/en/latest/Python-Intro.html

## 需求分析

针对此次试验已开始背景介绍为假设场景，当前的数据包含要推广的商品信息以及商品的推广记录，我们已推广记录为样本数据，同时针对推广是否成功作为标签数据，数据格式假设如下：

商品ID 商品名称 用户ID 用户名称 年龄 城市 是否成功
    1       A  1      张三   30  西安 是

通过如上数据训练一个模型，后续使用时当需要推广一款产品前，可以通过对应的商品和计划推广的人群信息数据集为输入进行批量推理，从而会得出该商品对应每个人的投放成功概率，再输出的结果集合中选取推广概率高的客户集合进行推广。由于该问题是一个二分类问题，因此选择lightgbm的二分类参数，同时我们使用auc作为评估指标，这个会指导后续我们的超惨输配置。

## 本地训练

下面是本地的训练代码，lightgbm框架可以直接支持直接使用category特征直接作为输入，这样就不需要针对离散特征进行one-hot编码了，使用上更加便利，需要注意的是针对category特征需要在训练阶段需要通过categorical_feature指定对应的列。针对超惨输设置部分此次实验使用的是二分类算法来解决
``
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
import lightgbm as lgb

if __name__ == "__main__":
    
    X_data = pd.read_csv('/home/sagemaker-user/sagemaker-lightgbm/datasets/CR_train_x.csv')
    Y_data = pd.read_csv('/home/sagemaker-user/sagemaker-lightgbm/datasets/CR_train_y.csv')
    
    X_test_data = pd.read_csv('/home/sagemaker-user/sagemaker-lightgbm/datasets/CR_test_x.csv')
    Y_test_data = pd.read_csv('/home/sagemaker-user/sagemaker-lightgbm/datasets/CR_test_y.csv')
 
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

# 超惨设置&模型训练
    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data],categorical_feature=cf)

# 模型存储
joblib.dump(bst, 'classifier.pkl')
    
    

超惨设置，objective指定选择的具体目标比如是二分类，多分类还是回归问题，metric用于指定评估指标，num_round定义训练的轮次每个轮次会已全量数据集作为输入进行训练，lgb.train会触发具体的训练过程。

# 超惨设置&模型训练
    param = {'num_leaves': 31, 'objective': 'binary'}
    param['metric'] = 'auc'
    num_round = 10
    bst = lgb.train(param, train_data, num_round, valid_sets=[test_data],categorical_feature=cf)

``

下来我们运行一下如上代码通过命令行执行 python train.py
``
bash-4.2$ python train.py 
[LightGBM] [Info] Number of positive: 3, number of negative: 2376
[LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000277 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 803
[LightGBM] [Info] Number of data points in the train set: 2379, number of used features: 14
/opt/conda/lib/python3.7/site-packages/lightgbm/basic.py:1780: UserWarning: Overriding the parameters from Reference Dataset.
  _log_warning('Overriding the parameters from Reference Dataset.')
/opt/conda/lib/python3.7/site-packages/lightgbm/basic.py:1513: UserWarning: categorical_column in param dict is overridden.
  _log_warning(f'{cat_alias} in param dict is overridden.')
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.001261 -> initscore=-6.674561
[LightGBM] [Info] Start training from score -6.674561
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[1]     valid_0's auc: 1
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[2]     valid_0's auc: 1
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[3]     valid_0's auc: 1
[4]     valid_0's auc: 1
[5]     valid_0's auc: 1
[6]     valid_0's auc: 1
[7]     valid_0's auc: 1
[8]     valid_0's auc: 1
[9]     valid_0's auc: 1
[10]    valid_0's auc: 1
``

训练完成后会将模型保存为.pkl文件，该文件存储了训练好的模型数据信息（
PKL 文件是由 pickle 创建的文件，pickle 是一个 Python 模块，可以将对象序列化为磁盘上的文件，并在运行时反序列化回程序。它包含一个表示对象的字节流）。

## 推理

针对该实验推理的过程就是使用业务数作为输入，通过模型进行计算和预测输出可能的标签值概率。
推理代码如下：

``
import joblib
import pandas as pd
import lightgbm as lgb

X_test_data = pd.read_csv('/home/sagemaker-user/sagemaker-lightgbm/datasets/CR_test_x.csv')
## 预处理测试数据，针对离散数据转换为category类型    
obj_feat_test = list(X_test_data.loc[:, X_test_data.dtypes == 'object'].columns.values)
for feature in obj_feat_test:
    X_test_data[feature] = pd.Series(X_test_data[feature], dtype="category")
# # # # 模型加载
lgb = joblib.load('classifier.pkl')

# 模型预测
y_pred = lgb.predict(X_test_data)
print(y_pred)

命令行执行：python predict.py

ash-4.2$ python predict.py
[0.00059869 0.00117996 0.0004638 0.00046388 0.00046389 0.00046388
0.00046388 0.00046388 0.00046389 0.00046389 0.00046388 0.00046389
0.00046389 0.00046388 0.00046388 0.00046388 0.00046388 0.00046386
0.00046385 0.00046389 0.00042613 0.00046388 0.00046388 0.00100174
0.00046388 0.000449 0.00046389 0.00046388 0.00046388 0.00056108
0.00046388 0.00058866 0.00040568 0.00045066 0.00120037 0.0004514
0.00046389 0.00046389 0.00046388 0.0017976 0.00059873 0.00059872
0.00042908 0.00043004 0.00046388 0.00046385 0.00046388 0.00046388
0.00046388 0.00046389 0.00046389 0.00046388 0.00046389 0.0232526]
``
当前本地测试lightgbm成功，下来我们看一下如何将训练和推理的过程集成到Sagemaker并利用Sagemaker 来提高训练和推理效率。

## 如何在Sagemaker上运行Lightgbm

Sagemaker除了可以通过内置算法来使用之外，也可以使用自定义镜像的方式将自己的代码和框架集成到Sagemaker中，下文将重点进行介绍。

### 构建lightgbm运行环境的镜像

首先我们需要将lightgbm框架打包成镜像并上传到ECR中，这里使用AWS 提供的内置Sklearn镜像作为基础镜像来构建步骤如下：

1. 使用ECR仓库提示的命令进行镜像构建和上传（https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-console.html） (https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-console.html%EF%BC%89)
2. 运行步骤1中ECR提示命令构建镜像

docker file 内容如下：

FROM 246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3

COPY requirements.txt /requirements.txt
RUN pip install --no-cache -r /requirements.txt && \
    rm /requirements.txt

### 训练

训练前需要根据自己的数据和镜像地址做如下修改：

1. 替换image_uri 为自己上步骤构建的镜像地址
2. entry_point 指定训练代码的位置，训练代码同本地训练代码，需要注意的是增加超参数的传递
3. output_path 替换为模型输出位置
4. 注意data_channels 数据替换为自己桶存储数据的位置

hyperparameters = {
    "tree_num_leaves": 31,
    "num_round": 5
}

_estimator = SKLearn(
    image_uri='517141035927.dkr.ecr.us-west-2.amazonaws.com/training-lightgbm:latest',
    entry_point='entry_point.py',
    source_dir='training_code',
    hyperparameters=hyperparameters,
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.c5.xlarge',
    output_path='s3://sagemaker-us-west-2-517141035927/output',
    base_job_name='lightgbm-model-training',
    disable_profiler=True
)
data_channels = {
    'x_train': 's3://sagemaker-us-west-2-517141035927/dataset/CR_train_x.csv',
    'y_train': 's3://sagemaker-us-west-2-517141035927/dataset/CR_train_y.csv',
    'x_test': 's3://sagemaker-us-west-2-517141035927/dataset/CR_test_x.csv',
    'y_test': 's3://sagemaker-us-west-2-517141035927/dataset/CR_test_y.csv'
                }
_estimator.fit(data_channels)

### 训练代码调整

针对上文提到的训练代码超参数设置部分主要是针对模型存储路径和lightgbm的超参数设置需要通过sagemaker的超惨输设置来传递，这个步骤是可选的也可以将相关参数在训练代码中设置为了代码的配置解耦以及后续的超参数优化更好的利用sagemaker可以将参数进行提取。修改后的代码如下：

import os
import sys

import argparse
import joblib

import numpy as np 
import pandas as pd 

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
    
    # prepare datasets for model training with sagemaker ec2 local path
    X_data = pd.read_csv('/opt/ml/input/data/x_train/CR_train_x.csv')
    Y_data = pd.read_csv('/opt/ml/input/data/y_train/CR_train_y.csv')
    X_test_data = pd.read_csv('/opt/ml/input/data/x_test/CR_test_x.csv')
    Y_test_data = pd.read_csv('/opt/ml/input/data/y_test/CR_test_y.csv')
 
    cf = ['pkg_name','pkg_size','category','sub_category','country','platform','device_brand_name','lang','osv']
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

从代码可以看到读区训练和测试数据的路径是从/opt/ml/input/data/读取的，这个是sagemaker启动训练后开启的训练机器的默认训练数据存储路径sagemaker会自动将存储在S3的数据下载到该目录用于训练，只需要将这里的文件名称修改为自己的训练数据名称可，如果是批量数据可以指定文件目录，关于不同训练数据输入模式的详细说明参见：https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo-running-container.html

### 批量推理

针对此次实验场景推理部分采用批量推理，Sagemaker提供了简单易用，功能强大的批量推理能力，只需要通过Sagemaker Python SDK启动批量推理任务就可以了，同时针对推理的输入的处理，输出的处理还提供了挂钩函数用于定制，此外批量推理函数还提供了输入和输出的过滤功能，比如当您需要将推理结果与输入数据关联时只需要在该方法配置output_filter就可以了。下面针对批量推理代码和过程进行详细说明：
首先需要根据训练好的模型数据，构建批量推理所需模型该模型包含了训练好的模型数据，推理的挂钩函数等，这里我们使用与训练阶段相同的Sklearn版本进行构建。

import sagemaker
from sagemaker.sklearn import SKLearnModel
model_data = "s3://sagemaker-us-west-2-517141035927/output/lightgbm-model-training-2021-10-27-12-17-13-171/output/model.tar.gz"
_model = SKLearnModel(
    model_data=model_data,
    role=sagemaker.get_execution_role(),
    entry_point='entry_point.py',
    source_dir='batch_code',
    framework_version = '0.20.0',
    py_version='py3'
)

关于SklearnMode类的详细使用方法参见 https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html

推理挂钩函数包含四个方法
model_fn：用于模型加载，本示例需要加载model_data 中的classifier.pkl文件，名称与训练阶段保存的模型名称需保持一致
input_fn：解序列化输入数据用于传入模型，本示例在推理之前针对输入数据中的离散数据转换类型为category
predict_fn：使用input_fn输出的数据作为数据放入加载的模型进行推理，然后返回推理结果
output_fn：序列化模型推理的结果并返回，本示例为了返回结果能够与输入进行关联返回将返回数据用换行符换行，从而可以与输入数据对齐，如果无法对齐推理关联输入时会报错。

import sys
import os
os.system('pip install joblib pathlib lightgbm numpy==1.20.1 pandas==1.3.4')

import numpy as np
from pathlib import Path
import json
import joblib
import warnings
import pandas as pd
## https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html 推理预处理函数说明

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

定义批量任务实现批量数据推理，下面我们需要指定用于推理的数据存储路径，推理结果的输出路径，推理所需的机器配置和数量，同时也可以指定推理最大并发数（_max_concurrent_transforms），最大的推理输入数据大小已M为单位（_max_payload）参数配置详情参见：https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-batch-code.html。同时针对转换任务的创建参数说明参见：https://sagemaker.readthedocs.io/en/stable/api/inference/model.html。

from sagemaker import get_execution_role
from time import strftime, gmtime
sagemaker_session = sagemaker.Session()

role = get_execution_role()
region = sagemaker_session.boto_session.region_name

prediction_data_path = 's3://sagemaker-us-west-2-517141035927/dataset/validate_data.csv'
out_predict_data_path = 's3://sagemaker-us-west-2-517141035927/output/'

_instance_type = 'ml.c5.4xlarge'
_instance_count = 1

_max_concurrent_transforms = 1
_max_payload = 10

_job_name = 'lightgbm-batch-{}'.format(strftime("%Y-%m-%d-%H-%M-%S", gmtime()))
print(_job_name)

lightgbm_transformer = _model.transformer(
                            instance_count=_instance_count,
                            instance_type=_instance_type,
                            strategy = 'MultiRecord',
                            max_concurrent_transforms=_max_concurrent_transforms,
                            max_payload=_max_payload,
                            output_path=out_predict_data_path,
                            assemble_with='Line',
                            accept='text/csv')


批量推理所需的模型转换任务模版创建好后，就可以启动转换任务了，这里需要设置输入数据的存储位置，类型，分割方式，输入过滤器，关联的数据来源，输出过滤器等。本示例设置为使用原有输入数据不过滤，同时输出数据与输入数据关联，同时输出时只关联输入数据的第一列进行返回。
transform方法的参数配置详见https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html

lightgbm_transformer.transform(
    data=prediction_data_path,
    content_type='text/csv',
    split_type='Line',
    input_filter="$", 
    join_source="Input",
    output_filter="$[0 ,-1]",
    job_name=_job_name
)
lightgbm_transformer.wait()

备注：如上代码块均包含在lightgbm-sagemaker.ipynb (https://github.com/VerRan/sagemaker-lightgbm/blob/main/lightgbm-sagemaker.ipynb)，该文件可以通过sagemaker notebook实例或者sagemaker studio进行运行。
代码运行后，会在sagemaker 控制台推理→批量转换任务菜单对应页面中看到任务的执行情况，同时可以在该界面进行任务监控和执行日志查看。
[Image: image.png]任务执行成功后可以通过如上界面设置页面看到输出结果的存放目录，点击就可以进入S3查看推理结果，设置页的配置信息与如上代码设置的信息是一致的，下图为推理结果输出位置配置信息：
[Image: image.png]通过S3查看推理结果，为了方便查看可以使用S3的select功能直接查看推理结果，显示如下：
[Image: image.png]

## 总结

本文介绍了lightgbm的使用以及如何运行在Sagemaker中，同时针对Sagemaker的批量推理部分做了详细说明。在实际的使用中可以根据自己的场景和需要对函数的输入参数进行修改，此外针对输入的数据集也可以目录这样便于大数据集的训练和推理，同时针对标签数据和特征数据也可以放置在一个文件中通过lightgbm进行区分。本文尽管是针对lightgbm框架做的集成介绍，实践中也可以将其他框架或者您自己的算法通过类似的方式进行集成，从而使用Sagemaker的特性来加速和优化您的机器学习过程。

# 代码使用方法
## lightgbm 算法代码测试
先本地测试代码逻辑
python test.py

## lightgbm sagemaker 集成入口类改造与测试
主要关注参数的设置与传递，改造后可以通过Sagemaker 超参数设置进行传递，模型输出路径等
下面是本地测试改造后的代码
python entry_point.py --tree_num_leaves 30 --num_round 30 --model_dir '/home/sagemaker-user/yeahmobi-bj-lightgbm'

## lightgbm集成到Sagemaker
集成到sagemaker进行训练和推理
参见action-lightgbm.ipynb


