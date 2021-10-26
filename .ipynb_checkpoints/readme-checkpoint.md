# lightgbm 算法代码测试
先本地测试代码逻辑
python test.py

# lightgbm sagemaker 集成入口类改造与测试
主要关注参数的设置与传递，改造后可以通过Sagemaker 超参数设置进行传递，模型输出路径等
下面是本地测试改造后的代码
python entry_point.py --tree_num_leaves 30 --num_round 30 --model_dir '/home/sagemaker-user/yeahmobi-bj-lightgbm'

# lightgbm集成到Sagemaker
集成到sagemaker进行训练和推理
参见action-lightgbm.ipynb