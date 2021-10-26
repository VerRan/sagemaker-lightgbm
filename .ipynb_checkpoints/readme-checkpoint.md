# lightgbm 镜像构建
 login skitlear imgage account  , 这一步需要提前执行，否则无法加载到基础镜像
 1. aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 246618743249.dkr.ecr.us-west-2.amazonaws.com
 2. docker build -t training-lightgbm .
 3. docker tag training-lightgbm:latest xxxxx.dkr.ecr.us-west-2.amazonaws.com/training-lightgbm:latest
 4. docker push xxxx.dkr.ecr.us-west-2.amazonaws.com/training-lightgbm:latest  ，，2，3，4 步骤命令已自己仓库命令为准

# lightgbm 算法代码测试
python test.py

# lightgbm sagemaker entry_point改造与测试
python entry_point.py --tree_num_leaves 30 --num_round 30 --model_dir '/home/sagemaker-user/yeahmobi-bj-lightgbm'

# lightgbm集成到Sagemaker
