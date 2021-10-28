import boto3

#  创建模型---start--- 此部分针对需要自带模型到Sagemaker进行推理的场景
sm = boto3.client("sagemaker")
role = 'arn:aws:iam::517141035927:role/service-role/AmazonSageMaker-ExecutionRole-20190819T155749'
container='517141035927.dkr.ecr.us-west-2.amazonaws.com/lgb-inference:latest'
model_data = 's3://sagemaker-us-west-2-517141035927/output/lightgbm-model-training-2021-10-26-12-17-52-220/output/model.tar.gz'
model_name = 'lightgbm-inference-model'
primary_container = {"Image": container, "ModelDataUrl": model_data}

lightgbm_model = sm.create_model(
    ModelName=model_name, ExecutionRoleArn=role, PrimaryContainer=primary_container
)

print(lightgbm_model["ModelArn"])
#  创建模型---end---  