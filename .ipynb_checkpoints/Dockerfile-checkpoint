FROM 246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3

COPY requirements.txt /requirements.txt
RUN pip install --no-cache -r /requirements.txt && \
    rm /requirements.txt
    
# additional container dependencies can be added here