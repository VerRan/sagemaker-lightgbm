import sys
import os
os.system('pip install joblib pathlib lightgbm numpy==1.20.1 pandas==1.3.4')

import numpy as np
from pathlib import Path
import json
import joblib
import warnings

def model_fn(model_dir):
    model_dir = Path(model_dir)
    # load regressor
    lgb = joblib.load(Path(model_dir, "classifier.pkl"))
    
    return lgb


def assert_json(content_type):
    assert (
        content_type == "application/json"
    ), "content_type must be 'application/json'"


def input_fn(request_body_str, request_content_type):
    request = {
        'data': json.loads(request_body_str),
    }
    return request

def predict_fn(request, model):
    data = request['data']
    print('data:'+str(data))
    response = {}

    response['predictions'] = model.predict(data['inputs']).tolist()
#     data = request['data']
#     entities = request['entities']
#     response = {}
#     if 'data' in entities:
#         response['data'] = data
#     features = preprocess_fn(data, model_assets)
#     if 'features' in entities:
#         feature_names = model_assets["features_schema"].item_titles
#         feature_values = features[0].tolist()
#         response['features'] = {k: v for k, v in zip(feature_names, feature_values)}
#     if 'descriptions' in entities:
#         response['descriptions'] = model_assets["features_schema"].item_descriptions_dict
#     if 'prediction' in entities:
#         prediction = model_assets["classifier"].predict_proba(features)
#         # take first sample (idx=0)
#         # and second probability (idx=1) corresponding to the positive class
#         response['prediction'] = prediction[0][1].tolist()
#     if ('explanation_shap_values' in entities) or ('explanation_shap_interaction_values' in entities):
#         explanation = {}
#         expected_value = model_assets["explainer"].expected_value
#         # see https://github.com/slundberg/shap/issues/729: handle both cases
#         if expected_value.shape == (1,):
#             explanation['expected_value'] = expected_value[0].tolist()
#         else:
#             explanation['expected_value'] = expected_value[1].tolist()
#         if 'explanation_shap_values' in entities:
#             # second probability (idx=1) corresponding to the positive class
#             # and take first sample (idx=0)
#             feature_names = model_assets["features_schema"].item_titles
#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore")
#                 shap_values = model_assets["explainer"].shap_values(features)[1][0]
#             explanation['shap_values'] = {k: v for k, v in zip(feature_names, shap_values.tolist())}
#         if 'explanation_shap_interaction_values' in entities:
#             labels = model_assets["features_schema"].item_titles
#             # take first sample (idx=0)
#             values = model_assets["explainer"].shap_interaction_values(features)[0].tolist()
#             explanation['shap_interaction_values'] = {
#                 'labels': labels,
#                 'values': values
#             }
#         # see https://github.com/slundberg/shap/issues/729: setting back to original
#         model_assets["explainer"].expected_value = expected_value
#         response['explanation'] = explanation
    return response


def output_fn(response, response_content_type):
    assert (
        response_content_type == "application/json"
    ), "accept must be 'application/json'"
    print('response:' + str(response))
    return response