import pandas as pd
import numpy as np


def form_dataset(request_data, ml_model_artifacts,
                 na_values=['', '?']):
    
    # convert request records into a list of dicts
    request_data = [request_data] if type(request_data) == dict else request_data
    # for each record add in missing fields
    for record in request_data:
        # get list of inital data features
        feature_names = list(ml_model_artifacts['cat_init_features']) + list(ml_model_artifacts['num_init_features'])
        # get list of features missing in record
        features_not_present = list(set(feature_names) - set(record.keys()))
        # fill feature names with a missing value placeholder
        for feature in features_not_present:
            record[feature] = '?'
    
    # convert list of record dicts into a dataframe     
    request_df = pd.DataFrame(request_data)
    # convert missing value tokens to NaNs
    for token in na_values:
        request_df = request_df.replace({token : np.NaN})

    return request_df


def impute_and_encode_features(request_df, ml_model_artifacts):
    
    # separate categorical and numeric features
    categorical_features_init = ml_model_artifacts['cat_init_features']
    numeric_features_init = ml_model_artifacts['num_init_features']
    request_df_cat = request_df[categorical_features_init]
    request_df_num = request_df[numeric_features_init]
    
    # impute categorical features
    categorical_imputer = ml_model_artifacts['cat_imputer']
    request_df_cat = pd.DataFrame(categorical_imputer.transform(request_df_cat), 
                                  columns=categorical_features_init)
    
    # one-hot encode categorical features (dummy variables)
    categorical_ohe = ml_model_artifacts['dummy_encoder']
    request_df_cat_ohe = categorical_ohe.transform(request_df_cat).toarray()
    
    categorical_features_ohe = ml_model_artifacts['cat_ohe_features']
    request_df_cat_ohe = pd.DataFrame(request_df_cat_ohe, 
                                      columns=categorical_features_ohe)
    
    # impute numeric features
    numeric_imputer = ml_model_artifacts['num_imputer']
    request_df_num = pd.DataFrame(numeric_imputer.transform(request_df_num), 
                                  columns=numeric_features_init)
    
    # combine numeric and categorical features
    request_df = pd.concat([request_df_num, request_df_cat_ohe], axis=1)
    # align column names for feature set
    column_names = ml_model_artifacts['column_names_order']
    request_df = request_df[column_names]
    
    return request_df