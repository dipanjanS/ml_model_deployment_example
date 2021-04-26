from ml_app.utils.config import  XGB_ML_ARTIFACT_PATH
from ml_app.utils.ml_model_management import load_model_artifacts
from ml_app.data_processing.pre_processor import form_dataset, impute_and_encode_features
from ml_app.modeling.ml_inference import make_xgb_model_predictions



def load_xgb_ml_artifacts(path=XGB_ML_ARTIFACT_PATH):
    
    # 1. Load model artifacts
    ml_artifacts = load_model_artifacts(path=path)
    return ml_artifacts



def run_xgb_ml_pipeline(request_data, ml_artifacts):
    
    # 2. Create request dataset
    request_df = form_dataset(request_data=request_data,
                              ml_model_artifacts=ml_artifacts)
    
    # 3. Impute and Encode Features
    request_df = impute_and_encode_features(request_df=request_df, 
                                            ml_model_artifacts=ml_artifacts)
    
    # 4. Load and make ML model predictions
    pred_response = make_xgb_model_predictions(request_df=request_df, 
                                           ml_model_artifacts=ml_artifacts)
    
    # return response
    return pred_response