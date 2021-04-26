import os

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) # get location of parent dir automatically
XGB_ML_ARTIFACT_PATH = PARENT_DIR+'/saved_models/census_xgb_artifacts.pkl'


MODEL_HISTORY = {
    'version_1': {
        'model_type' : 'XGBoost',
        'model_artifact_location': XGB_ML_ARTIFACT_PATH
    }
}
    