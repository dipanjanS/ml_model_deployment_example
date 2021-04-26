
def make_xgb_model_predictions(request_df, ml_model_artifacts):
    
    # load saved ML model
    ml_model = ml_model_artifacts['xgb_model']
    
    # make model predictions
    predictions = ml_model.predict(request_df)
    
    # return predictions
    return {
        'predicted_classes' : list(predictions)
    }