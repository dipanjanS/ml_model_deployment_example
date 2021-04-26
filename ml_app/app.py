## Make module accessible ##
import os.path
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
#####

from flask import Flask, request, jsonify
from flask_cors import CORS # Cross Origin Resource Sharing (CORS), making cross-origin AJAX possible

from ml_app.modeling import ml_pipeline as mp


HEADERS = {'content-type': 'application/json'}

# Instantiate Flask App
app = Flask(__name__)
CORS(app)

# This runs as soon as we setup our web service to run
XGB_ML_ARTIFACTS = mp.load_xgb_ml_artifacts()


# Liveness test
@app.route('/income_classifier/api/v1/liveness', methods=['GET', 'POST'])
def liveness():
    return 'API Live!'


# Model 2 inference endpoint
@app.route('/income_classifier/api/v1/predict', methods=['POST'])
def xgb_model_inference():
    input_data = request.get_json(force=True)['data']
    response = mp.run_xgb_ml_pipeline(input_data, XGB_ML_ARTIFACTS)
    return jsonify(response)


# running REST interface, port=5000 for direct test
# use debug=True when debugging, NOT when deploying
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8900)