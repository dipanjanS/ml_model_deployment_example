## Make module accessible ##
import os.path
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
#####



data = [{'education': 'HS-grad',
  'workclass': '?',
  'occupation': '?',
  'capital.gain': 0,
  'capital.loss': 4356,
  'hours.per.week': 40},
 {'education': 'HS-grad',
  'workclass': 'Private',
  'native.country': 'United-States',
  'sex': 'Female',
  'education.num': 9,
  'race': 'White',
  'occupation': 'Exec-managerial',
  'capital.gain': 0,
  'capital.loss': 4356,
  'marital.status': 'Widowed',
  'relationship': 'Not-in-family',
  'hours.per.week': 18,
  'age': 82},
 {'education': 'Some-college',
  'workclass': '',
  'native.country': 'United-States',
  'sex': 'Female',
  'education.num': 10,
  'race': '?',
  'occupation': '?',
  'capital.gain': 0,
  'capital.loss': 4356,
  'marital.status': 'Widowed',
  'relationship': 'Unmarried',
  'hours.per.week': '?',
  'age': 66}]

print(data)

from ml_app.modeling import ml_pipeline as mp

ml_artifacts = mp.load_xgb_ml_artifacts()
print(mp.run_xgb_ml_pipeline(data, ml_artifacts))