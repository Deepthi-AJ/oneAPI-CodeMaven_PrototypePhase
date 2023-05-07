# Import the necessary libraries:

from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the trained models from the .pkl files:

# model_lr = joblib.load('Logistic Regression.pkl')
# model_svm = joblib.load('Support Vector Machine.pkl')
# model_dt = joblib.load('Decision Tree.pkl')
# model_rf = joblib.load('Random Forest.pkl')
model_dnn = joblib.load('deep_model.pkl')
model_xgb = joblib.load('xgb_model.pkl')
model_xgb = joblib.load('daal_model.pkl')


# Create a Flask application instance:

app = Flask(__name__)

# Define a route for the home page where users can upload the ECG data file:

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            data = pd.read_csv(file)
            # Perform diagnosis using the models
#             diagnosis_lr = model_lr.predict(data)
#             diagnosis_svm = model_svm.predict(data)
#             diagnosis_dt = model_dt.predict(data)
#             diagnosis_rf = model_rf.predict(data)
            diagnosis_dnn = model_dnn.predict(data)
            diagnosis_xgb = model_xgb.predict(data)
            diagnosis_daal = model_daal.predict(data)

            # Render the results template with the diagnosis
            return render_template('results.html',
                                   diagnosis_dnn = diagnosis_dnn,
                                   diagnosis_daal = diagnosis_daal,
                                   diagnosis_xgb = diagnosis_xgb)
#                                    diagnosis_lr=diagnosis_lr,
#                                    diagnosis_svm=diagnosis_svm,
#                                    diagnosis_dt=diagnosis_dt,
#                                    diagnosis_rf=diagnosis_rf,
#                                    diagnosis_dnn=diagnosis_dnn)
    return render_template('index.html')

# Run the Flask application:

if __name__ == '__main__':
    app.run()
