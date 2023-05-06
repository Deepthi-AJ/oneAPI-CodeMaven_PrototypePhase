# Create a new Python file, for example, app.py, and import the necessary libraries:

from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the trained models from the .pkl files:

model_lr = joblib.load('Logistic Regression.pkl')
model_svm = joblib.load('Support Vector Machine.pkl')
model_dt = joblib.load('Decision Tree.pkl')
model_rf = joblib.load('Random Forest.pkl')
model_dnn = joblib.load('Deep Neural Network.pkl')

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
            diagnosis_lr = model_lr.predict(data)
            diagnosis_svm = model_svm.predict(data)
            diagnosis_dt = model_dt.predict(data)
            diagnosis_rf = model_rf.predict(data)
            diagnosis_dnn = model_dnn.predict(data)

            # Render the results template with the diagnosis
            return render_template('results.html',
                                   diagnosis_lr=diagnosis_lr,
                                   diagnosis_svm=diagnosis_svm,
                                   diagnosis_dt=diagnosis_dt,
                                   diagnosis_rf=diagnosis_rf,
                                   diagnosis_dnn=diagnosis_dnn)
    return render_template('index.html')


# Create the index.html and results.html templates in a templates folder.

# Run the Flask application:

if __name__ == '__main__':
    app.run()


# In OneAPI JupyterLab, open a terminal and navigate to the folder containing app.py and run the command python app.py to start the Flask application.

# Access the web application by visiting the provided URL in the terminal.

# Make sure you have the necessary HTML templates (index.html and results.html) in a templates folder in the same directory as app.py. You may need to adjust the code and HTML templates according to your specific requirements.

# This implementation allows users to upload an ECG data file, and the Flask application uses the trained models to perform diagnosis and displays the results on the results page.



