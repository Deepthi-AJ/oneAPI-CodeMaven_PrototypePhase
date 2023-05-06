#!/usr/bin/env python
# coding: utf-8

# # Install Libraries

# In[ ]:


get_ipython().system('pip install dpcpp-cpp-rt')
get_ipython().system('pip install scikit-learn-intelex')
get_ipython().system('pip install intel-tensorflow')
get_ipython().system('pip install modin[all]')
get_ipython().system('pip install neural-compressor')
get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install Flask')
# !pip install pandas
# !pip install tensorflow


# # Import Libraries

# In[1]:


import os
import wfdb
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import glob

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers import Dense
from intel_neural_compressor import Compressor

import joblib

from flask import Flask, request

import warnings
warnings.filterwarnings('ignore')


# 

# # Preprocessing

# # Define data path

# In[2]:


data_path = "./ptbdb"


# # Preprocessing the data
# 
# Use bandpass filter to remove noise
# 
# Normalize signal data
# 
# Save preprocessed file and plot the data

# #### Define filter coefficients

# In[3]:


b, a = signal.butter(4, [0.01, 0.4], btype='bandpass')


# In[6]:


for patient_folder in os.listdir(data_path):
    patient_path = os.path.join(data_path, patient_folder)
    
    # Loop through each record file
    for record_file in os.listdir(patient_path):
        if record_file.endswith(".hea"):
            # Read header file to get signal names, number of signals, and sampling frequency
            record_path = os.path.join(patient_path, record_file[:-4])
            record = wfdb.rdheader(record_path)
            signal_names = record.sig_name
            num_signals = record.n_sig
            sampling_freq = record.fs

            # Read signal data from .dat file
            signal_data = wfdb.rdsamp(record_path)[0]

            # Apply bandpass filter to remove noise
            filtered_signal_data = signal.filtfilt(b, a, signal_data, axis=0)

            # Normalize signal data
            normalized_signal_data = (filtered_signal_data - np.mean(filtered_signal_data, axis=0)) / np.std(filtered_signal_data, axis=0)

            # Save preprocessed signal data to .npy file
            if not any(file.endswith(".npy") for file in os.listdir(patient_path)):
                np.save(os.path.join(record_path, "preprocessed.npy"), normalized_signal_data)


            # Convert signal data to CSV format
            if not any(file.endswith(".csv") for file in os.listdir(patient_path)):
                signal_df, fields = wfdb.rdsamp(record_path)
                signal_df = pd.DataFrame(signal_df, columns=fields['sig_name'])
                signal_df['diagnosis'] = record.comments[4].split(": ")[1]
                csv_path = os.path.join(patient_path, record_file[:-4] + ".csv")
                signal_df.to_csv(csv_path, index=False) 

            # Plot the signal data
            wfdb.plot_items(signal=signal_data, title=record_file[:-4])


# # Single Patient Data Analysis
# 
# Load the CSV file for the patient using pandas.read_csv().
# 
# Plot the ECG waveform using matplotlib.pyplot.plot().
# 
# Plot the power spectral density (PSD) of the ECG waveform using scipy.signal.welch().
# 
# Calculate the heart rate (HR) by counting the number of QRS complexes and dividing by the total duration of the ECG recording.
# 
# Calculate the mean and standard deviation of the RR interval (the time between successive R-peaks) to get an estimate of the variability of the heart rate.
# 
# Calculate the mean and standard deviation of the amplitude of the QRS complex to get an estimate of the strength of the ECG signal.

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# Load CSV file for patient
data_path = "./ptbdb"
patient_folder = "patient001"
record_file = "s0010_re"
csv_path = f"{data_path}/{patient_folder}/{record_file}.csv"
df = pd.read_csv(csv_path)
print(df.keys(), df.shape)

# Plot ECG waveform
plt.figure(figsize=(12, 4))
plt.plot(df["i"])
plt.title("ECG Waveform")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude (mV)")
plt.show()

# Calculate and plot PSD of ECG waveform
f, Pxx = signal.welch(df["i"], fs=360, nperseg=1024)
plt.figure(figsize=(12, 4))
plt.plot(f, Pxx)
plt.title("Power Spectral Density (PSD) of ECG Waveform")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.show()

# Calculate heart rate (HR)
qrs_peaks, _ = signal.find_peaks(df["i"], height=0.5)
duration = len(df) / 360  # total duration of recording in seconds
hr = len(qrs_peaks) / duration
print(f"Heart Rate: {hr:.2f} bpm")
print("The patient has been diagnosed with ",df.loc[0]["diagnosis"])

# Calculate mean and standard deviation of RR interval
rr_intervals = [qrs_peaks[i] - qrs_peaks[i-1] for i in range(1, len(qrs_peaks))]
mean_rr = sum(rr_intervals) / len(rr_intervals) / 360  # convert to seconds
std_rr = (sum((rr - mean_rr*360)**2 for rr in rr_intervals) / (len(rr_intervals) - 1) / 360)**0.5  # convert to seconds
print(f"Mean RR Interval: {mean_rr:.3f} s")
print(f"Standard Deviation of RR Interval: {std_rr:.3f} s")

# Calculate mean and standard deviation of QRS amplitude
mean_qrs_amp = df.loc[qrs_peaks, "i"].mean()
std_qrs_amp = df.loc[qrs_peaks, "i"].std()
print(f"Mean QRS Amplitude: {mean_qrs_amp:.3f} mV")
print(f"Standard Deviation of QRS Amplitude: {std_qrs_amp:.3f} mV")


# # Creating Dataframe with diagnosis data of all patients
# 
# #### Combine all the CSV files of patients with the diagnosis information from .hea files added to a column against each patient number specified in the patient folder row-wise
# 
# Create an empty DataFrame to store the combined data.
# 
# Loop through each patient folder and read the diagnosis information from the .hea file.
# 
# Loop through each CSV file in the patient folder and read the data into a DataFrame.
# 
# Add a new column to the DataFrame with the diagnosis information.
# 
# Append the DataFrame to the empty DataFrame created in step 1.
# 
# Save the combined data to a CSV file.

# In[8]:


import os
import glob

import wfdb
import pandas as pd

# Define data path
data_path = "./ptbdb"

# Create an empty DataFrame to store the combined data
combined_data = pd.DataFrame()


# Loop through each patient folder
for patient_folder in os.listdir(data_path):
    patient_path = os.path.join(data_path, patient_folder)
    
    # Read the diagnosis information from the .hea file
    diagnosis_files = glob.glob(os.path.join(patient_path, "*.hea"))
    diagnosis = []

    if diagnosis_files:
        for i in range(len(diagnosis_files)):
            diagnosis_file = diagnosis_files[i]
            with open(diagnosis_file, 'r') as f:
                diagnosis.append(f.readline().split()[-1])
    else:
        print(f"No diagnosis file found in {patient_path}")
        continue

    # Loop through each CSV file in the patient folder
    for record_file in os.listdir(patient_path):
        if record_file.endswith(".csv"):
            # Read the data into a DataFrame
            csv_path = glob.glob(os.path.join(patient_path, "*.csv"))
            print(csv_path)
            for i in range(len(csv_path)):
                
                csv_data = pd.read_csv(csv_path[i])

                # Add a new column with the diagnosis information
                csv_data['Diagnosis'] = diagnosis[i]

                # Append the DataFrame to the combined data
                combined_data = pd.concat([combined_data, csv_data])

# Save the combined data to a CSV file
if not "combined_data.csv" in os.listdir(data_path):
    combined_data.to_csv("combined_data.csv", index=False)


# In[ ]:





# # perform automatic diagnosis using hyperparameter tuned models on a PTB ECG diagnostic database and then visualize the comparative analysis using graphs

# The data is loaded and split into training and testing sets. 
# 
# Then, the code trains and evaluates different machine learning models, including logistic regression, support vector machine, decision tree, random forest, and deep neural network, with hyperparameters tuned based on the specific ECG database being used. 
# 
# The accuracy of each model is stored in a list called "model_accuracies". 
# 
# Finally, a bar plot is created to visualize the comparative analysis of the different models.

# In[ ]:


# Load ECG data from CSV file
data = pd.read_csv('data_path/combined_data.csv')

# Split data into training and testing sets
X = data.iloc[:, :-1] # Features
y = data.iloc[:, -1] # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate different machine learning models
models = {
    'Logistic Regression': LogisticRegression(C=0.1, penalty='l2', solver='newton-cg'),
    'Support Vector Machine': SVC(C=10, kernel='rbf', gamma=0.1),
    'Decision Tree': DecisionTreeClassifier(max_depth=7, min_samples_split=50),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2),
    'Deep Neural Network': Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
}

model_accuracies = []

for name, model in models.items():
    if name != 'Deep Neural Network':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_accuracy = accuracy_score(y_test, y_pred)
        model_accuracies.append(model_accuracy)
        print(f'{name} accuracy: {model_accuracy:.4f}')
    else:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
        _, accuracy = model.evaluate(X_test, y_test)
        model_accuracies.append(accuracy)
        print(f'{name} accuracy: {accuracy:.4f}')

    # Save the trained model as a .pkl file
    filename = f'{name}.pkl'
    joblib.dump(model, filename)

# Plot model accuracies
plt.bar(models.keys(), model_accuracies)
plt.title('Comparative Analysis of ECG Classification Models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()

# Print the accuracy of patient diagnosis comparison
best_model = max(model_accuracies, key=model_accuracies.get)
print(f"Best Model: {best_model}")
print(f"Accuracy: {model_accuracies[best_model]:.4f}")


# In[ ]:


# Load ECG data from CSV file
data = pd.read_csv('data_path/combined_data.csv')

# Split data into training and testing sets
X = data.iloc[:, :-1] # Features
y = data.iloc[:, -1] # Labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate different machine learning models
models = {
    'Logistic Regression': LogisticRegression(C=0.1, penalty='l2', solver='newton-cg'),
    'Support Vector Machine': SVC(C=10, kernel='rbf', gamma=0.1),
    'Decision Tree': DecisionTreeClassifier(max_depth=7, min_samples_split=50),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2),
    'Deep Neural Network': Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
}

model_accuracies = {}

for name, model in models.items():
    if name == 'Deep Neural Network':
        # Using Intel Neural Compressor (INC) for the DNN model
        compressor = Compressor(model)
        compressed_model = compressor.compress()
        compressed_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        _, accuracy = compressed_model.evaluate(X_test, y_test)
        model_accuracies[name] = accuracy
        print(f'{name} accuracy (with INC): {accuracy:.4f}')
        
        # Save the compressed model as a .pkl file
        filename = f'{name}_INC.pkl'
        joblib.dump(compressed_model, filename)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[name] = accuracy
        print(f'{name} accuracy: {accuracy:.4f}')

# Plot model accuracies
plt.bar(model_accuracies.keys(), model_accuracies.values())
plt.title('Comparative Analysis of ECG Classification Models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.show()

# Print the accuracy of patient diagnosis comparison
best_model = max(model_accuracies, key=model_accuracies.get)
print(f"Best Model: {best_model}")
print(f"Accuracy: {model_accuracies[best_model]:.4f}")


# Application

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.externals import joblib

# Load the trained model
model = joblib.load('Deep_Neural_Network_INC.pkl')

# Load the preprocessed input data
data = pd.read_csv('./pbdb/patient001/s0010_re.csv')

# Process the input data using the model
predictions = model.predict(data)

# Output the results
with open('results.txt', 'w') as f:
    for prediction in predictions:
        f.write(str(prediction) + '\n')

