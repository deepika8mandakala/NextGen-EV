# NextGen-EV
Electric Vehicle Compatibility Analyzer &amp; Diagnostic Suite Designed ML models for predicting EV compatibility based on user preferences and diagnosing potential issues for predictive maintenance.
# NextGen-EV: AI-Powered EV Analytics Platform

## Objective

The objective is to build an EV analytics platform that uses machine learning to assist users and manufacturers in:

* Analyzing vehicle performance
* Detecting faults
* Performing predictive maintenance
* Recommending suitable EVs

## Why Electric Vehicles?

Electric Vehicles are the future of mobility, but they face challenges such as:

* Fire incidents
* Component failures
* Limited support infrastructure

This platform aims to leverage AI and IoT to make EVs safer and more intelligent.

## Key Features

* Manufacturing defect detection
* Predictive maintenance using IoT sensor data
* Anomaly detection
* EV recommendation system
* Multi-user support for both manufacturers and consumers

## Modules / Pages

* Index page
* User / Manufacturer login
* User dashboard (Profile, Diagnosis, Notifications)
* Diagnosis tools
* Review page
* Backend ML model integration

## Data & Preprocessing

### Data Used

The platform uses simulated or real IoT EV data, including:

* Temperature
* Vibration
* Voltage
* Current
* Battery health

### Preprocessing Steps

* Handling of null values
* Feature scaling using StandardScaler
* Encoding categorical variables
* Outlier removal
* Optional feature selection

### Missing Value Handling

Approaches include forward fill, mean imputation, or conditional removal based on feature importance.

## Machine Learning Models Used

* Random Forest for defect classification
* Isolation Forest for anomaly detection
* Logistic Regression for health prediction
* LSTM / Autoencoders for time-series maintenance

### Model Rationale

* Random Forest is robust to noisy, non-linear data and avoids overfitting.
* Isolation Forest efficiently detects outliers in an unsupervised setting.

## Training & Validation

### Data Split

The data is split into 80% training and 20% testing using train\_test\_split.

### Evaluation Metrics

* For classification: Accuracy, Precision, Recall, F1-score
* For regression: RMSE, MAE
* Binary models: Confusion Matrix, AUC-ROC

### Validation Strategy

K-Fold Cross Validation ensures model generalization.

## System Architecture & Backend Integration

### ML Integration

A Python backend (FastAPI or Flask) loads the trained ML models and exposes REST API endpoints.

### Real-Time Data Handling

IoT sensor data is sent via API, and the backend processes the data to return predictions on faults or system health.

### Backend Libraries Used

* scikit-learn
* pandas
* joblib
* FastAPI or Flask
* uvicorn or gunicorn

## Predictive Maintenance

Predictive maintenance uses historical and real-time data to estimate when a component might fail, allowing proactive intervention. Time-series models like LSTM or Random Forest Regressor are used depending on the data structure.

## Anomaly Detection

### Definition

Anomalies refer to significant deviations in sensor readings, such as sudden voltage spikes or temperature rises, which may indicate a problem.

### Techniques Used

* Isolation Forest
* One-Class SVM

## EV Recommendation System

EV recommendations are based on:

* User location
* Average travel range
* Usage frequency
* Preferences

The system uses classification or recommender algorithms to suggest the most suitable EVs.

## Frontend Interaction

Users interact with the models through:

* Manual form inputs
* Real-time sensor integration

The backend responds with diagnostic results such as "Battery needs service" or "Safe to drive."

## Deployment Strategy

* Host ML backend on cloud platforms like AWS or GCP
* Integrate with real-time IoT data pipelines
* Provide dashboards for both users and manufacturers

## Error Handling

All inputs are validated on both frontend (JavaScript) and backend (Python) before being passed to ML models.

## Benefits

The system provides early warnings and predictive alerts, helping reduce the risk of EV fires or major failures.

## Future Improvements

* Use real-world EV data
* Integrate deep learning models like CNN for image diagnostics
* Add mobile app compatibility
* Incorporate live GPS tracking and vehicle monitoring

## System Architecture Flowchart:
![nextgen-ev](https://github.com/user-attachments/assets/31209e23-0690-4942-a156-86bb031526f1)
[EV Platform Flowchart]



## Repository URL

[https://github.com/deepika8mandakala/NextGen-EV](https://github.com/deepika8mandakala/NextGen-EV)

---

Developed by:
Team [TECHNOVA]
- Deepika Mandakala
- Ruchira Nalluri
- Divya Sai Veeravalli
- Vyshnavi Manam
