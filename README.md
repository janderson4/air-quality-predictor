# LSTM Machine Learning Air Quality Prediction

## Overview

This project implements a Long Short-Term Memory (LSTM) neural network using MATLAB's Deep Learning Toolbox to predict PM2.5 levels and other environmental variables crucial for urban health. The model forecasts future values based on historical data from a hybrid sensor network, evaluating both short-term and long-term air quality predictions.

## Key Features

- **LSTM Implementation**: Utilizes MATLAB Deep Learning Toolbox to train LSTM networks.
- **Multi-Variable Input**: Capable of handling multiple input features such as PM2.5 concentrations, humidity, temperature, and vehicle speed.
- **Hybrid Forecasting Approach**: Combines open and closed loop methods for more accurate predictions:
  - **Open-loop Forecasting**: Utilizes actual values for updating the network between time steps.
  - **Closed-loop Forecasting**: Updates the network with its own predictions when true values are unavailable.
- **Prediction Types**:
  - **Short-term Predictions**: Outputs PM2.5 values for the next 3 hours based on 3 days of historical data.
  - **Long-term Predictions**: Estimates PM2.5 levels for the next 24 hours based on 7 days of data.
  - **Interpolation Predictions**: Estimates values at 5-minute intervals for an hour over a span of 3 days.

## Objectives

The primary goal of this project is to develop a machine learning algorithm that accurately predicts air quality levels, specifically PM2.5. The project aims to:

- Identify and calibrate an effective model for analyzing air pollution data.
- Provide actionable insights on air quality trends to city officials for better urban health management.
- Utilize heterogeneous data collected from mobile and static sensors to improve prediction accuracy.

## Usage

1. **Requirements**:
   - MATLAB with Deep Learning Toolbox
   - Data from a hybrid sensor network (mobile and static sensors)

2. **Installation**:
   - Clone this repository:
     ```bash
     git clone https://github.com/janderson4/air-quality-prediction.git
     ```
   - Navigate to the project directory:
     ```bash
     cd air-quality-prediction
     ```

3. **Running the Model**:
   - Use the MATLAB script `train_lstm_model.m` to train the LSTM network and make predictions.
   - Adjust parameters as necessary based on your dataset and prediction requirements.

## Conclusion

Implementing this LSTM-based prediction model demonstrates the potential for machine learning in managing urban air quality issues. By analyzing and predicting PM2.5 levels effectively, city officials can develop targeted interventions to mitigate pollution and improve public health.

## References

[1] MathWorks. (2024). Time Series Forecasting Using Deep Learning. MathWorks. Retrieved from https://www.mathworks.com/help/deeplearning/ug/time-series-forecasting-using-deep-learning.html  
