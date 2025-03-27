# Global-Weather-Prediction
ANN Global Weather Prediction
This repository contains an Artificial Neural Network (ANN) model built with TensorFlow/Keras to predict temperature based on weather data. The dataset used is GlobalWeatherRepository.csv, which includes various meteorological parameters such as temperature, humidity, pressure, wind speed, and air quality metrics like CO, PM2.5, and PM10. The dataset also contains categorical features like weather conditions and timestamps, which have been encoded for model training.

The project involves several steps, including data preprocessing, model training, and evaluation. The preprocessing phase handles missing values, removes duplicates, encodes categorical variables, and scales numerical data using MinMaxScaler. The ANN model is built with five layers using ReLU activation functions and trained with the Adam optimizer to minimize Mean Squared Error (MSE). The performance of the model is evaluated using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

To use this project, clone the repository using Git and install the required dependencies using pip. The dataset is preprocessed before training the model, which learns to predict temperature based on the input features. Once the model is trained, it is tested on unseen data to evaluate its performance. Predictions can be made by providing new weather data as input, and the model outputs the estimated temperature.

An example prediction can be made using a normalized input array representing weather conditions, and the model predicts the corresponding temperature. This project demonstrates the application of deep learning in weather prediction and can be further improved with additional features and larger datasets.
