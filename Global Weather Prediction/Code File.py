import tensorflow 
from tensorflow import keras 
import pandas 
import numpy 

df=pandas.read_csv('GlobalWeatherRepository.csv')

df.shape

df.info()

df.isnull().sum()

df.duplicated().sum()

df

df.columns 

df.drop(columns=['country', 'location_name','sunrise',
       'sunset', 'moonrise', 'moonset', 'moon_phase', 'moon_illumination'], inplace=True)

df

for i in df.columns:
    if df[i].dtypes == 'object':
        print(i)

df['condition_text'].unique()

from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

df['wind_direction']=encoder.fit_transform(df['wind_direction'])
df['timezone']=encoder.fit_transform(df['timezone'])
df['last_updated']=encoder.fit_transform(df['last_updated'])
df['condition_text']=encoder.fit_transform(df['condition_text'])

from sklearn.preprocessing import MinMaxScaler

numerical_features = ['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 
                      'air_quality_Carbon_Monoxide', 'air_quality_PM2.5', 'air_quality_PM10']

scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])


from sklearn.model_selection import train_test_split

X = df[numerical_features]  
Y = df['temperature_celsius']  

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ANN architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  
    Dense(32, activation='relu'), 
    Dense(16, activation='relu'),  
    Dense(8,  activation='relu'),
    Dense(1, activation='linear')  
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_test, Y_test))


loss, mae = model.evaluate(X_test, Y_test)
print(f"Test MAE: {mae}")


predictions = model.predict(X_test)


import numpy as np

predictions = model.predict(X_test)

predictions = np.round(predictions, 2)  

for i in range(5): 
    print(f"Predicted: {predictions[i][0]}°C | Actual: {Y_test.iloc[i]}°C")


from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(Y_test, predictions)
mse = mean_squared_error(Y_test, predictions)
rmse = np.sqrt(mse) 

print(f"MAE: {mae}°C")    
print(f"MSE: {mse}")
print(f"RMSE: {rmse}°C")


new_data = np.array([[0.75, 0.60, 0.85, 0.20, 0.05, 0.10, 0.12]])  # Normalized input

new_data = new_data.reshape(1, -1)

# Predict temperature
predicted_temp = model.predict(new_data)
print(f"Predicted Temperature: {predicted_temp[0][0]:.2f}°C")
