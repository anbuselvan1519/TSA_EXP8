# Ex.No:08 - MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING

### Name: Anbuselvan.S
### Register No: 212223240008
### Date: 

## AIM:
To implement Moving Average Model and Exponential smoothing Using Python for yahoo stock prediction.

## ALGORITHM:
1. Import necessary libraries
2. Read the AirLinePassengers data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph.
    
## PROGRAM:
### Import the packages:
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```
### Read the Airline Passengers dataset from a CSV file:
```py
data = pd.read_csv("/content/yahoo_stock.csv")
```
### Display the shape and the first 50 rows of the dataset:
```py
print("Shape of the dataset:", data.shape)
print("First 50 rows of the dataset:")
print(data.head(50))
```
### Plot the first 50 values of the 'International' column:
```py
plt.plot(data['Volume'].head(50))
plt.title('First 50 values of the "Volume" column')
plt.xlabel('Index')
plt.ylabel('Volume')
plt.show()
```
### Perform rolling average transformation with a window size of 5:
```py
rolling_mean_5 = data['Volume'].rolling(window=5).mean()
```
### Display the first 10 values of the rolling mean:
```py
print("First 10 values of the rolling mean with window size 5:")
print(rolling_mean_5.head(10))
```
### Perform rolling average transformation with a window size of 10:
```py
rolling_mean_10 = data['Volume'].rolling(window=10).mean()
```
### Plot the original data and fitted value (rolling mean with window size 10):
```py
plt.plot(data['Volume'], label='Original Data')
plt.plot(rolling_mean_10, label='Rolling Mean (window=10)')
plt.title('Original Data and Fitted Value (Rolling Mean)')
plt.xlabel('Index')
plt.ylabel('Stock Volume')
plt.legend()
plt.show()
```
### Fit an AutoRegressive (AR) model with 13 lags:
```py
lag_order = 13
model = AutoReg(data['Volume'], lags=lag_order)
model_fit = model.fit()
```
### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF):
```py
plot_acf(data['Volume'])
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['Volume'])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```
### Make predictions using the AR model:
```py
predictions = model_fit.predict(start=lag_order, end=len(data)-1)
```
### Compare the predictions with the original data:
```py
mse = mean_squared_error(data['Volume'][lag_order:], predictions)
print('Mean Squared Error (MSE):', mse)
```
### Plot the original data and predictions:
```py
plt.plot(data['Volume'][lag_order:], label='Original Data')
plt.plot(predictions, label='Predictions')
plt.title('AR Model Predictions vs Original Data')
plt.xlabel('Index')
plt.ylabel('Stock Volume')
plt.legend()
plt.show()
```

## OUTPUT:

### Plot the original data and fitted value:
![image](https://github.com/user-attachments/assets/9a603bdf-f765-4366-b66c-ee8ffeef95ce)

### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF):
![image](https://github.com/user-attachments/assets/0fddb388-f469-45ab-ac3b-45b8fd4fdbb9)
![image](https://github.com/user-attachments/assets/f89cd35a-86a5-4697-909f-2d14658af286)

### Plot the original data and predictions:
![image](https://github.com/user-attachments/assets/9b58ac05-5990-4305-9ccf-639047ea56b9)

## RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
