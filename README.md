# Ex.No:08 - MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING

### Name: Anbuselvan.S  
### Register No: 212223240008  
### Date: 

## AIM:
To implement Moving Average Model and Exponential Smoothing using Python for Yahoo stock prediction.

## ALGORITHM:
1. Import necessary libraries.
2. Read the `yahoo_stock.csv` data from a CSV file and display the shape and the first 20 rows of the dataset.
3. Set the figure size for plots.
4. Plot the first 50 values of the 'Volume' column.
5. Perform rolling average transformation with a window size of 5.
6. Display the first 10 values of the rolling mean.
7. Perform rolling average transformation with a window size of 10.
8. Create a new figure for plotting. Plot the original data and fitted values.
9. Show the plot.
10. Also perform exponential smoothing and plot the graph.

## PROGRAM:

### Import the packages:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
```

### Load the data:
```python
data = pd.read_csv("/content/yahoo_stock.csv")
print("Shape of the dataset:", data.shape)
print("First 50 rows of the dataset:")
print(data.head(50))
```

### Plot the first 50 values of the 'Volume' column:
```python
plt.plot(data['Volume'].head(50))
plt.title('First 50 values of the "Volume" column')
plt.xlabel('Index')
plt.ylabel('Volume')
plt.show()
```

### Perform rolling average transformation with a window size of 5 and display:
```python
rolling_mean_5 = data['Volume'].rolling(window=5).mean()
print("First 10 values of the rolling mean with window size 5:")
print(rolling_mean_5.head(10))
```

### Perform rolling average transformation with a window size of 10:
```python
rolling_mean_10 = data['Volume'].rolling(window=10).mean()
```

### Plot the original data and fitted value (rolling mean with window size 10):
```python
plt.plot(data['Volume'], label='Original Data')
plt.plot(rolling_mean_10, label='Rolling Mean (window=10)')
plt.title('Original Data and Fitted Value (Rolling Mean)')
plt.xlabel('Index')
plt.ylabel('Stock Volume')
plt.legend()
plt.show()
```

### Fit a Moving Average (MA) model:
```python
order = (0, 0, 13)  # MA model with 13 lags
model = ARIMA(data['Volume'], order=order)
model_fit = model.fit()
```

### Plot Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF):
```python
plot_acf(data['Volume'])
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['Volume'])
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```

### Make predictions using the MA model:
```python
predictions = model_fit.predict(start=13, end=len(data)-1)
```

### Compare the predictions with the original data:
```python
mse = mean_squared_error(data['Volume'][13:], predictions)
print('Mean Squared Error (MSE):', mse)
```

### Plot the original data and predictions:
```python
plt.plot(data['Volume'][13:], label='Original Data')
plt.plot(predictions, label='Predictions')
plt.title('MA Model Predictions vs Original Data')
plt.xlabel('Index')
plt.ylabel('Stock Volume')
plt.legend()
plt.show()
```

### Exponential Smoothing:
```python
model_exp_smoothing = SimpleExpSmoothing(data['Volume']).fit(smoothing_level=0.2, optimized=False)
exp_smoothing_predictions = model_exp_smoothing.fittedvalues

plt.plot(data['Volume'], label='Original Data')
plt.plot(exp_smoothing_predictions, label='Exponential Smoothing')
plt.title('Exponential Smoothing Predictions vs Original Data')
plt.xlabel('Index')
plt.ylabel('Stock Volume')
plt.legend()
plt.show()
```

## OUTPUT:

### Plot the original data and fitted value:
![Plot](https://github.com/user-attachments/assets/9a603bdf-f765-4366-b66c-ee8ffeef95ce)

### Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF):
![ACF](https://github.com/user-attachments/assets/0fddb388-f469-45ab-ac3b-45b8fd4fdbb9)  
![PACF](https://github.com/user-attachments/assets/f89cd35a-86a5-4697-909f-2d14658af286)

### Plot the original data and MA model predictions:
![MA Model Predictions](https://github.com/user-attachments/assets/9b58ac05-5990-4305-9ccf-639047ea56b9)

### Plot the original data and Exponential Smoothing predictions:
![image](https://github.com/user-attachments/assets/3a664437-0672-4fbe-aadb-3034972241c1)

## RESULT:
Thus, the implementation of the Moving Average Model and Exponential Smoothing using Python for Yahoo stock prediction is successfully completed.
