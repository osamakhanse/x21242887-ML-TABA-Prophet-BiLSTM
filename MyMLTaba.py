#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np
import tensorflow as tf
import pandas as pd
from fbprophet import Prophet
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy import stats

def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['dteday'])
    data = data.groupby(['dteday', 'hr']).agg({'cnt': 'sum'}).reset_index()
    data = data.loc[data.groupby('dteday')['cnt'].idxmax()]
    data = data.set_index('dteday')
    data = data.resample('W').agg({'cnt': 'max', 'hr': 'first'})
    data = data.reset_index()
    return data

def remove_outliers(data, threshold=3):
    z_scores = np.abs(stats.zscore(data['cnt']))
    data_clean = data[z_scores < threshold]
    return data_clean

def plot_data_before_after_outlier_removal(data, data_clean):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    ax[0].plot(data['dteday'], data['cnt'], 'o-', label='Before Outlier Removal')
    ax[0].set_xlabel('Date')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Bike Rental Count Before Outlier Removal')
    ax[0].legend()
    
    ax[1].plot(data_clean['dteday'], data_clean['cnt'], 'o-', label='After Outlier Removal')
    ax[1].set_xlabel('Date')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Bike Rental Count After Outlier Removal')
    ax[1].legend()
    
    plt.show()




def train_prophet(data):
    model = Prophet()
    model.fit(data)
    return model

def train_bilstm(data, look_back, n_features, epochs, batch_size):
    def create_dataset(dataset, look_back):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)

    trainX, trainY = create_dataset(dataset, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], n_features))

    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='tanh'), input_shape=(look_back, n_features)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    optimizer = Adam(clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='mse')

    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)

    return model, scaler


def create_future_dataframe(data, start_date, end_date):
    future_dates = pd.date_range(start_date, end_date, freq='W')
    future = pd.DataFrame({'ds': future_dates})
    return future

def predict_prophet(model, future):
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

def predict_bilstm(model, scaler, data, look_back, n_features):
    data = np.reshape(data, (1, look_back, n_features))
    pred = model.predict(data)
    return scaler.inverse_transform(pred)[0, 0]

def calculate_weights(prophet_forecast, bilstm_forecast):
    prophet_forecast = prophet_forecast[~np.isnan(bilstm_forecast)]
    bilstm_forecast = bilstm_forecast[~np.isnan(bilstm_forecast)]
    
    model = Ridge(alpha=1.0)
    model.fit(prophet_forecast.reshape(-1, 1), bilstm_forecast)
    return model.coef_

def combined_forecast(prophet_forecast, bilstm_forecast, weights):
    combined = np.zeros_like(prophet_forecast)
    for i in range(len(prophet_forecast)):
        if np.isnan(bilstm_forecast[i]):
            combined[i] = prophet_forecast[i]
        else:
            combined[i] = prophet_forecast[i] * (1 - weights) + bilstm_forecast[i] * weights
    return combined


def plot_results(forecast_prophet, forecast_bilstm, forecast_combined, dates, hours):
    plt.figure(figsize=(16, 8))
    
    bar_width = 0.25
    x_pos = np.arange(len(dates))
    
    plt.bar(x_pos - bar_width, forecast_prophet, width=bar_width, label='Prophet', alpha=0.6)
    plt.bar(x_pos, forecast_bilstm, width=bar_width, label='BiLSTM', alpha=0.6)
    plt.bar(x_pos + bar_width, forecast_combined, width=bar_width, label='Combined', alpha=0.6)
    
    for i, (date, hour, cnt) in enumerate(zip(x_pos + bar_width, hours, forecast_combined)):
        plt.text(date, cnt, f"{hour}: {cnt:.2f}", rotation=90, ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('Bike Rental Count Predictions')
    plt.legend()
    plt.xticks(x_pos, [f"{d.date()} {h:02d}:00" for d, h in zip(dates, hours)], rotation=45)
    plt.tight_layout()
    plt.show()




def main():
    # Load data
    file_path = "hour.csv"
    data = load_data(file_path)
    data = data[data['dteday'] < '2012-01-01']

    # Remove outliers and display data before and after outlier removal
    data_clean = remove_outliers(data)
    plot_data_before_after_outlier_removal(data, data_clean)

    # Train Prophet model
    prophet_data = data_clean[['dteday', 'cnt']].rename(columns={'dteday': 'ds', 'cnt': 'y'})
    prophet = train_prophet(prophet_data)

    # Train BiLSTM model
    bilstm_data = data_clean['cnt'].values.reshape(-1, 1)
    look_back, n_features = 24, 1
    epochs, batch_size = 50, 32
    bilstm_model, scaler = train_bilstm(bilstm_data, look_back, n_features, epochs, batch_size)

    # Create future dataframe for 2012
    future = create_future_dataframe(data, pd.Timestamp('2012-01-01'), pd.Timestamp('2012-12-31'))

    # Predict for 2012 using Prophet
    forecast_prophet = predict_prophet(prophet, future)
    forecast_prophet = forecast_prophet.set_index('ds')['yhat']

    # Predict for 2012 using BiLSTM
    forecast_bilstm = []
    for i in range(0, len(future)):
        future_week = future.iloc[i]
        pred = predict_bilstm(bilstm_model, scaler, bilstm_data[-look_back:], look_back, n_features)
        forecast_bilstm.append(pred)
        bilstm_data = np.append(bilstm_data, np.array(pred).reshape(-1, 1), axis=0)

    forecast_bilstm = pd.Series(forecast_bilstm, index=future['ds'])

    # Calculate weights
    weights = calculate_weights(forecast_prophet.values.reshape(-1, 1), forecast_bilstm.values.reshape(-1, 1))
    print("Weights:", weights)

    # Combine forecasts
    forecast_combined = combined_forecast(forecast_prophet, forecast_bilstm, weights)

    # Print data
    print("Prophet forecast:")
    print(forecast_prophet)
    print("\nBiLSTM forecast:")
    print(forecast_bilstm)
    print("\nCombined forecast:")
    print(forecast_combined)

    # Plot predictions
    plot_results(forecast_prophet, forecast_bilstm, forecast_combined, future['ds'], data['hr'])

main()



# In[ ]:





# In[ ]:




