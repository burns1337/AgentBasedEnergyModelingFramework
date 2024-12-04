import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Scripts.utils import get_eplus_output_for_single_lstm


def fetch_energy_prices(file_path):
    df = pd.read_csv(file_path, delimiter=',', index_col=0, parse_dates=True)
    return df

def preprocess_data(df, feature_col):
    data = df[feature_col].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(train, valid, predictions, rmse, mae, mape):
    plt.figure(figsize=(16, 8))
    plt.title(f'Indoor Temperature Prediction with a single feature LSTM Model (RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%)', fontsize=16)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Mean Air Temperature [C](Hourly)', fontsize=15)
    plt.plot(train['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'])
    plt.plot(valid[['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')
    plt.grid(True)
    plt.show()

def main():
    # directory above the current working directory
    eplus_out_df = get_eplus_output_for_single_lstm()
    # eplusout_temp_col = eplus_out_df.pop('THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)')
    print(eplus_out_df)
    # file_path = Path(Path().cwd().parent, 'data', 'strompreise_08_2024.csv')
    # energy_prices_df = fetch_energy_prices(file_path)

    # Ensure the datetime format is correctly parsed and set as the index
    eplus_out_df.index = pd.to_datetime(eplus_out_df.index, format='%d.%m.%Y %H:%M:%S')

    # Preprocess the data
    # feature_col = energy_prices_df.columns.tolist()
    feature_col = 'THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'
    data, scaler = preprocess_data(eplus_out_df, feature_col)

    # Create the dataset
    time_step = 1
    X, y = create_dataset(data, time_step)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    best_rmse = float('inf')
    best_model_path = 'best_model.keras'

    for i in range(10):
        model = build_lstm_model((X_train.shape[1], 1))
        checkpoint_filepath = 'best_model.keras'
        checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min')

        model.fit(X_train, y_train, batch_size=1, epochs=1, validation_data=(X_test, y_test), callbacks=[checkpoint_callback])
        model.load_weights(checkpoint_filepath)

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        train = eplus_out_df[:train_size]
        valid = eplus_out_df[train_size:train_size + len(predictions)]
        valid['Predictions'] = predictions.flatten()
        # print(valid)

        # Calculate error metrics
        mae = mean_absolute_error(valid[feature_col], valid['Predictions'])
        mse = mean_squared_error(valid[feature_col], valid['Predictions'])
        rmse = np.sqrt(mse)
        actual_values = valid[feature_col].values
        non_zero_indices = actual_values != 0
        mape = np.mean(np.abs((actual_values[non_zero_indices] - predictions.flatten()[non_zero_indices]) / actual_values[non_zero_indices])) * 100

        if rmse < best_rmse:
            best_rmse = rmse
            best_mae = mae
            best_mape = mape
            best_predictions = predictions
            # model.save(best_model_path)

        print(f'Run {i+1}: MAE: {mae}, RMSE: {rmse}, MAPE: {mape}')

    print(f'Best Model: MAE: {best_mae}, RMSE: {best_rmse}, MAPE: {best_mape}')
    plot_predictions(train, valid, best_predictions, best_rmse, best_mae, best_mape)

if __name__ == '__main__':
    main()