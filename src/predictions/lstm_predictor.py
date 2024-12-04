import numpy as np
import pandas
import matplotlib.pyplot as plt
from docutils.nodes import target
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from joblib import dump
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


from utils import get_eplus_output
# from Scripts.utils import get_eplus_output


class LSTM_Predictor:
    def __init__(self, args):
        self.y_test = None
        self.X_test = None
        self.X_train = None
        self.y_train = None
        self.output_dir = args.output if args.output else "output"
        self.eplusout_df = get_eplus_output()
        self.preprocess_eplus_output_df()
        self.model = None
        self.scaler_target = None
        self.scaled_train_data = None
        self.num_features = None
        self.test_seq_timestamps = None
        self.predictions = None

    def preprocess_eplus_output_df(self):
        self.eplusout_df.drop(
            labels='THERMAL ZONE 1:Zone Thermal Comfort ASHRAE 55 Simple Model Summer Clothes Not Comfortable Time [hr](Hourly)',
            axis=1)
        temp_col = self.eplusout_df.pop('THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)')
        # Append it back to the DataFrame, making it the last column
        self.eplusout_df['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'] = temp_col

    def train_lstm_with_lookback(self, lookback=12, horizon=12):
        """ Train LSTM model with lookback. """
        targets = ['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)']
        features = self.eplusout_df.columns.drop(targets).to_list()
        # self.num_features = len(features)
        df = self.eplusout_df.copy()

        # x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(df[features], df[targets], test_size=0.2,
        #                                                                 shuffle=False)

        scaler = MinMaxScaler()
        scaler_target = MinMaxScaler()
        train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)

        train_timestamps = train_df.index
        test_timestamps = test_df.index
        _, y_train_scaling, train_seq_timestamps = self.create_sequences(train_df.values, train_timestamps, lookback,
                                                                         horizon)
        scaler_target.fit_transform(y_train_scaling)
        self.scaled_train_data = scaler.fit_transform(train_df)
        self.X_train, self.y_train, train_seq_timestamps = self.create_sequences(self.scaled_train_data, train_timestamps,
                                                                                 lookback, horizon)

        scaled_test_data = scaler.transform(test_df)
        # dump(scaler, 'std_scaler.bin', compress=True)
        # dump(scaler_target, 'std_scaler_target.bin', compress=True)
        self.X_test, self.y_test, test_seq_timestamps = self.create_sequences(scaled_test_data, test_timestamps,
                                                                              lookback, horizon)

        self.num_features = self.X_train.shape[2]
        self.X_train = self.X_train.reshape((self.X_train.shape[0], lookback, self.num_features))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], lookback, self.num_features))

        model = Sequential()
        model.add(LSTM(100, return_sequences=True, input_shape=(lookback, self.num_features)))
        model.add(LSTM(50))
        model.add(Dense(horizon))  # Predicting next 12 steps
        model.compile(optimizer='adam', loss='mse')
        self.model = model
        self.scaler_target = scaler_target
        self.scaler_train = scaler
        self.test_seq_timestamps = test_seq_timestamps
        # print(f"!! lookback was: {lookback}")

        # history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
        # return model, X_train, y_train, X_test, y_test
        self.plot_lstm_training_and_validation_loss(self.X_train, self.y_train)

    def create_sequences(self, data, timestamps, lookback, horizon):
        X, y, ts = [], [], []
        for i in range(len(data) - lookback - horizon + 1):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback:i + lookback + horizon, -1])
            ts.append(timestamps[i + lookback + horizon - 1])
        return np.array(X), np.array(y), np.array(ts)

    def rolling_window_cross_validation_lstm(self, n_splits=4, lookback=6, horizon=12):
        """Perform Rolling Window Cross-Validation for LSTM model."""
        errors = []

        # Initialize Time Series Cross-Validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for split_num, (train_index, test_index) in enumerate(tscv.split(self.eplusout_df)):
            # Split data into train and test sets
            train_df = self.eplusout_df.iloc[train_index]
            test_df = self.eplusout_df.iloc[test_index]

            targets = ['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)']
            features = self.eplusout_df.columns.drop(targets).to_list()

            # Initialize and fit scalers
            scaler = MinMaxScaler()
            scaler_target = MinMaxScaler()

            # Fit the feature scaler on the training set and transform both train and test sets
            scaled_train_data = scaler.fit_transform(train_df[features])
            scaled_test_data = scaler.transform(test_df[features])

            # Fit the target scaler only on the training target column
            train_target_scaled = scaler_target.fit_transform(train_df[targets])

            # Create sequences for training
            train_timestamps = train_df.index
            test_timestamps = test_df.index
            X_train, y_train, _ = self.create_sequences(scaled_train_data, train_timestamps, lookback, horizon)

            # Reshape y_train to 2D for scaling, then back to original shape after scaling
            y_train_reshaped = y_train.reshape(-1, 1)
            y_train_scaled = scaler_target.transform(y_train_reshaped).reshape(y_train.shape)

            # Create sequences for testing (ensure consistent scaling)
            X_test, y_test, _ = self.create_sequences(scaled_test_data, test_timestamps, lookback, horizon)

            # Reshape y_test to 2D for scaling, then back to original shape after scaling
            y_test_reshaped = y_test.reshape(-1, 1)
            y_test_scaled = scaler_target.transform(y_test_reshaped).reshape(y_test.shape)

            # Reshape the input data for LSTM compatibility
            num_features = X_train.shape[2]
            X_train = X_train.reshape((X_train.shape[0], lookback, num_features))
            X_test = X_test.reshape((X_test.shape[0], lookback, num_features))

            # Build the LSTM model
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(lookback, num_features)),
                LSTM(50),
                Dense(horizon)
            ])
            model.compile(optimizer='adam', loss='mse')

            # Train the model on the training data
            history = model.fit(X_train, y_train_scaled, epochs=20, batch_size=32, verbose=0, validation_split=0.2)

            # Make predictions on the test set
            y_pred = model.predict(X_test)

            # Rescale predictions and test targets back to original scale
            y_test_rescaled = scaler_target.inverse_transform(y_test_scaled.reshape(-1, 1)).reshape(y_test.shape)
            y_pred_rescaled = scaler_target.inverse_transform(y_pred.reshape(-1, 1)).reshape(y_pred.shape)

            # Calculate RMSE for this split
            mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
            rmse = np.sqrt(mse)
            errors.append(rmse)
            print(f'RMSE for split {split_num + 1}: {rmse:.4f}')

            # Plot training and validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Split {split_num + 1} - Training and Validation Loss')
            plt.legend()
            plt.show()

            # Plot predictions vs actuals and training data for the current split
            plt.figure(figsize=(10, 5))

            # Plot training data
            plt.plot(train_df.index, train_df[targets].values, label="Training Data", color='green')

            # Plot actual test data
            plt.plot(test_df.index[:len(y_test_rescaled)], y_test_rescaled[:, 0], label="Actual Test Data",
                     color='blue')

            # Plot predictions
            plt.plot(test_timestamps[:len(y_pred_rescaled)], y_pred_rescaled[:, 0], label="Predicted Test Data",
                     color='orange')

            # Add labels and title for clarity
            plt.title(f'Split {split_num + 1} - Predictions vs Actuals vs Training Data')
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            plt.legend()
            plt.show()

        # Calculate and display average RMSE across all splits
        average_rmse = np.mean(errors)
        print(f'Average RMSE over all splits: {average_rmse:.4f}')


    def rolling_window_cross_validation_lstm_old(self, n_splits=4, lookback=6, horizon=12):
        """Perform Rolling Window Cross-Validation for LSTM model."""
        errors = []

        # Initialisierung der Zeitreihen-Kreuzvalidierung
        tscv = TimeSeriesSplit(n_splits=n_splits)

        for split_num, (train_index, test_index) in enumerate(tscv.split(self.eplusout_df)):            # Aufteilung der Daten in Train- und Test-Set basierend auf den Indizes
            train_df = self.eplusout_df.iloc[train_index]
            test_df = self.eplusout_df.iloc[test_index]

            targets = ['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)']
            features = self.eplusout_df.columns.drop(targets).to_list()

            # Initialisierung und Anpassung der Skaler
            scaler = MinMaxScaler()
            scaler_target = MinMaxScaler()

            # Target-Skalierung und Sequenzbildung für das Training
            train_timestamps = train_df.index
            test_timestamps = test_df.index

            _, y_train_scaling, _ = self.create_sequences(train_df[features].values, train_timestamps, lookback,
                                                          horizon)
            scaler_target.fit_transform(y_train_scaling)

            scaled_train_data = scaler.fit_transform(train_df[features])
            X_train, y_train, _ = self.create_sequences(scaled_train_data, train_timestamps, lookback, horizon)

            # Target-Skalierung und Sequenzbildung für das Testen
            scaled_test_data = scaler.transform(test_df[features])
            X_test, y_test, _ = self.create_sequences(scaled_test_data, test_timestamps, lookback, horizon)

            # LSTM-Modell vorbereiten und an die Eingabegröße anpassen
            num_features = X_train.shape[2]
            X_train = X_train.reshape((X_train.shape[0], lookback, num_features))
            X_test = X_test.reshape((X_test.shape[0], lookback, num_features))

            # Erstellen des LSTM-Modells
            model = Sequential()
            model.add(LSTM(100, return_sequences=True, input_shape=(lookback, num_features)))
            model.add(LSTM(50))
            model.add(Dense(horizon))  # Horizon = Schritte, die vorhergesagt werden
            model.compile(optimizer='adam', loss='mse')

            # Training des Modells auf den aktuellen Trainingsdaten
            history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.2)


            # Vorhersagen auf dem aktuellen Test-Set und Berechnung des MSE
            y_pred = model.predict(X_test)
            y_test_rescaled = scaler_target.inverse_transform(y_test)
            y_pred_rescaled = scaler_target.inverse_transform(y_pred)

            mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
            rmse = np.sqrt(mse)
            errors.append(rmse)
            print(f'RMSE for split {split_num + 1}: {rmse:.4f}')

            # train test validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Split {split_num + 1} - Training and Validation Loss')
            plt.legend()
            plt.show()

            # Plot für den aktuellen Split
            plt.figure(figsize=(10, 5))
            # plt.plot(test_timestamps[:len(y_test_rescaled)], y_test_rescaled[:, 0], label="Actual")
            plt.plot(train_df.index, train_df[targets].values, label="Training Data", color='green')
            # Plot actual test data in another color (e.g., blue)
            plt.plot(test_df.index[:len(test_df)], test_df[targets].values, label="Actual Test Data", color='blue')
            plt.plot(test_timestamps[:len(y_pred_rescaled)], y_pred_rescaled[:, 0], label="Prediction", color='orange')

            plt.title(f'Split {split_num + 1} - Predictions vs Actuals')
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            plt.legend()
            plt.show()

        # Durchschnittlicher MSE über alle Splits
        average_rmse = np.mean(errors)
        print(f'Average RMSE over all splits: {average_rmse:.4f}')


    # def rolling_window_cross_validation_lstm(self, n_splits=5, lookback=14 * 24, horizon=7 * 24):
    def rolling_window_cross_validation_lstm_wrong(self, n_splits=4, lookback=14 * 24, horizon=7 * 24):
        """Perform Rolling Window Cross-Validation for LSTM model with 2 weeks train, 1 week test."""
        errors = []

        # Define number of samples per training (2 weeks) and testing (1 week)
        train_size = 14 * 24  # Assuming hourly data
        test_size = 7 * 24  # 1 week test

        # Loop over splits
        for split_num in range(n_splits):
            start_train = split_num * test_size
            end_train = start_train + train_size
            start_test = end_train
            end_test = start_test + test_size

            train_df = self.eplusout_df.iloc[start_train:end_train]
            test_df = self.eplusout_df.iloc[start_test:end_test]

            targets = ['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)']
            features = self.eplusout_df.columns.drop(targets).to_list()

            # Initialize scalers
            scaler = MinMaxScaler()
            scaler_target = MinMaxScaler()

            # Scale the training and target columns
            scaled_train_data = scaler.fit_transform(train_df[features])
            scaled_train_target = scaler_target.fit_transform(train_df[targets])

            # Create sequences for training
            X_train, y_train, _ = self.create_sequences(scaled_train_data, train_df.index, lookback, horizon)

            # Scale test data with the same scalers
            scaled_test_data = scaler.transform(test_df[features])
            scaled_test_target = scaler_target.transform(test_df[targets])

            # Create sequences for testing
            X_test, y_test, _ = self.create_sequences(scaled_test_data, test_df.index, lookback, horizon)

            # Reshape inputs for LSTM
            num_features = X_train.shape[2]
            X_train = X_train.reshape((X_train.shape[0], lookback, num_features))
            X_test = X_test.reshape((X_test.shape[0], lookback, num_features))

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(lookback, num_features)))
            model.add(LSTM(50))
            model.add(Dense(horizon))
            model.compile(optimizer='adam', loss='mse')

            # Train the model
            history = model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0, validation_split=0.2) # save this in a variable to plot the training and validation loss


            # Make predictions and inverse scale
            y_pred = model.predict(X_test)
            y_test_rescaled = scaler_target.inverse_transform(y_test)
            y_pred_rescaled = scaler_target.inverse_transform(y_pred)

            # Calculate and store RMSE
            mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
            rmse = np.sqrt(mse)
            errors.append(rmse)
            print(f'RMSE for split {split_num + 1}: {rmse:.4f}')

            # history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
            plt.figure(figsize=(10, 5))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Split {split_num + 1} - Training and Validation Loss')
            plt.legend()
            plt.show()

            # Plot predictions vs actuals for the current split
            plt.figure(figsize=(10, 5))
            # plt.plot(test_df.index[:len(y_test_rescaled)], y_test_rescaled[:, 0], label="Actual")
            # plt.plot(test_df.index[:len(y_pred_rescaled)], y_pred_rescaled[:, 0], label="Prediction")
            # plt.title(f'Split {split_num + 1} - Predictions vs Actuals')
            # Plot training data in one color (e.g., green)
            plt.plot(train_df.index, train_df[targets].values, label="Training Data", color='green')
            # Plot actual test data in another color (e.g., blue)
            plt.plot(test_df.index[:len(test_df)], test_df[targets].values, label="Actual Test Data", color='blue')
            # plt.plot(test_df.index[:len(y_test_rescaled)], y_test_rescaled[:, 0], label="Actual Test Data", color='blue')
            # Plot predicted test data in a third color (e.g., orange)
            plt.plot(test_df.index[:len(y_pred_rescaled)], y_pred_rescaled[:, 0], label="Predicted Test Data",
                     color='orange')
            # Add labels, title, and legend for better visualization
            plt.title(f'Split {split_num + 1} - Predictions vs Actuals vs Training Data')
            plt.xlabel('Time')
            plt.ylabel('Temperature')
            plt.legend()
            plt.show()

        # Average RMSE over all splits
        average_rmse = np.mean(errors)
        print(f'Average RMSE over all splits: {average_rmse:.4f}')


    def run_lstm_prediction(self):
        self.train_lstm_with_lookback(lookback=12)
        print(f"\nPerforming LSTM prediction with file \n {self.eplusout_df} ...")
        # self.plot_train_and_test_data()
        self.plot_lstm_prediction()
        # self.rolling_window_cross_validation_lstm_old(n_splits=4, lookback=6, horizon=12)  #  lookback=6, horizon=12) # lookback=14*24, horizon=7*24

    def plot_lstm_training_and_validation_loss(self, X_train, y_train):
        history = self.model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()

    def plot_lstm_prediction(self):
        predictions = self.model.predict(self.X_test)
        # Inverse transform the predictions
        # predictions_inv_transformed = self.scaler_target.inverse_transform(predictions)

        predictions_full = np.zeros((predictions.shape[0], self.num_features))
        predictions_full[:, -1] = predictions[:, 0]
        predictions_transformed = self.scaler_train.inverse_transform(predictions_full)[:, -1]

        y_test_full = np.zeros((self.y_test.shape[0], self.num_features))
        y_test_full[:, -1] = self.y_test[:, 0]
        true_test_values = self.scaler_train.inverse_transform(y_test_full)[:, -1]

        plt.figure(figsize=(15, 5))
        plt.plot(self.test_seq_timestamps.flatten(), true_test_values.flatten(), label='True Values')
        plt.plot(self.test_seq_timestamps.flatten(), predictions_transformed.flatten(), label='Predictions',
                 linestyle='dashed')
        plt.xlabel('Time')
        plt.ylabel('Temperature [C]')
        plt.title('True Temperature vs Predictions')
        plt.legend()
        # plt.gcf().autofmt_xdate()
        plt.show()

        # make df out of timestemps and predictions, with the column name predictions, and index the timestamps
        df_predictions_transformed = pandas.DataFrame(predictions_transformed, index=self.test_seq_timestamps,
                                                      columns=['predictions'])
        # add column datetime with the index
        df_predictions_transformed['datetime'] = df_predictions_transformed.index
        
        # df_predictions_transformed = pandas.DataFrame(predictions_transformed, index=self.test_seq_timestamps)
        self.predictions = df_predictions_transformed  # predictions_transformed #.flatten()

        # print(f"y_test: {true_test_values}")
        # print(f"y_test flatten: {true_test_values.flatten()}")
        # print(f"predictions: {predictions_transformed}")
        # print(f"predictions_transformed flatten: {predictions_transformed.flatten()}")
        # print(f"test_seq_timestamps flatten: {self.test_seq_timestamps.flatten()}")

    def plot_train_and_test_data(self):
        # print(f"X_train shape: {self.X_train.shape}")
        # print(f"y_train shape: {self.y_train.shape}")
        # print(f"X_test shape: {self.X_test.shape}")

        plt.figure(figsize=(15, 5))
        # plt.plot(self.test_seq_timestamps.flatten(), self.y_test.flatten(), label='True Values')
        plt.plot(self.test_seq_timestamps.flatten(), self.y_test, label='True Values')
        # plt.plot(self.test_seq_timestamps.flatten(), self.X_test[:, -1].flatten(), label='Predictions', linestyle='dashed')
        plt.plot(self.test_seq_timestamps.flatten(), self.predictions.flatten(), label='Predictions', linestyle='dashed')
        plt.xlabel('Time')
        plt.ylabel('Temperature [C]')
        plt.title('True Temperature vs Predictions')
        plt.legend()
        plt.show()
