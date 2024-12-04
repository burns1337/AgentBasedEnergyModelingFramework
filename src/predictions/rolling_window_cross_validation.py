import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from pathlib import Path
from datetime import timedelta


def get_eplus_output_for_single_lstm():
    file_path = Path(Path().cwd().parent, '../output/eplusout.csv')
    with open(file_path) as f:
        eplus_output_df = pd.read_csv(f, header=0, index_col=0)
    eplus_output_df["THERMAL ZONE 1:Zone People Occupant Count [](Hourly)"] = eplus_output_df["THERMAL ZONE 1:Zone People Occupant Count [](Hourly)"].round()  # make occupancy count natural numbers
    eplus_output_df = convert_eplus_index_to_datetime(eplus_output_df)
    return eplus_output_df


def get_eplus_output():
    file_path = Path(Path().cwd().parent, 'src/output/eplusout.csv')  # .parent
    with open(file_path) as f:
        eplus_output_df = pd.read_csv(f, header=0, index_col=0)
    eplus_output_df["THERMAL ZONE 1:Zone People Occupant Count [](Hourly)"] = eplus_output_df["THERMAL ZONE 1:Zone People Occupant Count [](Hourly)"].round()  # make occupancy count natural numbers
    eplus_output_df = convert_eplus_index_to_datetime(eplus_output_df)
    return eplus_output_df


def preprocess_eplus_output_df(eplusout_df):
        eplusout_df.drop(
            labels='THERMAL ZONE 1:Zone Thermal Comfort ASHRAE 55 Simple Model Summer Clothes Not Comfortable Time [hr](Hourly)',
            axis=1)
        temp_col = eplusout_df.pop('THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)')
        # Append it back to the DataFrame, making it the last column
        eplusout_df['THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)'] = temp_col
        return eplusout_df


def convert_eplus_index_to_datetime(df, year=2022):
    """
    Convert the index of a DataFrame from EnergyPlus output time format to standard datetime format.

    EnergyPlus output time format example: ' 07/22  15:00:00'
    Desired datetime format: '2022-07-22 15:00:00'

    Parameters:
    - df: pandas.DataFrame
        The DataFrame whose index needs to be converted. The index should be in EnergyPlus output time format.
    - year: int, optional (default=2022)
        The year to prepend to the datetime. Adjust if the year is different.

    Returns:
    - pandas.DataFrame
        The DataFrame with the index converted to standard datetime format.

    Notes:
    - Handles the special case wh ere the time part is '24:00:00' by converting it to '00:00:00'
      and incrementing the day by one.

    Example usage:
    df = pd.read_csv('your_data.csv', index_col=0)
    df = convert_index_to_datetime(df)
    print(df)
    """

    def convert_to_datetime(date_str):
        # Removing leading and excessive internal spaces
        date_str = date_str.strip()
        parts = date_str.split()
        if len(parts) == 2:
            month_day = parts[0].replace('/', '-')
            time_part = parts[1]

            # Check if time part is '24:00:00' and handle it
            if time_part == '24:00:00':
                time_part = '00:00:00'
                # Convert to datetime and add one day
                datetime_obj = pd.to_datetime(f'{year}-{month_day} {time_part}', format='%Y-%m-%d %H:%M:%S')
                datetime_obj += timedelta(days=1)
            else:
                # Convert to datetime directly
                datetime_obj = pd.to_datetime(f'{year}-{month_day} {time_part}', format='%Y-%m-%d %H:%M:%S')

            return datetime_obj
        else:
            raise ValueError(f"Unexpected date string format: {date_str}")

    # Apply the conversion function to the index
    df.index = df.index.map(convert_to_datetime)
    return df


# Beispiel: Erstellen einer Zeitreihe mit Pandas (ersetze dies mit deinen eigenen Daten)
# date_rng = pd.date_range(start='2022-01-01', end='2022-04-01', freq='D')
# data = pd.DataFrame(date_rng, columns=['date'])
# data['value'] = np.sin(np.linspace(0, 10, len(data))) + np.random.normal(scale=0.1, size=len(data))
# data.set_index('date', inplace=True)

# Beispiel: Laden von Daten aus einer CSV-Datei
data = get_eplus_output_for_single_lstm()
data1 = preprocess_eplus_output_df(data)


# Hyperparameter: Anzahl der Splits für TimeSeriesSplit definieren
n_splits = 5
tscv = TimeSeriesSplit(n_splits=n_splits)

# Liste zur Speicherung der Fehler für jedes Split
errors = []

# Rolling Window Cross-Validation durchführen
for train_index, test_index in tscv.split(data):
    train, test = data.iloc[train_index], data.iloc[test_index]

    # Hier kannst du ein Modell wie LSTM trainieren
    # Beispiel: Dummy-Modell, das den letzten Wert als Vorhersage nutzt
    last_value = train['value'].iloc[-1]
    predictions = [last_value] * len(test)

    # MSE für den aktuellen Split berechnen und speichern
    mse = mean_squared_error(test['value'], predictions)
    errors.append(mse)
    print(f'Test MSE for split: {mse:.4f}')

# Durchschnittlicher MSE über alle Splits
average_mse = np.mean(errors)
print(f'Average MSE over all splits: {average_mse:.4f}')
