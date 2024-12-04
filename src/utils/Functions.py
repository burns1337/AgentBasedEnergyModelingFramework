# function to convert timestamps to datetime and add to dataframe
import pandas as pd


def convert_timestamps_to_datetime(df):
    df.index = pd.to_datetime(df.index // 1000, unit='s')
    return df


def convert_timestamps(df, col):
    # index colum to datetime and set as index column of dataframe

    df['datetime_col'] = pd.to_datetime(df[col] // 1000, unit='s')

    # https://stackoverflow.com/questions/35321812/move-column-in-pandas-dataframe
    column_to_move = df.pop('datetime_col')
    # insert column with insert(location, column_name, column_value)
    df.insert(0, 'datetime_col', column_to_move)
    df['datetime_col'] = pd.to_datetime(df['datetime_col'])
    # df.set_index('datetime_col', inplace=True)
    return df


def resample_data(df, col, freq):
    """ resample data to given frequency"""
    # resample data to 10 minute intervals
    # data_tur_A01_resample = data_tur_A01.resample('10T', on='datetime_col').max()
    df_resample = df.resample(freq, on=col).max()
    # remove nan values from dataframe
    df_resample = df_resample.dropna()
    return df_resample


def filter_rows_by_date(df, start_date, end_date):
    """
    Filter rows in a DataFrame based on the datetime index column.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    start_date (str): The start date in 'YYYY-MM-DD' format.
    end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """
    # Ensure the index is a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Filter the DataFrame
    filtered_df = df.loc[start_date:end_date]

    return filtered_df


def filter_dataframe_or_not(df, column, value):
    """returns only data where the given value is in column."""
    if value:
        df_mask = df[column] == value
        filtered_df = df[df_mask]
        return filtered_df
    else:
        return df


def resample_1min_pad(df, c_name):
    """ resample data to 1 minute intervals and pad missing values with last known value"""
    df_datetime = convert_timestamps(df, 'Time')
    df_resample = df_datetime.resample('1T').fillna("pad")
    df_resample = df_resample.rename(columns={"sensor-readings.reading": c_name})
    # print(df_resample)
    return df_resample


def resample_1min(df, c_name):
    """ resample data to 1 minute intervals and pad missing values with last known value"""
    df_datetime = convert_timestamps(df, 'Time')
    df_resample = df_datetime.resample('1T')
    # df_resample = df_resample.rename(columns={"sensor-readings.reading": c_name})
    print(df_resample)
    return df_resample

def resample_60min(df):
    """ resample data to 1 minute intervals and pad missing values with last known value"""
    df_datetime = convert_timestamps(df, 'Time')
    # df_resample = df_datetime.resample('60min', on='datetime_col').fillna("pad")
    df_resample = df_datetime.resample('60min', on='datetime_col').mean()
    # df_resample = df_resample.rename(columns={"sensor-readings.reading": c_name})
    # print(df_resample)
    return df_resample

def resample_60min_predictions(df):
    """ resample data to 1 minute intervals and pad missing values with last known value"""
    df_datetime = convert_timestamps(df, 'datetime')
    # df_resample = df_datetime.resample('60min', on='datetime_col').fillna("pad")
    df_resample = df_datetime.resample('60min', on='datetime_col').mean()
    # df_resample = df_resample.rename(columns={"sensor-readings.reading": c_name})
    # print(df_resample)
    return df_resample


def resample_30min_pad_mean(df, c_name):
    """ resample data to 1 minute intervals and pad missing values with last known value"""
    df_datetime = convert_timestamps(df, 'Time')
    print(df_datetime)
    df_resample = df_datetime.resample('1T').fillna("pad")
    df_resample = df_resample.rename(columns={"sensor-readings.reading": c_name})
    print(df_resample)
    return df_resample


# decorator function to describe dataframe amount of rows and columns and first 5 rows
def describe_df(func):
    def wrapper(*args, **kwargs):
        print(f"Dataframe has {args[0].shape[0]} rows and {args[0].shape[1]} columns")
        print(f"First 5 rows of dataframe: \n {args[0].head()}")
        return func(*args, **kwargs)

    return wrapper


def occupacy_data_diff_merge(df_in, df_out):
    """ merge occupancy dataframes and calculate difference between occupancy and non-occupancy"""
    df_in_out = pd.merge(df_in, df_out, how='outer', left_index=True, right_index=True)
    df_in_out = df_in_out.fillna(0)
    df_in_out['Occupancy_diff'] = df_in_out['People go in room'] - df_in_out['People go out of room']
    df_in_out['Occupancy_diff'] = df_in_out['Occupancy_diff'].cumsum()
    df_in_out['Occupancy_diff'] = df_in_out['Occupancy_diff'].astype(int)
    print(df_in_out)
    # df_out['Occupancy_diff'] = df_out['Occupancy_diff'].fillna(0)
    return df_in_out


# TODO: write func to add occupancy of month to dataframe
# write func to add occupancy of week to dataframe
# write func to add occupancy of day to dataframe

# TODO: write func to adjust schedule of EnergyPlus API to occupancy of month
# write func to adjust schedule of EnergyPlus API to occupancy of week
# write func to adjust schedule of EnergyPlus API to occupancy of day




# TODO: write func to set cooling temperature setpoints to EnergyPlus API
from eppy.modeleditor import IDF


def set_cooling_setpoint(idf_file_path, thermostat_name, new_cooling_setpoint):
    """
    Set the cooling setpoint for a thermostat in an IDF file.

    Parameters:
    idf_file_path (str): The path to the IDF file.
    thermostat_name (str): The name of the ThermostatSetpoint:DualSetpoint object.
    new_cooling_setpoint (float): The new cooling setpoint in Celsius.

    Returns:
    None
    """
    # Set the EnergyPlus version for eppy
    IDF.setiddname("Energy+.idd")

    # Load the IDF file
    idf = IDF(idf_file_path)

    # Get the thermostat
    thermostat = idf.getobject('ThermostatSetpoint:DualSetpoint', thermostat_name)

    # Get the cooling setpoint schedule
    cooling_schedule = idf.getobject('Schedule:Compact', thermostat.Cooling_Setpoint_Temperature_Schedule_Name)

    # Set the new cooling setpoint for all hours
    for field in cooling_schedule.fieldnames[3:]:
        cooling_schedule[field] = new_cooling_setpoint

    # Save the IDF file
    idf.save()

    # APPLY:
    # set_cooling_setpoint('path_to_your_idf_file.idf', 'Your Thermostat Name', 24.0)
