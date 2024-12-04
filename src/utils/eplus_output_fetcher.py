import pandas as pd
from pathlib import Path
from datetime import timedelta


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


def get_office_occupancy_df():
    file_path = Path(Path().cwd().parent, 'src/output/eplusout.csv')
    with open(file_path) as f:
        eplus_output = pd.read_csv(f)
    eplus_output_df = pd.DataFrame(eplus_output)


def get_eplus_output():
    file_path = Path(Path().cwd().parent, 'src/output/eplusout.csv')  # .parent
    with open(file_path) as f:
        eplus_output_df = pd.read_csv(f, header=0, index_col=0)
    eplus_output_df["THERMAL ZONE 1:Zone People Occupant Count [](Hourly)"] = eplus_output_df["THERMAL ZONE 1:Zone People Occupant Count [](Hourly)"].round()  # make occupancy count natural numbers
    eplus_output_df = convert_eplus_index_to_datetime(eplus_output_df)
    return eplus_output_df


def get_eplus_output_for_single_lstm():
    file_path = Path(Path().cwd().parent, '../output/eplusout.csv')
    with open(file_path) as f:
        eplus_output_df = pd.read_csv(f, header=0, index_col=0)
    eplus_output_df["THERMAL ZONE 1:Zone People Occupant Count [](Hourly)"] = eplus_output_df["THERMAL ZONE 1:Zone People Occupant Count [](Hourly)"].round()  # make occupancy count natural numbers
    eplus_output_df = convert_eplus_index_to_datetime(eplus_output_df)
    return eplus_output_df
