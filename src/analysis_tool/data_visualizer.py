import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


class DataVisualizer:
    def __init__(self):
        self.occupancy_df = None
        pass

    def avg_occupancy_weekday(self, df, values='InOfficeCount'):
        # df = df.to_frame(name='InOfficeCount')
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna()
        df['hour'] = df.index.hour
        # hourly_avg_diff = df.groupby('hour')['diff'].mean()
        hourly_diff_df = df.pivot_table(values=values, index=df.index.date, columns='hour', aggfunc='mean')
        self.occupancy_df = hourly_diff_df
        return hourly_diff_df


    def plot_hourly_diff(self, df):
        # Transpose the DataFrame so that hours are the x-axis and each day is a separate line
        # df_transposed = df.transpose()
        # Plot each column (each day) as points
        plt.figure(figsize=(12, 6))
        # for column in df.columns:
        #     plt.plot(df.index, df[column], 'o', label=column)
        plt.plot(df.index, df['InOfficeCount'], 'o-')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Occupants')
        plt.title('Hourly Occupants for Each Day from ABM Simulation')
        plt.legend(title='Date', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.show()

    def plot_hourly_boxplot(self, df):
        # Melt the DataFrame to long format
        df_long = df.melt(var_name='Hour', value_name='InOfficeCount')

        # Plotting the boxplot
        plt.figure(figsize=(12, 6))
        boxplot = df_long.boxplot(by='Hour', column=['InOfficeCount'], grid=False, showmeans=True)

        # Set plot labels and title
        plt.xlabel('Hour of the Day')
        plt.ylabel('Occupants')
        plt.title('Boxplot of Hourly Occupancy with Median and Mean')
        plt.suptitle('')  # Remove the default title
        plt.grid(True)

        # Add custom legend
        handles = [
            plt.Line2D([0], [0], color='green', lw=2, label='Mean', marker='^', markersize=6, linestyle=''),
            plt.Line2D([0], [0], color='green', lw=2, label='Median')
        ]
        plt.legend(handles=handles, loc='upper right')
        plt.show()

    def plot_hourly_points_plotly(self, df):
        # Convert the DataFrame to a long format
        df_long = df.reset_index().melt(id_vars='index', var_name='Hour', value_name='Diff')
        df_long = df_long.dropna()  # Drop NaN values to avoid plotting them

        # Create a scatter plot with Plotly
        fig = px.scatter(df_long, x='Hour', y='Diff', color='index', title='Hourly Occupants for Each Day and Hour')

        # Customize the layout
        fig.update_layout(xaxis_title='Hour of the Day', yaxis_title='Occupants')

        # Show the plot
        fig.show()

    def plot_hourly_boxplot_plotly(self, df):
        # Convert the DataFrame to a long format
        df_long = df.melt(var_name='Hour', value_name='Diff')
        df_long = df_long.dropna()  # Drop NaN values to avoid plotting them

        fig = go.Figure()

        for hour in sorted(df_long['Hour'].unique()):
            fig.add_trace(go.Box(
                y=df_long[df_long['Hour'] == hour]['Diff'],
                name=str(hour),
                boxmean='sd'  # Displays mean and standard deviation
            ))
        fig.update_layout(title='Boxplot of Hourly Occupancy with Median and Mean',
                          xaxis_title='Hour of the Day',
                          yaxis_title='Occupants')
        fig.show()

    def plot_points_and_boxplot(self, df):
        """ Plot a boxplot with data points overlayed for each hour of the day. """
        # Melt the DataFrame to long format
        df_long = df.melt(var_name='Hour', value_name='Diff')

        fig, ax = plt.subplots(figsize=(12, 6))

        boxplot = df_long.boxplot(by='Hour', column=['Diff'], grid=False, showmeans=True, meanline=False,
                                  patch_artist=True,
                                  ax=ax,
                                  boxprops=dict(facecolor='cyan', color='blue', alpha=0.5),
                                  medianprops=dict(color='blue'),
                                  meanprops=dict(marker='^', markerfacecolor='orange', markeredgecolor='black',
                                                 markersize=8),
                                  flierprops=dict(marker='o', color='black', markersize=6, alpha=0.5))

        # Overlay the scatter plot with correct alignment
        positions = np.arange(len(df_long['Hour'].unique())) + 1
        for i, pos in enumerate(sorted(df_long['Hour'].unique())):
            values = df_long[df_long['Hour'] == pos]['Diff']
            ax.scatter(np.full_like(values, positions[i]), values, color='blue', alpha=0.5,
                       label='Data Points' if i == 0 else "")

        # Set plot labels and title
        ax.set_xlabel('Hour of the Day')
        ax.set_ylabel('Occupants')
        ax.set_title('Boxplot of average Hourly Occupancy with Data Points for August 2022')
        plt.suptitle('')  # Remove the default title
        ax.grid(True)

        # Add custom legend matching the plot symbols
        handles = [
            plt.Line2D([0], [0], color='orange', lw=2, linestyle='--', marker='^', markersize=8, label='Mean'),
            plt.Line2D([0], [0], color='blue', lw=2, label='Median'),
            plt.Line2D([0], [0], color='blue', marker='o', linestyle='None', markersize=8, alpha=0.5,
                       label='Data Points')
        ]
        ax.legend(handles=handles, loc='upper right')

        # Show the plot
        plt.show()

    def plot_agents_in_office(self, agent_objects):
        # Plot the number of agents in the office over time
        agent_ids = [agent['agent_id'] for agent in agent_objects]
        in_office_counts = [agent['in_office_count'] for agent in agent_objects]

        plt.plot(agent_ids, in_office_counts)
        plt.xlabel('Agent ID')
        plt.ylabel('In Office Count')
        plt.title('Number of Agents in the Office over Time')
        plt.show()


    def plot_temperature_comparison(self, eplus_data, sensor_data, start_date, end_date):
        """
        Plots the outdoor and indoor temperature comparison from EnergyPlus simulation, weather forecast, and sensor data
        for the same time period.
        """
        # Timestamp('2022-08-19 19:00:00')  --- start of test data. end of test data: Timestamp('2022-08-31 00:00:00')
        eplus_outdoor_temp = eplus_data["Date/Time", "Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)"]
        # forecast_outdoor_temp = forecast_data[["Timestamp", "Forecasted Outdoor Temp"]]
        sensor_outdoor_temp = sensor_data["datetime_col", "sensor-readings.reading"]

        # Filter data by date range
        eplus_outdoor_temp = eplus_outdoor_temp[
            (eplus_outdoor_temp['Date/Time'] >= start_date) & (eplus_outdoor_temp['Date/Time'] <= end_date)]
        # forecast_outdoor_temp = forecast_outdoor_temp[
        #     (forecast_outdoor_temp['Timestamp'] >= start_date) & (forecast_outdoor_temp['Timestamp'] <= end_date)]
        sensor_outdoor_temp = sensor_outdoor_temp[
            (sensor_outdoor_temp['datetime_col'] >= start_date) & (sensor_outdoor_temp['datetime_col'] <= end_date)]

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(eplus_outdoor_temp['Date/Time'], eplus_outdoor_temp['Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)'],
                 label='EnergyPlus Outdoor Temp', color='blue')
        # plt.plot(forecast_outdoor_temp['Timestamp'], forecast_outdoor_temp['Forecasted Outdoor Temp'],
        #          label='Forecasted Outdoor Temp', color='green')
        plt.plot(sensor_outdoor_temp['datetime_col'], sensor_outdoor_temp['sensor-readings.reading'],
                 label='Sensor Outdoor Temp', color='red')
        plt.xlabel('Time')
        plt.ylabel('Temperature [C]')
        plt.title('Outdoor Temperature Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_temperature_comparison_simple(self, eplus_data, sensor_data, start_date, end_date):
        # Filter data by date range + Ensure the index is a datetime index
        if not isinstance(eplus_data.index, pd.DatetimeIndex):
            eplus_data.index = pd.to_datetime(eplus_data.index)
        eplus_outdoor_temp = eplus_data.loc[start_date:end_date]

        if not isinstance(sensor_data.index, pd.DatetimeIndex):
            sensor_data.index = pd.to_datetime(sensor_data.index)
        sensor_outdoor_temp = sensor_data.loc[start_date:end_date]

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(eplus_outdoor_temp,
                 label='EnergyPlus Outdoor Temp', color='blue')
        # plt.plot(forecast_outdoor_temp['Timestamp'], forecast_outdoor_temp['Forecasted Outdoor Temp'],
        #          label='Forecasted Outdoor Temp', color='green')
        plt.plot(sensor_outdoor_temp['sensor-readings.reading'],
                 label='Sensor Outdoor Temp', color='red')
        plt.xlabel('Time')
        plt.ylabel('Temperature [C]')
        plt.title('Outdoor Temperature Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_indoor_temperature_comparison_simple(self, eplus_data, sensor_data, predictions_data, start_date, end_date):
        # Filter data by date range + Ensure the index is a datetime index
        if not isinstance(eplus_data.index, pd.DatetimeIndex):
            eplus_data.index = pd.to_datetime(eplus_data.index)
        eplus_outdoor_temp = eplus_data.loc[start_date:end_date]

        if not isinstance(sensor_data.index, pd.DatetimeIndex):
            sensor_data.index = pd.to_datetime(sensor_data.index)
        sensor_outdoor_temp = sensor_data.loc[start_date:end_date]

        if not isinstance(predictions_data.index, pd.DatetimeIndex):
            predictions_data.index = pd.to_datetime(predictions_data.index)
        predictions_outdoor_temp = predictions_data.loc[start_date:end_date]
        # predictions_outdoor_temp = predictions_data
        # shift the predictions by 12 hours to align with the sensor data and EnergyPlus data
        predictions_outdoor_temp = predictions_outdoor_temp.shift(-12, freq='H')

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.plot(eplus_outdoor_temp,
                 label='EnergyPlus Indoor Temp', color='blue')
        # plt.plot(forecast_outdoor_temp['Timestamp'], forecast_outdoor_temp['Forecasted Outdoor Temp'],
        #          label='Forecasted Outdoor Temp', color='green')
        plt.plot(sensor_outdoor_temp['sensor-readings.reading'],
                 label='Sensor Indoor Temp', color='red')
        plt.plot(predictions_outdoor_temp['predictions'], label='Predictions', color='green')
        plt.xlabel('Time')
        plt.ylabel('Temperature [C]')
        plt.title('Indoor Temperature Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()




