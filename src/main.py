# Project: ecom4future
# Authors: Workgroup Gerald Schweiger, Bernhard Lugger, TU Graz, Institute for Software Technology (IST)
# Version: v0.2
# Date: 2024-09-24

# Description: Dieses Skript implementiert ein Kommandozeilen-Tool, das verschiedene Funktionen zum Weather History + Forecast Fetchen und zur Verarbeitung von IDF- zum einen,
# und EnergyPlus und Repast4Py (ABM) Simulationen und LSTM- oder Regression('Nowcast')-based Predictions zum Andersen bereitstellt.
# Das Tool kann über die Kommandozeile mit verschiedenen Argumenten aufgerufen werden, um die gewünschte Funktionalität zu aktivieren.
# Es kann zum Beispiel verwendet werden, um eine IDF-Datei zu laden, ein epw. File von der angegebenen Location downzuloaden,
# Wetter Daten aus den letzten n Stunden oder den zukünftigen n Stunden zu fetchen,
# eine LSTM-Vorhersage durchzuführen, eine Nowcast-Prognose durchzuführen oder die Stadt für die Analyse festzulegen.
# Das Tool kann auch als Modul importiert und in anderen Skripten verwendet werden, um die Funktionalität in andere Anwendungen zu integrieren.

import argparse
import argcomplete
from pathlib import Path

from utils.location_handler import LocationHandler
from utils.epw_downloader import EPWDownloader

from abm import WeatherFetcher
from abm import ABMSimulationRunner
from abm.Agents_Info_Fetcher import get_office_occupancy_df

from EnergyPlus.simulation_runner import SimulationRunner
from utils.eplus_output_fetcher import get_eplus_output
from utils.Functions import *

from analysis_tool.data_visualizer import DataVisualizer
from predictions.lstm_predictor import LSTM_Predictor
from predictions.cooling_load_optimizer import CoolingLoadOptimizer
# from predictions.cooling_load_optimizer import predict_horizon_hours_from_start_index_with_given_cooling_load




def init_argparse():
    parser = argparse.ArgumentParser(
        description="A command-line tool for processing IDF- and weather files and performing predictions.")

    parser.add_argument('-i', '--idf', type=str, help='Path to the IDF file')
    parser.add_argument('-e', '--epw', type=str, help='Path to the EPW file')
    parser.add_argument('-s', '--sketchup', type=str, help='Path to the SketchUp file')
    parser.add_argument('-ls', '--lstm', action='store_true', help='Perform LSTM prediction')
    parser.add_argument('-n', '--nowcast', action='store_true', help='Perform nowcast')
    parser.add_argument('-lo', '--location', type=str, help='Set the location for analysis')
    parser.add_argument('-o', '--output', type=str, help='Output directory for results')
    parser.add_argument('-N', '--number', type=int, default=1,
                        help='Number of energyplus simulations to run, with different parameters, using LHS')
    # parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-st', '--start-date', type=str, help='Start date for the simulation in format e.g. 2024-07-18')
    parser.add_argument('-a', '--annual', type=str, help='Annual simulation')
    parser.add_argument('-p', '--people', type=int, help='Number of people in the room in average')
    parser.add_argument('-c', '--cooling', type=float, help='static Cooling setpoint')

    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    return args


def fetch_epw_file_and_save_it_and_the_location(args, _location):
    # -------- check if there is already an epw file in cwd and if not, then download it from the energyplus website --------
    found_local_epw_file = find_files_with_substring_and_return_first_one(_location)
    if not args.epw or not found_local_epw_file:
        args.epw = found_local_epw_file
        print(f"EPW file for {_location} found in the current working directory: {args.epw}")
    else:
        print("No EPW file found in the current working directory. Fetching EPW file from EnergyPlus website...")

    # -------- get the weather file from energyplus website if only location is provided and not the epw file --------
    if (args.location or _location) and not args.epw:
        epw_downloader = EPWDownloader(_location)
        epw_filename = epw_downloader.fetch_epw_file()
        if epw_filename:
            args.epw = epw_filename
        else:
            print(f"No EPW file found. Sticking with default Location {_location}...")

    output_dir = Path('output')
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Created 'output' directory.")
    else:
        print("'Output' directory already exists.")

    with open('output/weather_location.txt', 'w') as f:
        f.write(_location)


def run_energyplus_simulation(args):
    #TODO modify idf file with start date and other parameters BEFORE running the simulation

    if args.number and args.number > 1:
        run_multiple_simulations(args)

    # run the simulation once
    simulation = SimulationRunner(args)
    simulation.run_simulation()


def run_multiple_simulations(args):
    #TODO implement the logic to run multiple simulations with different parameters using LHS sampling and the number of simulations to run
    for i in range(args.number):
        # modify the idf file with the new parameters, do LHS sampling

        simulation = SimulationRunner(args)
        simulation.run_simulation()
        # save the results in a new folder with the simulation number as the folder name and the results inside the folder


def check_prediction_args_and_apply_them(args):
    if args.lstm:
        predict_lstm(args)

    if args.nowcast:
        predict_nowcast()


def find_files_with_substring_and_return_first_one(substring, cwd=None):
    """Find files in the current working directory (or a specified directory) that contain a specific substring."""
    if cwd is None:
        cwd = Path.cwd()
    else:
        cwd = Path(cwd)

    try:
        # Use the glob pattern to match files containing the substring
        files = list(cwd.glob(f"*{substring}*"))
        # return only the filename
        files = [file.name for file in files]
        print(f"Files containing '{substring}': {files}. Found {len(files)} files. Returning the first one: {files[0]}")
        return files[0]

    except Exception as e:
        print(f"Error while searching for files containing '{substring}': {e}")
        return None


def predict_lstm(args):
    print("Performing LSTM prediction...")
    lstm_prediction = LSTM_Predictor(args)
    lstm_prediction.run_lstm_prediction()
    # save lstm_prediction.predictions to a csv file
    lstm_prediction.predictions.to_csv('output/lstm_predictions.csv')


def predict_nowcast():
    print("Performing nowcast...")
    # Hier wird die Logik zur Durchführung einer Nowcast-Prognose implementiert.
    pass


def plot_occupancy():
    data_visualizer = DataVisualizer()

    # ------------- Plot the mean and median number of agents in the office for every hour ABM-------------
    agents_in_office_over_time_df = get_office_occupancy_df()
    new_df_hourly = data_visualizer.avg_occupancy_weekday(agents_in_office_over_time_df, "InOfficeCount")
    data_visualizer.plot_points_and_boxplot(new_df_hourly)

    # ------------- plot also boxplot for EPLUS simulation results -------------
    eplus_output_df = get_eplus_output()
    # Plot the mean and median number of occupants in the office for every hour
    new_df_hourly_eplus = data_visualizer.avg_occupancy_weekday(eplus_output_df, "THERMAL ZONE 1:Zone People Occupant Count [](Hourly)")
    data_visualizer.plot_points_and_boxplot(new_df_hourly_eplus)

    # ------------- plot also boxplot for sensor avg values -------------
    file_path_sensor_diff = Path(Path().cwd(), 'data_diff_abs_positive_2022-08.csv')
    with open(file_path_sensor_diff) as f:
        sensor_diff_df = pd.read_csv(f, index_col=0)
    new_df_hourly_sensor = data_visualizer.avg_occupancy_weekday(sensor_diff_df, "diff")
    data_visualizer.plot_points_and_boxplot(new_df_hourly_sensor)


def plot_temperatures():
    data_visualizer = DataVisualizer()
    eplus_output_df = get_eplus_output()
    eplus_output_df_outdoor_temp = eplus_output_df["Environment:Site Outdoor Air Drybulb Temperature [C](Hourly)"]
    eplus_output_df_indoor_temp = eplus_output_df["THERMAL ZONE 1:Zone Mean Air Temperature [C](Hourly)"]

    # get predictions from csv file and plot them against the eplus output data
    lstm_predictions_df = pd.read_csv('output/lstm_predictions.csv', header=0, index_col=0)
    # print(lstm_predictions_df)

    # sensors_temperatures_df outdoor temperature weather station data + eplus data
    sensors_temperatures_df = pd.read_csv('mesured_data/WeatherStation_OutsideTemp.csv', header=0)
    sensors_temperatures_df['datetime'] = pd.to_datetime(sensors_temperatures_df['Time'])
    sensors_temperatures_df_resample_60min = resample_60min(sensors_temperatures_df)

    start_date = '2022-08-01 00:00:00'  # '2022-08-19 19:00:00'
    end_date = '2022-08-31 00:00:00'
    data_visualizer.plot_temperature_comparison_simple(eplus_output_df_outdoor_temp, sensors_temperatures_df_resample_60min, start_date, end_date)

    # indoor temperature from the eplus simulation + the sensor data for the same time period
    sensors_temperatures_indoor_df = pd.read_csv('mesured_data/TEIS001_Temp.csv', header=0)
    sensors_temperatures_indoor_df['datetime'] = pd.to_datetime(sensors_temperatures_indoor_df['Time'])
    sensors_temperatures_indoor_df_resample_60min = resample_60min(sensors_temperatures_indoor_df)

    data_visualizer.plot_indoor_temperature_comparison_simple(eplus_output_df_indoor_temp, sensors_temperatures_indoor_df_resample_60min, lstm_predictions_df, start_date, end_date)

def plot_cooling_load():
    data_visualizer = DataVisualizer()
    eplus_output_df = get_eplus_output()
    # start_date = '2022-08-01 00:00:00'  # '2022-08-19 19:00:00'
    # end_date = '2022-08-31 00:00:00'

    # plot the cooling load
    data_visualizer.plot_eplus_cooling_load(eplus_output_df)
    data_visualizer.plot_eplus_cooling_load_from_august(eplus_output_df)
    data_visualizer.plot_eplus_cooling_load_from_end_of_august(eplus_output_df)



def analyse_results():
    # ------- comparison of the ABM and EPLUS simulation results -------
    #     results = sim.get_results()
    #     analysis = Analysis(results)
    #     analysis.plot_energy_consumption()
    pass

def optimize_AC_energy_consumption():
    # ------- optimization of the AC energy consumption -------
    optimizer = CoolingLoadOptimizer()
    optimizer.run_main()


def in_the_end_clean_up():
    # ------- clean up the output directory -------
    pass


def main():
    args = init_argparse()

    # ----------- First, get the location for the weather forecast fetcher, either 1) from the CLI flag or 2) from the epw file name or 3) from the IDF file -----------
    location_handler = LocationHandler(args)
    location_choosen = location_handler.execute()
    fetch_epw_file_and_save_it_and_the_location(args, location_choosen)

    # ----------- load past und forecast weather data -----------
    weather_fetcher = WeatherFetcher(location_choosen)
    weather_fetcher.run_weather_fetcher()
    weather_fetcher.init_pyowm_and_get_weather()
    # weather_fetcher.get_meteostat_weather()
    #todo complex/large effort: convert weather data to a .epw file OR get latest up2date epw files from austrian cities;

    # ------------- Then, run the energyplus simulations -------------
    run_energyplus_simulation(args)

    # ----- ABM simulation with Repast4Py to get the agent objects with the desired parameters (temperature preferences depending on outdoor temperature) -----
    # abm_simulation = ABMSimulationRunner(args)
    # abm_simulation.run_simulation()
    #TODO check if the abm simulation is successful or not and if not, then use the default abm simulation output file
    # plot_occupancy()
    analyse_results()
    check_prediction_args_and_apply_them(args)

    plot_temperatures()
    plot_cooling_load()

    #TODO activate the optimization of the AC energy consumption
    optimize_AC_energy_consumption()

    in_the_end_clean_up()
    print("Done.")


if __name__ == '__main__':
    main()

    # TODO add tabulator findable files when running the script from the command line
