# AgentBasedEnergyModelingFramework
The room surrogate model is a machine learning model that predicts the room temperature and the optimum cooling load, based on the weather forecast, the room occupancy and other features. The model is trained on data from energyplus simulations


## Description
This repository contains the code for the AgentBasedEnergyModelingFramework.
The room surrogate model is a machine learning model that predicts the room temperature and the optimum cooling load,
based on the weather forecast, the room occupancy and other features. The model is trained on data from energyplus simulations
and is used to predict the office room indoor temperature. The model is based on a 1) Regression model adopted to a 'Nowcast'(copyright Thomas Hirsch).
And/Or a 2) LSTM model trained on the same data to predict the room temperature.

Further more, the repository fetches in the Beginning the EnergyPlus weatherfile .epw from the given location (if --location flag used), if it is not already present.
The EnergyPlus simulation is run with the eppy package and the results are ploted and used to train the machine learning model. The model is then used to predict the room temperature and optimum cooling load for the next 12-24 hours.
It fetches also the weather forecast from the OpenWeatherMap API and uses it to predict the room temperature and cooling load for the next 24 hours.


## Installation
- Tested on POP!OS (based on Ubuntu 22.04) with Python 3.10
- install EnergyPlus 24.1 from here https://energyplus.net/downloads
- install the required packages with `pip install eppy geopy rapidfuzz argcomplete selenium webdriver_manager mpi4py SALib plotly pyqtgraph pyqt5`
- install Repast4py: https://repast.github.io/repast4py.site/index.html
  - install MPI for the Agent Based Modeling Module with `sudo apt install mpich`
  - `env CC=mpicxx pip install repast4py`
- install the other packages with `pip install -r requirements.txt`

## Usage
- The main file is the `main.py` file. It can be run with `python main.py --location Burgos` or e.g. `python3 main.py -i small_office_TUGinff_variated_schedule_medium_occupancy.idf -e AUT_ST_Graz.Univ.112900_TMYx.epw`
- The `main.py` file contains the main function that runs the room surrogate model.
- flags can be set in the `main.py` file to change the behavior of the model. See the `main.py` file for more information.
- The `main.py` file can be run with the following flags:
  - `-i` or `--idf` to specify the path to the EnergyPlus IDF file
  - `-e` or `--epw` to specify the path to the EnergyPlus EPW file
  - `-l` or `--location` to specify the location of the weather forecast
  - `-n` or `--nowcast` to specify the nowcast model
  - `-lstm` or `--lstm` to specify the LSTM model
  - `-o` or `--output` to specify the output directory
  - `-N`, `--number` to specify the number of simulations
  - `-h` or `--help` to specify the help of the model

  
## Could Do's:
- [ ] Export Documentation to html when ready for submission, with https://pdoc.dev/docs/pdoc.html
- [ ] Correct the 'Nowcast' model for energeryload optimization in the CLI Tool.
- [ ] Modul Analysis ausbauen um Ergebnisse zu analysieren
  - [ ] analysis.plot_cooling_load()
- [ ] Seperate the WeatherFetcher from the ABM Module to use it independently.
- [ ] Create an API for the Results of the Simulations and Predictions

    
## Notes on the latest files
- LSTM model in LSTM_many_hours.ipynb
- latest file: eppy_multi_run_with_plots_2022_final.ipynb
- simulation results (100 x LHS) in folfder 'simulations_own_schedule_24_final_2'
-- derived from final_office_IdealLoad_summer_tiny_hourly_own_schedule_24_final_2.idf

### General Notes
- when appending csvs, shuffle=False is important, otherwise the order of the data is not preserved. 