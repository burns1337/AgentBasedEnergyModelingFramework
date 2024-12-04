# Description: A simple agent-based model (ABM) to simulate the movement of employees in an office building.
# The employees move randomly in the office building and meet each other. The number of meetings is logged and the maximum,
# minimum and total number of meetings is calculated. The model also logs the number of agents in the office over time.
# The Agents also have a temperature preference profile based on the outdoor temperature forecast and (random) working hours per week (10-30h).
# The model is implemented using the Repast4Py library, which is a Python implementation of the Repast Simphony ABM framework.

# USE PYTHON 3.10 for this script
# 0) install Repast4py: https://repast.github.io/repast4py.site/index.html
# 1) sudo apt install mpich
# 1.2) sudo apt install python3.10-dev (or the version you are using)
# 1.1) sudo apt install libpython3.10-dev (or the version you are using)
# sudo apt install libopenmpi-dev
# pip install mpi4py
# 2) env CC=mpicxx pip install repast4py
# 3) run in folder /src with: mpirun -np 1 python3 abm/repast4py_abm_employees_office_model_05.py random_walk.yaml

# Resources:
#GIS data to repast4py: https://repast.github.io/docs/RepastReference/RepastReference.html#gis

#----------------- TODOs ---------------------- #
#TODO: Adjust the model to include the AVERAGE user feedback from all temperature setpoint profiles of the agents.
#TODO: Adjust the model that the occupancy sum(agent.in_office) gets less, when decreasing the working hours per week.
#TODO: Add a second room to the office (NotWorkingRoom) where the agents can go to when they are not working.
#TODO(optional): Adjust the model to include a task for the agents, e.g. work, meeting, break, leave.
#TODO(optional): Ad GIS Data to the model to simulate the agents in a real office building.
#TODO(optional): Import libs directly in script, like https://stackoverflow.com/questions/31661188/import-files-in-python-with-a-for-loop-and-a-list-of-names

import csv
import os
from pathlib import Path

from pyqtgraph.examples.MultiDataPlot import rng
from repast4py import schedule
from repast4py.space import DiscretePoint
from repast4py import core
import numpy as np
from repast4py.parameters import create_args_parser
from repast4py import parameters
from repast4py import random
from repast4py.space import SharedGrid
from repast4py import context as ctx
from repast4py import space
from mpi4py import MPI
from typing import Dict, Tuple
from dataclasses import dataclass
from repast4py import logging
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random as rand
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import holidays

from weather_fetcher import WeatherFetcher
from userfeedback import Agent


@dataclass
class MeetLog:
    total_meets: int = 0
    min_meets: int = 0
    max_meets: int = 0
    occupancy: int = 0
    in_office: bool = False


# Read the location from the parent directory for the weather forecast fetcher to use
location_path = Path(Path().cwd(), 'output', 'weather_location.txt')
with open(location_path, 'r') as f:
    city = f.read()
    print(f"City for weather forecast fetcher: {city}")

# fetch and store the next 12 hours of outdoor temperature data for Graz
global global_temperature_data_12h_array
# global_temperature_data_12h_array = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
# read column "feels_like" from csv file with the next 12 hours outdoor temperature forecast

global_temperature_data_12h_df = pd.read_csv('next_n_hours_outdoor_temperature.csv')
global_temperature_data_12h_array = global_temperature_data_12h_df['feels_like'].to_list()
# print(f"Global temperature data for the next 12 hours: {global_temperature_data_12h_array}")
# save to csv file
df_12h_temp = pd.DataFrame(global_temperature_data_12h_array, columns=['feels_like'])
df_12h_temp.to_csv('next_n_hours_outdoor_temperature_array.csv', index=False, header=False)


def restore_employee(employee_data: Tuple):
    """ Restores the state of an Employee from a Tuple. """
    uid = employee_data[0]
    pt_array = employee_data[2]
    pt = DiscretePoint(pt_array[0], pt_array[1], 0)
    print(f"Restoring Employee: {uid}, {pt} with data: {employee_data}")

    if uid in employee_cache:
        employee = employee_cache[uid]
    else:
        # employee = Employee(uid[0], uid[2], pt, working_hours_per_week=np.random.choice(np.arange(20, 40)),
        #                     temp_pref=assign_temperature_setpoint())
        employee = Employee(uid[0], uid[2], pt, working_hours_per_week=np.random.choice(np.arange(params['employee.min_working_hours_per_week'], params['employee.max_working_hours_per_week'])),
                            temp_pref=assign_temperature_setpoint())
        employee_cache[uid] = employee

    employee.meet_count = employee_data[1]
    employee.pt = pt
    employee.in_office = True
    return employee


def assign_temperature_setpoint():
    """ Assigns the temperature setpoint profiles for the agents based on the outdoor temperature forecast. """
    global global_temperature_data_12h_array
    # print(f"Global temperature data for the next 12 hours: {global_temperature_data_12h_array}")

    # get the user feedback for the current outdoor temperature
    user = Agent((21, 21), (24, 19))
    setpoint_feedback = np.array([user.setpoint_feedback(t) for t in global_temperature_data_12h_array])
    return setpoint_feedback


def plot_setpoint_feedback():
    """ Plots the temperature setpoint feedback values. """
    plt.plot(assign_temperature_setpoint(), marker='o')
    plt.plot(global_temperature_data_12h_array, marker='x')
    # setpoint_feedback = np.array([user.setpoint_feedback(t) for t in outdoor_temp_12h])
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of Flattened List Feedback Values')
    plt.show()


def temperatur_preference():
    return np.random.choice(np.arange(18, 25))


def is_work_time(step):
    """
    Determines if the given simulation step falls within working hours (8-18) and on a weekday,
    and checks if the day is a national holiday in Austria.

    :param step: The simulation step (e.g., hourly steps from a start time)
    :return: A tuple (is_weekday, is_working_hour, is_holiday), where:
             - is_weekday: True if the step falls on a weekday, False for weekends
             - is_working_hour: True if the step falls between 8:00 and 18:00
             - is_holiday: True if the step falls on a national holiday in Austria
    """
    start_date = datetime(2022, 8, 1)  # Start date of the simulation
    current_time = start_date + timedelta(hours=step)

    is_weekday = current_time.weekday() < 5
    is_working_hour = 8 <= current_time.hour < 18
    austrian_holidays = holidays.Austria()
    is_holiday = current_time in austrian_holidays

    return is_weekday, is_working_hour, is_holiday


def is_agent_working_and_in_office_this_tick(step, agent):
    is_weekday, is_working_hour, is_holiday = is_work_time(step)
    current_time = datetime(2022, 8, 1) + timedelta(hours=step)
    current_day = current_time.weekday()
    current_hour = current_time.hour

    if is_holiday:
        agent.in_office = False
    elif current_day in agent.working_days:
        start_hour, end_hour = agent.working_hours[current_day]
        if start_hour <= current_hour < end_hour:
            if agent.hours_worked_this_day <= 8:
                if rand.random() < agent.home_office_rate:
                    agent.in_office = False
                else:
                    agent.in_office = True
                agent.hours_worked_this_day += 1
                agent.hours_worked_this_week += 1
            else:
                agent.hours_worked_this_day = 0
                agent.in_office = False

            if agent.hours_worked_this_week >= agent.working_hours_per_week:
                agent.in_office = False
        else:
            agent.in_office = False
    else:
        agent.in_office = False

    if agent.in_office:
        agent.task = "work"
    else:
        agent.task = "leave"

    if not is_working_hour:
        agent.hours_worked_this_day = 0

    if not is_weekday:
        agent.hours_worked_this_week = 0

    return agent


def leave_when_neighbour_leaves(agent):
    """ Agent leaves the office when a neighbour leaves. """
    # global employee_cache: self.context.employee
    current_pos = agent.pt
    neighbors = [empl for empl in employee_cache.values() if empl.pt == current_pos and empl.uid != agent.uid]

    for neighbor in neighbors:
        if neighbor.task == "leave":
            agent.task = "leave"
            agent.in_office = False
            return agent
    return agent


def plot_agents_in_office_over_time_barchart(log_file_path):
    """ Plots the number of agents in the office over time in a bar chart. """
    # read eplus scheudle occupancy data
    path_eplus_occ = Path.cwd() / 'final_office_IdealLoad_summer_tiny_hourly_own_schedule_24_final_2.csv'
    df_eplus_occ = pd.read_csv(path_eplus_occ)
    # print(f"Path to real average occupancy data: {df_eplus_occ}")
    # Apply the function to correct the date/time format
    df_eplus_occ['Date/Time'] = df_eplus_occ['Date/Time'].apply(correct_time_format)
    # keep only august data, and remove all other data from the eplus schedule occupancy data frame (df_eplus_occ)
    df_eplus_occ = df_eplus_occ[
        (df_eplus_occ['Date/Time'] >= datetime(2022, 8, 1)) & (df_eplus_occ['Date/Time'] < datetime(2022, 9, 1))]

    # Read the CSV file ABM simulated occ data
    df_abm_occ = pd.read_csv(log_file_path)

    # path to the real average occupancy sensor data
    cwd = Path.cwd()
    occ_avg_path = cwd / 'abm' / 'data_diff_abs_positive_2022-08.csv'
    # print(f"Path to real average occupancy data: {occ_avg_path}")
    df_real_avg = pd.read_csv(occ_avg_path)

    fig = go.Figure()

    # Add bar chart for eplus schedule Occ over datetime
    fig.add_trace(go.Bar(
        x=df_eplus_occ['Date/Time'],
        y=df_eplus_occ['THERMAL ZONE 1:Zone People Occupant Count [](Hourly)'],
        name='EnergyPlus Scheduled Occupancy',
        marker=dict(line=dict(width=2, color='blue'))
    ))

    # Add bar chart for InOfficeCount over datetime
    fig.add_trace(go.Bar(
        x=df_abm_occ['datetime'],
        y=df_abm_occ['InOfficeCount'],
        name='ABM simulated Occupancy',
        marker=dict(line=dict(width=4, color='darkorange'))  # orange, darkorange, coral, gold, khaki, lightgoldenrodyellow, moccasin, navajowhite, peachpuff, peru, sandybrown, seashell, sienna, tan
    ))

    # Add bar chart for the real average occupancy data
    fig.add_trace(go.Bar(
        x=df_real_avg['Time'],
        y=df_real_avg['diff'],
        name='Sensor Data Avg. in/out diff Occupancy',
        marker=dict(line=dict(width=4, color='yellowgreen'))   # green, yellowgreen, lime, limegreen, mediumseagreen, mediumspringgreen, olive, olivedrab, palegreen, seagreen, springgreen, yellow, lightgreen, lightseagreen
    ))

    # Update layout
    fig.update_layout(
        title='Number of Agents in Office Over Time from different sources/models',
        xaxis_title='Datetime',
        yaxis_title='Occupants Count',
    )
    
    # --------- statistics how much % the ABM model is off from the real average occupancy data in the same hours of the day ------
    # df_abm_occ_resampled = df_abm_occ["datetime"].resample('h').max()
    df_abm_occ['datetime'] = pd.to_datetime(df_abm_occ['datetime'])
    df_abm_occ['datetime_floor'] = df_abm_occ['datetime'].dt.floor('h')
    df_abm_occ['datetime'] = df_abm_occ['datetime_floor']
    df_abm_occ.drop(columns=['datetime_floor'], inplace=True)
    # concat df_abm_occ "InOfficeCount" with df_real_avg "diff" on the datetime column of df_abm_occ
    df_abm_occ['datetime'] = pd.to_datetime(df_abm_occ['datetime'])
    df_real_avg['Time'] = pd.to_datetime(df_real_avg['Time'])
    df_real_avg.rename(columns={'Time': 'datetime'}, inplace=True)

    # Setze die 'datetime'-Spalte als Index für beide DataFrames
    df_abm_occ.set_index('datetime', inplace=True)
    df_real_avg.set_index('datetime', inplace=True)

    df_abm_real_merge = pd.merge(df_abm_occ, df_real_avg, on='datetime', how='outer')

    df_abm_real_merge['diff_of_inOffice-realDiff'] = df_abm_real_merge['InOfficeCount'] - df_abm_real_merge['diff']        # calculate the difference between the ABM simulated occupancy data and the real average occupancy data
    # df_abm_real_merge['diff_of_inOffice-realDiff'] = df_abm_real_merge['InOfficeCount'] - df_abm_real_merge['diff']        # calculate the difference between the ABM simulated occupancy data and the real average occupancy data
    df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0'] = df_abm_real_merge['diff_of_inOffice-realDiff'].fillna(0)
    df_real_avg['diff_fillna'] = df_real_avg['diff'].fillna(0)

    print("Difference of every full hour of the day between the ABM model and the real average occupancy data:")
    print(f"median difference: {df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0'].median()}")  # median difference
    print(f"mean difference: {df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0'].mean()}")  # median difference

    # print(f"ABM model is off by {df_abm_occ['diff'].mean().round(3)} mean in avg. from the real average occupancy data (sensor data).")
    df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0_abs'] = df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0'].abs()
    df_abm_real_merge['diff_percentage'] = (df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0_abs'] / df_real_avg['diff_fillna']) * 100
    # print(f"ABM model is off by {df_abm_occ['diff_percentage']}% from the real average occupancy data (sensor data).")  # .mean()
    print(f"\n final df: \n {df_abm_real_merge.iloc[6:]}\n")

    # count the number of hours where the ABM model is off by more than 1 person from the real average occupancy data
    hours_off_by_more_than_1h = df_abm_real_merge[df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0_abs'] > 1].shape[0]
    df_hours_off_by_more_than_1h = df_abm_real_merge[df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0_abs'] > 1]
    print(df_hours_off_by_more_than_1h)
    print(f"Number of hours where the ABM model is off by more than 1 person from the real average occupancy data: {hours_off_by_more_than_1h}")
    # count total hours/sum of column "diff_of_inOffice-realDiff_fillna_0_abs"
    total_hours_off_sum = df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0_abs'].sum()
    # total_hours_off = df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0_abs'].count()

    # total_hours_off_sum = df_abm_real_merge['diff'].sum()
    total_hours = df_abm_real_merge['diff'].sum()
    print(f"Total hours sensor diff: {total_hours}", f"Total hours off from sensor sum (ABM): {total_hours_off_sum}")

    # count the number of hours where the ABM model is off by more than 0 person from the real average occupancy data
    hours_off_by_more_than_0h = df_abm_real_merge[df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0_abs'] > 0].shape[0]
    print(f"Number of hours where the ABM model is off by more than 0 person from the real average occupancy data: {hours_off_by_more_than_0h}")
    # calculate share of hours where the ABM model is off by more than 0 person from the real average occupancy data
    share_hours_off_by_more_than_0h = (total_hours_off_sum / total_hours) * 100
    print(f"Share of hours where the ABM model is off by more than 0 person from the real average occupancy data: {share_hours_off_by_more_than_0h}%")
    # calculate share of hours where the ABM model is off by more than 1 person from the real average occupancy data
    share_hours_off_by_more_than_1h = (hours_off_by_more_than_1h / total_hours) * 100
    print(f"Share of hours where the ABM model is off by more than 1 person from the real average occupancy data: {share_hours_off_by_more_than_1h}%")



    # Create a new figure for the difference in average occupancy
    fig_diff = go.Figure()

    # Add bar chart for the difference in average occupancy
    fig_diff.add_trace(go.Bar(
        x=df_abm_real_merge.index,
        y=df_abm_real_merge['diff'],
        name='Real Occupancy (Sensor In/Out Diff)',
        # transperent color for the bar chart

        marker=dict(line=dict(width=1, color='palevioletred', ))  # red
    ))

    # add bar chart for the percentage difference in average occupancy
    fig_diff.add_trace(go.Bar(
        x=df_abm_real_merge.index,
        y=df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0_abs'],
        name='NEW Diff ABM Model vs Real Avg Occupancy ',
        marker=dict(line=dict(width=4, color='firebrick'))  # red
    ))

    # Add bar chart for the diff between ABM and Sensor Data Avg. in/out diff Occupancy if it's more than +/1h (red)
    fig_diff.add_trace(go.Bar(
        x=df_hours_off_by_more_than_1h.index,
        y=df_hours_off_by_more_than_1h['diff_of_inOffice-realDiff_fillna_0_abs'],
        name='Difference of ABM and Sensor Data Avg. in/out diff Occupancy more than +/- 1h',
        marker=dict(line=dict(width=4, color='mediumvioletred'))   # firebrick, lightpink, hotpink, lavenderblush, mediumvioletred, orchid, palevioletred, pink, plum, purple, violet, rosybrown
    ))

    # Update layout for the difference plot
    fig_diff.update_layout(
        title='Difference in Average Occupancy: ABM Model vs Real Data',
        xaxis_title='Datetime',
        yaxis_title='Occupants Count Difference',
    )

    fig.show()
    fig_diff.show()

    # make table with all the results and save it to a csv file
    df_results = pd.DataFrame({'Total Hours': [total_hours], 'Hours off by more than 0h': [hours_off_by_more_than_0h], 'Share of hours off by more than 0h': [share_hours_off_by_more_than_0h], 'Hours off by more than 1h': [hours_off_by_more_than_1h], 'Share of hours off by more than 1h': [share_hours_off_by_more_than_1h]})
    df_results['Mean Difference'] = df_abm_real_merge['diff_of_inOffice-realDiff_fillna_0'].mean()
    df_results['Employee Count'] = params['employee.count']
    df_results['Min Working Hours per Week'] = params['employee.min_working_hours_per_week']
    df_results['Max Working Hours per Week'] = params['employee.max_working_hours_per_week']
    df_results['Home Office Rate'] = params['employee.home_office_rate']
    print(f"Results: {df_results}")

    file_path_ABM_results = Path(Path().cwd().parent, 'src/output/results_occupancy_comparison.csv')
    if os.path.isfile(file_path_ABM_results):
        df_results.to_csv(file_path_ABM_results, mode='a', header=False, index=True)
    else:
        df_results.to_csv(file_path_ABM_results, mode='w', header=True, index=True)


def round_down_to_hour(df):
    # return dt.replace(minute=0, second=0, microsecond=0)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetime_floor'] = df['datetime'].dt.floor('H')

def plot_one_week_occupancy(log_file_path):
    """ Plots the number of agents in the office over one week. """
    # Read the CSV file
    df = pd.read_csv(log_file_path)
    # TODO
    pass


def correct_time_format(date_str):
    date_str = date_str.strip()
    # Define the base year for the datetime object
    base_year = "2022"
    if "24:00:00" in date_str:
        # Replace "24:00:00" with "00:00:00" and add the base year at the start
        date_str = date_str.replace("24:00:00", "00:00:00")
        # Prepend the base year to the date string
        date_str = f"{base_year} " + date_str
        # Convert to datetime object with the year included
        date_obj = datetime.strptime(date_str, '%Y %m/%d %H:%M:%S')
        # Increment the day
        date_obj += timedelta(days=1)
    else:
        # Prepend the base year to the date string for other cases
        date_str = f"{base_year} " + date_str
        # Convert to datetime object with the year included
        date_obj = datetime.strptime(date_str, '%Y %m/%d %H:%M:%S')
    return date_obj


class Employee(core.Agent):
    """ An Employee agent that moves randomly in the office building/room. """
    TYPE = 0
    OFFSETS = np.array([-1, 1])

    def __init__(self, local_id: int, rank: int, pt: DiscretePoint, working_hours_per_week: int, temp_pref):
        super().__init__(id=local_id, type=Employee.TYPE, rank=rank)
        self.pt = pt
        self.meet_count = 0
        self.in_office = False
        self.temp_pref = temp_pref
        self.working_hours_per_week = working_hours_per_week
        self.hours_worked_this_day = 0
        self.hours_worked_this_week = 0
        self.task = rand.choice(["work", "meeting", "break"])
        self.home_office_rate = params['employee.home_office_rate']
        self.working_days = np.random.choice(range(5), size=np.random.randint(3, 6), replace=False)  # Assign random working days (e.g., Monday to Friday)
        self.working_hours = {day: (np.random.randint(7, 12), np.random.randint(13, 19)) for day in self.working_days}  # Assign random start and end times for each working day

    def walk(self, grid: SharedGrid):
        """ Moves the Employee agent randomly in the office building/room. """
        xy_dirs = random.default_rng.choice(Employee.OFFSETS, size=2)
        self.pt = grid.move(self, DiscretePoint(self.pt.x + xy_dirs[0],
                                                self.pt.y + xy_dirs[1], 0))
        if self.hours_worked_this_week >= self.working_hours_per_week:
            self.task = "leave"

    def save(self) -> Tuple:
        """Saves the state of this Employee as a Tuple.
        Returns:
            The saved state of this Employee.
        """
        return self.uid, self.meet_count, self.pt.coordinates, self.in_office

    def count_colocations(self, grid: SharedGrid, meet_log: MeetLog):
        """Counts the number of agents at the current location and updates the meet log."""
        num_here = grid.get_num_agents(self.pt) - 1
        meet_log.total_meets += num_here
        if num_here < meet_log.min_meets:
            meet_log.min_meets = num_here
        if num_here > meet_log.max_meets:
            meet_log.max_meets = num_here
        self.meet_count += num_here


employee_cache = {}


class Model:
    """ The main model class that initializes the agents and runs the simulation. """

    def __init__(self, comm: MPI.Intracomm, params: Dict):
        super().__init__()
        self.positions = []
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_repeating_event(1.1, 1, self.log_agents)  # log at every 10 steps
        self.runner.schedule_repeating_event(1.2, 1, self.log_in_office_count)
        self.runner.schedule_stop(params['stop.at'])
        schedule.runner().schedule_end_event(self.at_end)

        self.context = ctx.SharedContext(comm)
        box = space.BoundingBox(0, params['world.width'], 0, params['world.height'], 0, 0)
        self.grid = space.SharedGrid(name='grid', bounds=box, borders=space.BorderType.Sticky,
                                     occupancy=space.OccupancyType.Multiple,
                                     buffer_size=2, comm=comm)
        self.context.add_projection(self.grid)
        self.meet_log = MeetLog()

        self.agent_logger = logging.TabularLogger(comm, params['agent_log_file'],
                                                  ['tick', 'datetime', 'agent_id', 'agent_uid_rank',
                                                   'meet_count', 'in_office', 'pt'])
        self.output_folder = 'output'
        os.makedirs(self.output_folder, exist_ok=True)
        self.in_office_count_log_file = os.path.join(self.output_folder, 'in_office_count_log.csv')
        self.initialize_in_office_count_log()

        rank = comm.Get_rank()
        for i in range(params['employee.count']):
            # get a random x,y location in the grid
            pt = self.grid.get_random_local_pt(rng)
            # create and add the Employee to the context
            # employee = Employee(i, rank, pt, working_hours_per_week=np.random.choice(np.arange(20, 40)),
            #                     temp_pref=assign_temperature_setpoint())
            employee = Employee(i, rank, pt, working_hours_per_week=np.random.choice(np.arange(params['employee.min_working_hours_per_week'], params['employee.max_working_hours_per_week'])), temp_pref=assign_temperature_setpoint())

            self.context.add(employee)
            self.grid.move(employee, pt)

        self.meet_log = MeetLog()
        loggers = logging.create_loggers(self.meet_log, op=MPI.SUM,
                                         names={'total_meets': 'total'}, rank=rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.MIN,
                                          names={'min_meets': 'min'}, rank=rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.MAX,
                                          names={'max_meets': 'max'}, rank=rank)

        self.data_set = logging.ReducingDataSet(loggers, MPI.COMM_WORLD,
                                                params['meet_log_file'])

    def start(self):
        """ Starts the simulation. """
        self.runner.execute()

    def step(self):
        """ Executes a single step of the simulation. """
        tick = self.runner.schedule.tick
        # is_weekday, is_working_hour, is_holiday = is_work_time(tick)
        # count_agents_occupancy = 0
        for employee in self.context.agents():
            employee = is_agent_working_and_in_office_this_tick(tick, employee)
            #todo: implement leave_when_neighbour_leaves function here to let agents leave when a neighbour leaves the office too
            # employee = leave_when_neighbour_leaves(tick, employee)
            employee.walk(self.grid)
            employee.count_colocations(self.grid, self.meet_log)

            # print(
            #     f"Tick: {tick}, Datum: {datetime(2022, 8, 1) + timedelta(hours=tick)}, is_weekday: {is_weekday}, is_working_hour: {is_working_hour}, "
            #     f"Employee: {employee.id}, Working Hours: {employee.working_hours_per_week}, Worked Hours this Day: {employee.hours_worked_this_day}, Worked Hours this week: {employee.hours_worked_this_week}, in Office: {employee.in_office} ")

        self.context.synchronize(restore_employee)

        # --> Agent move is happening after the step function!

        # print(
        #     f"Tick: {tick}, Datum: {datetime(2022, 8, 1) + timedelta(hours=tick)}, Total meets: {self.meet_log.total_meets}, is_weekday: {is_weekday}, is_working_hour: {is_working_hour}, "
        #     f"Temp Pref: {employee.temp_pref}, Working Hours: {employee.working_hours_per_week}, Task: {employee.task}")

        self.data_set.log(tick)
        self.meet_log.max_meets = self.meet_log.min_meets = self.meet_log.total_meets = 0
        # Collect positions for plotting
        self.positions.append([employee.pt.coordinates for employee in self.context.agents()])

    def log_agents(self):
        """ Logs the agents' data to a CSV file. """
        tick = self.runner.schedule.tick
        for employee in self.context.agents():
            self.agent_logger.log_row(tick, datetime(2022, 8, 1) + timedelta(hours=tick), employee.id,
                                      employee.uid_rank,
                                      employee.meet_count, employee.in_office, employee.pt.coordinates)
        self.agent_logger.write()

    def initialize_in_office_count_log(self):
        """ Initializes the in-office count log file. """
        with open(self.in_office_count_log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Step', 'datetime', 'InOfficeCount'])

    def log_in_office_count(self):
        """ Logs the number of agents in the office over time. """
        tick = self.runner.schedule.tick
        in_office_count = sum(1 for employee in self.context.agents() if employee.in_office)
        with open(self.in_office_count_log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([tick, datetime(2022, 8, 1) + timedelta(hours=tick), in_office_count])

    def at_end(self):
        """ Finalizes the simulation. """
        self.data_set.close()
        self.agent_logger.close()

    def plot_office_animation(self):
        """ Plots the agents' positions over time. """
        # read data from agents log file and plot the agents' positions over time
        df = pd.read_csv('agents_log.csv')
        # Create a figure
        fig = go.Figure()
        #TODO add animation


def run(params: Dict):
    """ Runs the simulation. """
    model = Model(MPI.COMM_WORLD, params)
    model.start()

    log_file_path = 'output/in_office_count_log.csv'
    # plot_setpoint_feedback()
    plot_agents_in_office_over_time_barchart(log_file_path)


if __name__ == "__main__":
    parser = parameters.create_args_parser()
    args = parser.parse_args()
    # params = parameters.init_params(args.parameters_file, args.parameters)
    params = parameters.init_params(
        'abm/random_walk.yaml', '{}')
    run(params)
