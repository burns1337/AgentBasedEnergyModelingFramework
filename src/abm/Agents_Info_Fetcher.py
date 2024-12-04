

# Import the necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Then, get the agent objects and their parameters from the ABM simulation output from the csv file output agent_log.csv
# agent_objects = get_agent_objects_from_abm_simulation_output()
def get_agent_objects_from_abm_simulation_output():
    """ Get the agent objects and their parameters from the ABM simulation output """
    # Load the agent log file in_office_count_log.csv
    # open file with Path current working directory/output/in_office_count_log.csv and read it as a pandas DataFrame

    # Define the path to the CSV file
    file_path = Path(Path().cwd().parent, 'src/abm/output/in_office_count_log.csv')
    with open(file_path) as f:
        agent_log = pd.read_csv(f)

    # Get the agent objects and their parameters from the agent log
    agent_objects = []
    for i in range(len(agent_log)):
        agent = agent_log.iloc[i]
        agent_objects.append(agent)

    return agent_objects


def get_office_occupancy_df():
    # Get the number of agents in the office over time
    file_path = Path(Path().cwd().parent, 'src/output/in_office_count_log.csv')   # .parent
    with open(file_path) as f:
        agent_log = pd.read_csv(f)
    agent_objects_df = pd.DataFrame(agent_log)
    agent_objects_df['datetime'] = pd.to_datetime(agent_objects_df['datetime'])
    agent_objects_df.set_index('datetime', inplace=True)
    agent_objects_df['InOfficeCount'] = agent_objects_df['InOfficeCount'].astype(int)

    return agent_objects_df

