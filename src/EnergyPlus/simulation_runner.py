import os
import random
import subprocess
from eppy import modeleditor
from eppy.modeleditor import IDF
from SALib.sample import latin

idd_path = '/usr/local/EnergyPlus-24-1-0/Energy+.idd'
IDF.setiddname(idd_path)


class SimulationRunner:
    def __init__(self, args):
        self.idf_file = args.idf
        self.weather_file = args.epw
        self.output_dir = args.output if args.output else "output"
        # self.verbose = args.verbose

    def _build_command(self):
        """Erstellt den Kommandozeilenbefehl für die EnergyPlus-Simulation."""
        #  Could DO: add option:  -a, --annual Force annual simulation
        # https://github.com/NREL/EnergyPlus/blob/develop/doc/running-energyplus-from-command-line.md (Command Line Options)

        if self.weather_file and self.idf_file:
            command = [
                "energyplus -r -w " + self.weather_file + " -d " + self.output_dir + " -r " + self.idf_file + " -x "]

        elif not self.weather_file:
            print("No .epw weather file given! using default Graz weather file")
            weaterfile_default = "AUT_ST_Graz.Univ.112900_TMYx.epw"
            command = [
                "energyplus -r -w " + weaterfile_default + " -d " + self.output_dir + " -r small_office_TUGinff_variated_schedule_medium_occupancy.idf" + " -x "]

        elif not self.idf_file:
            print("No IDF file given! Using default Graz idf file")
            idffile_default = "small_office_TUGinff_variated_schedule_medium_occupancy.idf"
            command = [
                "energyplus -r -w " + self.weather_file + " -d " + self.output_dir + " -r " + idffile_default + " -x "]

        elif not self.weather_file and not self.idf_file:
            print("No .epw weather file and no IDF file given! Using default Graz weather and idf file")
            # weaterfile_default = "AUT_ST_Graz.Univ.112900_TMYx.epw"
            # idffile_default = "small_office_TUGinff_variated_schedule_medium_occupancy.idf"
            command = [
                "energyplus -r -w AUT_ST_Graz.Univ.112900_TMYx.epw -d " + self.output_dir + " -r small_office_TUGinff_variated_schedule_medium_occupancy.idf" + " -x "]
            # -x is for running the simulation with extended tools which is needed for HVAC Tempalte IdealLoads

        command_str = ' '.join(command)
        return command_str

    def modify_start_and_end_date(self, start_date, end_date):
        """Modifies the IDF file with the given start and end date."""
        idf = IDF(self.idf_file)
        for obj in idf.idfobjects['RunPeriod']:
            obj.Begin_Month = start_date.month
            obj.Begin_Day_of_Month = start_date.day
            obj.End_Month = end_date.month
            obj.End_Day_of_Month = end_date.day
        idf.save(self.idf_file)

    # Funktion zum Anpassen der IDF-Datei
    def modify_idf(template_path, output_path, cooling_setpoint, people_in_room, cooling_load):
        idf = IDF(template_path)
        print(template_path)

        # Cooling Setpoint ändern
        for obj in idf.idfobjects['HVACTemplate:Thermostat']:
            obj.Constant_Cooling_Setpoint = cooling_setpoint

        # People in Room ändern
        for obj in idf.idfobjects['PEOPLE']:
            obj.Number_of_People = people_in_room

        # Cooling Load ändern (z.B., durch Ändern von Equipment oder Zone Sizing)
        for obj in idf.idfobjects['HVACTemplate:Zone:IdealLoadsAirSystem']:
            obj.Maximum_Total_Cooling_Capacity = cooling_load

        # Geänderte IDF-Datei speichern
        idf.save(output_path)

    def build_problem(self):
        """Erstellt das Problem für das Latin Hypercube Sampling."""
        problem = {
            'num_vars': 3,
            'names': ['cooling_setpoint', 'people_in_room', 'cooling_load'],
            'bounds': [[15, 30],  # Cooling setpoint range
                       [0, 8],  # People in room range
                       [0, 1000]]  # Cooling load range
        }
        return problem

    def run_latin_hypercube(self, n_samples=10):
        """Führt eine Latin Hypercube Sampling-Simulation mit n_samples durch."""
        # Define the problem for LHS
        problem = {
            'num_vars': 3,
            'names': ['cooling_setpoint', 'people_in_room', 'cooling_load'],
            'bounds': [[15, 30],  # Cooling setpoint range
                       [0, 6],  # People in room range
                       [0, 1000]]  # Cooling load range
        }

        # Generate LHS samples
        samples = latin.sample(problem, n_samples)
        for i, sample in enumerate(samples):
            idf_file = self._modify_idf(sample, i)
            self.run_simulation()
            os.remove(idf_file)

    def run_simulation(self):
        """Führt die EnergyPlus-Simulation aus."""
        command = self._build_command()
        print(f"\nRunning simulation with command: {command} \n")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
            print(f"Simulation finished successfully: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Simulation failed: {e.stderr}")
            return False

    def run_future_weather_simulation(self, future_weather_file):
        """Führt eine Simulation mit einer zukünftigen Wetterdatei durch."""
        self.weather_file = future_weather_file
        simulation_dir = "simulations_2050"

        # Sicherstellen, dass das Ausgabeverzeichnis existiert
        if not os.path.exists(simulation_dir):
            os.makedirs(simulation_dir)

        self.run_simulation()

# ------ Part of an IDF File (Format): Schedule of the AC Setpoint --------
"""  --- 13. Juli bis 31. Dezember 2022, different Setpoints --------------

   From 7/13, 
    Through: 12/31,          !- Field 23
    For: Weekdays,           !- Field 24
    Until: 9:00,             !- Field 25
    23,                      !- Field 26
    Until: 12:00,            !- Field 27
    19,                      !- Field 28
    Until: 20:00,            !- Field 29
    20.5,                    !- Field 30
    Until: 24:00,            !- Field 31
    22,                      !- Field 32
    For: Weekends Holidays,  !- Field 33
    Until: 24:00,            !- Field 34
    21,                      !- Field 35
    For: AllOtherDays,       !- Field 36
    Until: 24:00,            !- Field 37
    19;                      !- Field 38
"""

#
# -> --------------- als Template Method: ---------------
#
# class SimulationRunner:
#     def run_simulation(self, output_dir="output"):
#         self.setup_environment()
#         command = self._build_command(output_dir)
#         result = self.execute_command(command)
#         self.cleanup()
#         return result
#
#     def setup_environment(self):
#         """Vorbereitende Schritte wie das Erstellen von Verzeichnissen."""
#         pass
#
#     def execute_command(self, command):
#         """Führt den eigentlichen Simulation-Befehl aus."""
#         pass
#
#     def cleanup(self):
#         """Aufräumarbeiten nach der Simulation."""
#         pass
#


# ----------------- als Strategy Pattern: -----------------
#
# from abc import ABC, abstractmethod
# import subprocess
#
#
# class SimulationStrategy(ABC):
#     """Abstrakte Basis-Klasse für alle Simulationsstrategien."""
#
#     @abstractmethod
#     def run(self, idf_file, weather_file=None, output_dir="output"):
#         """Methode zur Ausführung der Simulation, die von Unterklassen implementiert werden muss."""
#         pass
#
#
# class BasicSimulationStrategy(SimulationStrategy):
#     """Einfache Simulationsstrategie, die einen Standard-Simulationslauf ausführt."""
#
#     def run(self, idf_file, weather_file=None, output_dir="output"):
#         command = self._build_command(idf_file, weather_file, output_dir)
#         try:
#             result = subprocess.run(command, check=True, capture_output=True, text=True)
#             print(f"Simulation finished successfully: {result.stdout}")
#             return True
#         except subprocess.CalledProcessError as e:
#             print(f"Simulation failed: {e.stderr}")
#             return False
#
#     def _build_command(self, idf_file, weather_file, output_dir):
#         command = ["energyplus", "-r", "-w", weather_file, "-d", output_dir, "-r", idf_file]
#         if not weather_file:
#             command = command[:2] + command[4:]  # Wetterdatei entfernen, wenn nicht angegeben
#         return command
#
#
# class AdvancedSimulationStrategy(SimulationStrategy):
#     """Erweiterte Simulationsstrategie mit zusätzlichen Optionen und erweiterten Features."""
#
#     def run(self, idf_file, weather_file=None, output_dir="output"):
#         command = self._build_command(idf_file, weather_file, output_dir)
#         # Zusätzliche erweiterte Simulationsoptionen
#         command.append("--some-advanced-option")
#
#         try:
#             result = subprocess.run(command, check=True, capture_output=True, text=True)
#             print(f"Advanced simulation finished successfully: {result.stdout}")
#             return True
#         except subprocess.CalledProcessError as e:
#             print(f"Advanced simulation failed: {e.stderr}")
#             return False
#
#     def _build_command(self, idf_file, weather_file, output_dir):
#         command = ["energyplus", "-r", "-w", weather_file, "-d", output_dir, "-r", idf_file]
#         if not weather_file:
#             command = command[:2] + command[4:]  # Wetterdatei entfernen, wenn nicht angegeben
#         return command
#
#
# class SimulationRunner:
#     """Context-Klasse, die eine Simulationsstrategie verwendet."""
#
#     def __init__(self, strategy: SimulationStrategy):
#         self.strategy = strategy
#
#     def run(self, idf_file, weather_file=None, output_dir="output"):
#         """Führt die Simulation mit der gewählten Strategie aus."""
#         return self.strategy.run(idf_file, weather_file, output_dir)
#
#
# # Beispielverwendung des Strategy Patterns:
# basic_strategy = BasicSimulationStrategy()
# advanced_strategy = AdvancedSimulationStrategy()
#
# # Verwende die BasicSimulationStrategy
# runner = SimulationRunner(basic_strategy)
# runner.run("example.idf", weather_file="example.epw")
#
# # Wechsel zu AdvancedSimulationStrategy
# runner.strategy = advanced_strategy
# runner.run("example.idf", weather_file="example.epw")
#
