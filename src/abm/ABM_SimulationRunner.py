import os
import random
import subprocess
# from SALib.sample import latin


def _build_command():
    """Erstellt den Kommandozeilenbefehl für die ABM-Simulation."""
    command = ["mpirun -np 1 python3 abm/repast4py_abm_employees_office_model_06.py abm/random_walk.yaml"]
    command_str = ' '.join(command)
    return command_str


class ABMSimulationRunner:
    def __init__(self, args):
        self.output_dir = args.output if args.output else "ABM_Sim_output"

    def run_simulation(self):
        """Führt die EnergyPlus-Simulation aus."""
        command = _build_command()
        print(f"\nRunning ABM simulation with command:\n {command} \n")
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, shell=True)
            print(f"ABM Simulation finished successfully. {result.stdout}\n")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Simulation failed: {e.stderr}")
            return False
