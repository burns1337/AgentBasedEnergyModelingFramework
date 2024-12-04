import os
import random
import subprocess
from eppy import modeleditor
from eppy.modeleditor import IDF
from SALib.sample import latin

idd_path = '/usr/local/EnergyPlus-24-1-0/Energy+.idd'
IDF.setiddname(idd_path)


class IDFModificator:
    def __init__(self, args, file_path):
        self.file_path = file_path
        self.start_date = args.start

    def load_idf_file(self):
        print(f"Loading IDF file: {self.file_path}")
        idf = IDF(self.file_path)
        return idf

    def modify_idf(self, cooling_setpoint, people_in_room, cooling_load):
        idf = self.load_idf_file()

        # Cooling Setpoint ändern
        for obj in idf.idfobjects['HVACTemplate:Thermostat']:
            obj.Constant_Cooling_Setpoint = cooling_setpoint

        # People in Room ändern
        for obj in idf.idfobjects['PEOPLE']:
            obj.Number_of_People = people_in_room

        # Cooling Load ändern
        for obj in idf.idfobjects['HVACTemplate:System:UnitarySystem']:
            obj.Cooling_Supply_Air_Flow_Rate = cooling_load

        # IDF-Datei speichern
        idf.save(self.file_path)
