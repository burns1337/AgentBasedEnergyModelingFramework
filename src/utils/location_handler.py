import re
from pathlib import Path
from eppy.modeleditor import IDF
from geopy.geocoders import Nominatim
from rapidfuzz import process



class LocationHandler:
    def __init__(self, args):
        self.args = args
        self.epw = args.epw
        self.idf = args.idf
        self.chosen_location = None
        self.geolocator = Nominatim(user_agent="location_checker")

    def get_location(self):
        """ Methode zum Ermitteln der Location für die Wettervorhersage und den epw file download. """
        if self.args.location:
            location = self.check_location(self.args.location)
            location = location.capitalize()
            return location
        elif self.get_location_from_epw_file():
            return self.get_location_from_epw_file()
        else:
            return self.get_default_location()

    def check_location(self, location):
        """ Methode zum Überprüfen und Korrigieren der Location. """
        corrected_location = self.correct_location(location)
        if self.validate_location(corrected_location):
            return corrected_location
        else:
            print(f"Location '{location}' is not valid and could not be corrected.")
            return self.get_default_location()

    def validate_location(self, location):
        """ Methode zum Überprüfen, ob die Location gültig ist. """
        try:
            self.geolocator.geocode(location)
            return True
        except:
            return False

    def correct_location(self, location):
        """ Methode zum Korrigieren der Location. """
        # List of known locations for correction, from all_cities.csv in current directory
        cwd = Path.cwd()
        file_path = cwd / 'utils' / 'all_cities.txt'
        with file_path.open('r', encoding='utf-8') as file:
            known_locations = file.readlines()
        # known_locations = [location.strip() for location in known_locations]
        known_locations = [location.strip().lower() for location in known_locations]
        location = location.strip().lower()

        best_match = process.extractOne(location, known_locations)
        print(f"Chosen location for the EnergyPlus Simulation: '{location}'")
        print(f"(In case of a typo) Best match for location '{location}' with match rate of {best_match[1]}% to city {best_match[0]}")
        if best_match and best_match[1] > 90:  # Confidence threshold
            return best_match[0]
        else:
            return location

    def get_location_from_epw_file(self):
        """Methode zum Extrahieren der Stadt/Location aus dem EPW-Dateinamen."""
        try:
            #TODO: "ST_GRAZ" -> "Graz"
            match = re.search(r'_(\w+)\.', self.epw)
            if match:
                print(f"Location in epw filename: {match.group(1)}")
                return match.group(1)
        except Exception as e:
            print(f"An error occurred while extracting location from EPW file, or no --epw file given: {str(e)}")
            return None

    #ToDo: repair loading idf location name.
    def get_location_from_idf_file(self):
        """Methode zum Parsen der Location in der IDF-Datei."""
        # data = {}
        try:
            with open(self.idf, 'r') as file:
                idf1 = IDF(self.idf)
                site_location_object_name = idf1.idfobjects['Site:Location'][0].Name
                # site_location = site_location_object.name
                print(f'Site Location: {site_location_object_name} in IDF \n')
            return site_location_object_name
        except FileNotFoundError:
            raise ValueError(f"File {self.file_path} not found.")
        except Exception as e:
            raise ValueError(f"An error occurred while parsing the IDF file: {str(e)}")

    def get_default_location(self):
        default_location = "Graz"
        print(f"No location info found in --location, --epw, --idf. Setting default location: {default_location}")
        return default_location

    def set_location(self, location):
        # make capital first letter
        self.chosen_location = location
        print(f"Setting location in parameter LocationHandler.chosen_location: {location}")
        return 0

    def execute(self):
        location = self.get_location()
        self.set_location(location)
        return location

