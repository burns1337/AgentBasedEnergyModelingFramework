# src/EnergyPlus/idf_parser.py
from eppy.modeleditor import IDF


class IDFParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self._parse_idf()
        self.location_name = self.parse_site_location()

    def parse_site_location(self):
        """Methode zum Parsen der Location in der IDF-Datei."""
        data = {}
        try:
            with open(self.file_path, 'r') as file:
                idf1 = IDF(self.file_path)
                site_location_name = idf1.idfobjects['Site:Location'].Name
                # site_location = site_location_onject.name
                print(site_location_name)
            return data
        except FileNotFoundError:
            raise ValueError(f"File {self.file_path} not found.")
        except Exception as e:
            raise ValueError(f"An error occurred while parsing the IDF file: {str(e)}")

    def _parse_idf(self):
        """Private Methode zum Parsen der IDF-Datei."""
        data = {}
        try:
            with open(self.file_path, 'r') as file:
                for line in file:
                    # Beispielhafte Parsing-Logik für eine IDF-Datei
                    if not line.strip().startswith('!') and ',' in line:
                        key, value = line.split(',', 1)
                        data[key.strip()] = value.strip()
            return data
        except FileNotFoundError:
            raise ValueError(f"File {self.file_path} not found.")
        except Exception as e:
            raise ValueError(f"An error occurred while parsing the IDF file: {str(e)}")

    def get_value(self, key):
        """Gibt den Wert eines bestimmten Schlüssels in der IDF-Datei zurück."""
        return self.data.get(key, None)

    def set_value(self, key, value):
        """Setzt den Wert eines bestimmten Schlüssels in der IDF-Datei."""
        self.data[key] = value

    def save(self, output_path=None):
        """Speichert die modifizierte IDF-Datei."""
        output_path = output_path if output_path else self.file_path
        try:
            with open(output_path, 'w') as file:
                for key, value in self.data.items():
                    file.write(f"{key}, {value};\n")
        except Exception as e:
            raise ValueError(f"An error occurred while saving the IDF file: {str(e)}")





#
# # ------------------- als Strategy Pattern: -------------------
# class IDFParser:
#     def __init__(self, file_path, parser_strategy):
#         self.file_path = file_path
#         self.parser_strategy = parser_strategy
#         self.data = self.parser_strategy.parse(file_path)
#
# class DefaultParserStrategy:
#     def parse(self, file_path):
#         # Standard-Parsing-Logik
#         pass
#
# class AdvancedParserStrategy:
#     def parse(self, file_path):
#         # Erweiterte Parsing-Logik für neuere IDF-Versionen
#         pass
#


#
# # --------------------- IDFParser mit Factory Method ---------------------
# from abc import ABC, abstractmethod
#
#
# class IDFParser(ABC):
#     """Abstrakte Basis-Klasse für alle IDF-Parser."""
#
#     def __init__(self, file_path):
#         self.file_path = file_path
#         self.data = self._parse_idf()
#
#     @abstractmethod
#     def _parse_idf(self):
#         """Abstrakte Methode, die von Unterklassen implementiert werden muss."""
#         pass
#
#     @staticmethod
#     def create_parser(file_path, parser_type="default"):
#         """Factory Method zur Erstellung von IDFParser-Instanzen."""
#         if parser_type == "default":
#             return DefaultIDFParser(file_path)
#         elif parser_type == "advanced":
#             return AdvancedIDFParser(file_path)
#         else:
#             raise ValueError(f"Unknown parser type: {parser_type}")
#
#     def get_value(self, key):
#         """Gibt den Wert eines bestimmten Schlüssels in der IDF-Datei zurück."""
#         return self.data.get(key, None)
#
#     def set_value(self, key, value):
#         """Setzt den Wert eines bestimmten Schlüssels in der IDF-Datei."""
#         self.data[key] = value
#
#     def save(self, output_path=None):
#         """Speichert die modifizierte IDF-Datei."""
#         output_path = output_path if output_path else self.file_path
#         try:
#             with open(output_path, 'w') as file:
#                 for key, value in self.data.items():
#                     file.write(f"{key}, {value};\n")
#         except Exception as e:
#             raise ValueError(f"An error occurred while saving the IDF file: {str(e)}")
#
#
# class DefaultIDFParser(IDFParser):
#     """Standard-Parser für IDF-Dateien."""
#
#     def _parse_idf(self):
#         data = {}
#         try:
#             with open(self.file_path, 'r') as file:
#                 for line in file:
#                     # Beispielhafte Parsing-Logik für eine IDF-Datei
#                     if not line.strip().startswith('!') and ',' in line:
#                         key, value = line.split(',', 1)
#                         data[key.strip()] = value.strip()
#             return data
#         except FileNotFoundError:
#             raise ValueError(f"File {self.file_path} not found.")
#         except Exception as e:
#             raise ValueError(f"An error occurred while parsing the IDF file: {str(e)}")
#
#
# class AdvancedIDFParser(IDFParser):
#     """Erweiterter Parser für neuere Versionen von IDF-Dateien."""
#
#     def _parse_idf(self):
#         data = {}
#         try:
#             with open(self.file_path, 'r') as file:
#                 for line in file:
#                     # Erweiterte Parsing-Logik für neuere IDF-Dateien
#                     if not line.strip().startswith('!') and ',' in line:
#                         key, value = line.split(',', 1)
#                         data[key.strip()] = value.strip()
#             # Zusätzliche Logik für neuere IDF-Versionen könnte hier hinzugefügt werden
#             return data
#         except FileNotFoundError:
#             raise ValueError(f"File {self.file_path} not found.")
#         except Exception as e:
#             raise ValueError(f"An error occurred while parsing the IDF file: {str(e)}")
#
#
# # Beispielverwendung der Factory Method:
# parser = IDFParser.create_parser("example.idf", parser_type="advanced")
# value = parser.get_value("SomeKey")
#
