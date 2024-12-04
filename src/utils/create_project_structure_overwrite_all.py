# This script creates a basic project structure for a Python project.

# The structure of the project is as follows:

# Simulation_and_Building_Cooling_Control_test/
# │
# ├── src/
# │   ├── main.py
# │   ├── __init__.py
# │   ├── energyplus/              # EnergyPlus Modul
# │   │   ├── __init__.py
# │   │   ├── idf_parser.py        # Parser für IDF-Dateien
# │   │   ├── simulation_runner.py # Führt EnergyPlus-Simulationen durch
# │   │   └── result_analyzer.py   # Analysiert die Simulationsergebnisse
# │   │
# │   ├── weather_fetcher/          # WeatherFetcher Modul
# │   │   ├── __init__.py
# │   │   ├── weather_api.py       # Fetches Wetterdaten von APIs
# │   │   ├── data_processor.py    # Verarbeitet und normalisiert Wetterdaten
# │   │   └── cache_manager.py     # Caching von Wetterdaten
# │   │
# │   ├── abm/                     # Agent-Based Modeling (ABM) Modul
# │   │   ├── __init__.py
# │   │   ├── employee.py             # Definiert Agenten im ABM-Modell
# │   │   ├── model.py       # Definiert die Umgebung für die Agenten
# │   │   └── abm_simulator.py     # Führt ABM-Simulationen durch
# │   │
# │   ├── predictions/
# │   │   ├── lstm.py             # indoor temp prediction using LSTM
# │   │   ├── nowcast.py          # nowcast prediction using Regression
# │   │   └── others.py           # other prediction methods
# │   │
# │   ├── analysis/                # Analysis Tool Modul
# │   │   ├── __init__.py
# │   │   ├── data_visualizer.py   # Visualisiert Daten
# │   │   ├── statistical_tools.py # Statistische Analyse-Tools
# │   │   └── report_generator.py  # Generiert Berichte aus den Analysen
# │   │
# │   └── utils/                   # Hilfsfunktionen und Utilities
# │       ├── __init__.py
# │       └── helper.py
# │
# ├── tests/
# │   ├── __init__.py
# │   ├── test_energyplus.py       # Tests für das EnergyPlus Modul
# │   ├── test_weather_fetcher.py   # Tests für das WeatherFetcher Modul
# │   ├── test_abm.py              # Tests für das ABM Modul
# │   ├── test_predictions.py       # Tests für das Predictions Modul
# │   ├── test_analysis.py         # Tests für das Analysis Tool Modul
# │   └── test_utils.py            # Tests für Hilfsfunktionen
# │
# ├── docs/
# │   ├── index.md
# │   └── usage.md
# │
# ├── .gitignore
# ├── README.md
# ├── requirements.txt
# ├── setup.py
# └── LICENSE


import os


def create_structure(base_dir):
    # Definiere die Ordnerstruktur
    structure = {
        'src': {
            'energyplus': ['__init__.py', 'submodule1.py', 'submodule2.py'],
            'weather_fetcher': ['__init__.py', 'submodule1.py', 'submodule2.py'],
            'abm': ['__init__.py', 'submodule1.py', 'submodule2.py'],
            'predictions': ['__init__.py', 'submodule1.py', 'submodule2.py'],
            'analysis': ['__init__.py', 'submodule1.py', 'submodule2.py'],
            'utils': ['__init__.py', 'helper.py'],
            '__init__.py': [],
            'main.py': []
        },
        'tests': {
            '__init__.py': [],
            'test_energyplus.py': [],
            'test_weather_fetcher.py': [],
            'test_abm.py': [],
            'test_predictions.py': [],
            'test_analysis.py': [],
            'test_utils.py': []
        },
        'docs': {
            'index.md': [],
            'usage.md': []
        },
        'README.md': [],
        'requirements.txt': [],
        'setup.py': [],
        '.gitignore': [],
        'LICENSE': []
    }

    # Funktion zum Erstellen der Verzeichnisse und Dateien
    def create_files_and_dirs(base_path, structure):
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                create_files_and_dirs(path, content)
            else:
                if name.endswith('.py') or name.endswith('.md') or name in ['README.md', 'requirements.txt', 'setup.py',
                                                                            '.gitignore', 'LICENSE']:
                    with open(path, 'w') as f:
                        pass  # Leere Datei erstellen

    # Starte die Erstellung der Struktur
    create_files_and_dirs(base_dir, structure)


# Das Skript wird im aktuellen Verzeichnis ausgeführt
if __name__ == '__main__':
    pass
    # create_structure(os.getcwd())



