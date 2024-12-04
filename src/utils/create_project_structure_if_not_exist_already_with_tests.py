import os

def create_structure(base_dir):
    # Definiere die Ordnerstruktur und füge Inhalte für die Tests hinzu
    structure = {
        'src': {
            '__init__.py': [],
            'abm': {
                '__init__.py': [],
                'employee.py': [],
                'model.py': [],
                'abm_simulator.py': []
            },
            'analysis_tool': {
                '__init__.py': [],
                'data_visualizer.py': [],
                'statistical_tools.py': [],
                'report_generator.py': []
            },
            'EnergyPlus': {
                '__init__.py': [],
                'idf_parser.py': [],
                'simulation_runner.py': [],
                'result_analyzer.py': []
            },
            'predictions': {
                '__init__.py': [],
                'lstm_predictor.py': [],
                'nowcast.py': [],
                'city_predictor.py': []
            },
            'utils': {
                '__init__.py': [],
                'helper.py': []
            },
            'weatherFetcher': {
                '__init__.py': [],
                'weather_api.py': [],
                'data_processor.py': [],
                'cache_manager.py': []
            },
            'main.py': []
        },
        'tests': {
            '__init__.py': [],
            'test_EnergyPlus.py': [
                "import unittest",
                "from src.EnergyPlus.idf_parser import IDFParser",
                "from src.EnergyPlus.simulation_runner import SimulationRunner",
                "",
                "class TestEnergyPlus(unittest.TestCase):",
                "    def test_idf_parser(self):",
                "        parser = IDFParser('sample.idf')",
                "        self.assertIsNotNone(parser)",
                "        self.assertEqual(parser.file_path, 'sample.idf')",
                "",
                "    def test_simulation_runner(self):",
                "        runner = SimulationRunner('sample.idf')",
                "        self.assertTrue(runner.run_simulation())",
                "",
                "if __name__ == '__main__':",
                "    unittest.main()"
            ],
            'test_weatherFetcher.py': [
                "import unittest",
                "from src.weatherFetcher.weather_api import WeatherAPI",
                "from src.weatherFetcher.data_processor import DataProcessor",
                "",
                "class TestWeatherFetcher(unittest.TestCase):",
                "    def test_weather_api(self):",
                "        api = WeatherAPI('sample_key')",
                "        data = api.fetch_weather('Berlin')",
                "        self.assertIn('temperature', data)",
                "",
                "    def test_data_processor(self):",
                "        processor = DataProcessor()",
                "        raw_data = {'temperature': 22.5, 'humidity': 80}",
                "        processed_data = processor.process(raw_data)",
                "        self.assertEqual(processed_data['temperature'], 22.5)",
                "",
                "if __name__ == '__main__':",
                "    unittest.main()"
            ],
            'test_abm.py': [
                "import unittest",
                "from src.abm.agent import Agent",
                "from src.abm.environment import Environment",
                "",
                "class TestABM(unittest.TestCase):",
                "    def test_agent_creation(self):",
                "        agent = Agent('Agent1')",
                "        self.assertEqual(agent.name, 'Agent1')",
                "",
                "    def test_environment(self):",
                "        environment = Environment(size=(10, 10))",
                "        self.assertEqual(environment.size, (10, 10))",
                "",
                "if __name__ == '__main__':",
                "    unittest.main()"
            ],
            'test_analysis_tool.py': [
                "import unittest",
                "from src.analysis_tool.data_visualizer import DataVisualizer",
                "from src.analysis_tool.statistical_tools import StatisticalTools",
                "",
                "class TestAnalysisTool(unittest.TestCase):",
                "    def test_data_visualizer(self):",
                "        visualizer = DataVisualizer()",
                "        self.assertIsNotNone(visualizer)",
                "",
                "    def test_statistical_tools(self):",
                "        tools = StatisticalTools()",
                "        mean = tools.calculate_mean([1, 2, 3, 4, 5])",
                "        self.assertEqual(mean, 3)",
                "",
                "if __name__ == '__main__':",
                "    unittest.main()"
            ],
            'test_predictions.py': [
                "import unittest",
                "from src.predictions.lstm_predictor import LSTMPredictor",
                "from src.predictions.nowcast import Nowcast",
                "from src.predictions.city_predictor import CityPredictor",
                "",
                "class TestPredictions(unittest.TestCase):",
                "    def test_lstm_predictor(self):",
                "        predictor = LSTMPredictor(model='dummy_model')",
                "        prediction = predictor.predict([1, 2, 3, 4, 5])",
                "        self.assertEqual(len(prediction), 5)",
                "",
                "    def test_nowcast(self):",
                "        nowcast = Nowcast()",
                "        result = nowcast.run('Berlin')",
                "        self.assertIsInstance(result, dict)",
                "",
                "    def test_city_predictor(self):",
                "        predictor = CityPredictor(city='Berlin')",
                "        forecast = predictor.predict_population_growth()",
                "        self.assertGreaterEqual(forecast, 0)",
                "",
                "if __name__ == '__main__':",
                "    unittest.main()"
            ],
            'test_utils.py': [
                "import unittest",
                "from src.utils.helper import Helper",
                "",
                "class TestUtils(unittest.TestCase):",
                "    def test_helper(self):",
                "        result = Helper().some_utility_function()",
                "        self.assertTrue(result)",
                "",
                "if __name__ == '__main__':",
                "    unittest.main()"
            ]
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

    # Funktion zum Erstellen der Verzeichnisse und Dateien mit Inhalten
    def create_files_and_dirs(base_path, structure):
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            if isinstance(content, dict):
                if not os.path.exists(path):
                    os.makedirs(path)
                    print(f"Created directory: {path}")
                create_files_and_dirs(path, content)
            else:
                if not os.path.exists(path):
                    with open(path, 'w') as f:
                        f.write("\n".join(content))  # Schreibe den Inhalt in die Datei
                    print(f"Created file: {path} with content")
                else:
                    print(f"File already exists: {path}")

    # Starte die Erstellung der Struktur
    create_files_and_dirs(base_dir, structure)

# Das Skript wird im aktuellen Verzeichnis ausgeführt
if __name__ == '__main__':
    create_structure(os.getcwd())
