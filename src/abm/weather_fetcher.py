import os
from pprint import pprint
import json
import pandas
from dotenv import load_dotenv
from datetime import datetime, timedelta
import requests
from functools import wraps
from pyowm.owm import OWM
from meteostat import Point, Daily, Hourly, Stations

# ----------- load past und forecast weather data. run:  -----------
# weather_fetcher = WeatherFetcher(location_choosen)
# weather_fetcher.run_weather_fetcher()
# weather_fetcher.get_meteostat_weather()


def handle_api_errors(func):
    """ Decorator to handle errors that occur when making requests to the OpenWeatherMap API. """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            response_decorator = func(*args, **kwargs)
            response_decorator.raise_for_status()  # Raise an error for bad status codes
        except requests.exceptions.HTTPError as err:
            if response_decorator.status_code == 400:
                print("Error 400: Bad Request. Possible issue with the request parameters.")
            elif response_decorator.status_code == 401:
                print("Error 401: Unauthorized. Check your API key or authentication.")
            elif response_decorator.status_code == 429:
                print("Error 429: Too many requests. You have exceeded the daily limit.")
            else:
                print(f"HTTP error occurred: {err}")  # Other Errors
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        return response_decorator
    return wrapper


class WeatherFetcher:
    """ Fetches weather forecast data from the OpenWeatherMap API for a given city. """

    def __init__(self, chosen_location):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        # OpenWeatherMap API-Key
        self.owm = OWM(self.api_key)
        self.city = chosen_location

        city_graz = "Graz"
        self.latitude = 47.0707
        self.longitude = 15.4395    # Graz, Austria coordinates in default case
        self.response_api_25 = None
        self.response_api_3 = None
        self.n_past_weather_jsons = None
        
        self.count_daily_requests = self.count_daily_requests()

    def count_daily_requests(self):
        """Save date and count of requests in a file and check if the count is exceeded for the day."""
        requests_file = "daily_requests.json"
        today_date = datetime.now().date().isoformat()

        if os.path.exists(requests_file):
            with open(requests_file, "r") as file:
                data = json.load(file)
        else:
            data = {}

        if today_date in data:
            if data[today_date] >= 1000:  # Assuming daily limit is 1000 requests
                print("Daily request limit exceeded.")
                return False
            else:
                data[today_date] += 1
        else:
            data[today_date] = 1

        with open(requests_file, "w") as file:
            json.dump(data, file)

        return True
 
    @handle_api_errors
    def get_coords_from_city(self):
        """Get the coordinates of a city using the OpenWeatherMap API."""
        url = f'https://api.openweathermap.org/data/2.5/weather?q={self.city}&appid={self.api_key}'
        return requests.get(url)

    def assign_coords_from_city(self):
        response = self.get_coords_from_city()
        if response:
            json_data = response.json()
            self.latitude = json_data['coord']['lat']
            self.longitude = json_data['coord']['lon']
            print(f"Latitude: {self.latitude}, Longitude: {self.longitude}")

    @handle_api_errors
    def get_weather_data(self):
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={self.city}&appid={self.api_key}&units=metric"
        response_weather = requests.get(url)
        if response_weather:
            return response_weather.json()

    @handle_api_errors
    def get_weather_forecast_from_api_3(self):
        """ Current and forecasts weather data (up to 8 days) for a given city using the OpenWeatherMap API."""
        exclude_vars = "minutely,daily,alerts"
        url = f"https://api.openweathermap.org/data/3.0/onecall?lat={self.latitude}&lon={self.longitude}&exclude={exclude_vars}&appid={self.api_key}&units=metric"
        return requests.get(url)

    def assign_weather_forecast_response_api_3(self):
        response = self.get_weather_forecast_from_api_3()
        if response:
            self.response_api_3 = response.json()
            print(f"forecast Response from openweather API 3 successfull assigned \n ")

    @handle_api_errors
    def get_weather_data_from_timemachine_api_3(self, time):
        """ Weather data for a timestamp for a given city using the OpenWeatherMap API."""
        url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={self.latitude}&lon={self.longitude}&dt={time}&appid={self.api_key}&units=metric"
        return requests.get(url)

    def return_timestamps_from_n_hours_ago(self, lookback=12):
        """ Returns the timestamps of n full hours ago from the current time as Array.
            Args:
                n: The number of timestamps to retrieve.
            Returns:
                A list of timestamps."""
        now = datetime.now()
        last_full_hour = now.replace(minute=0, second=0, microsecond=0)

        timestamps = []
        for i in range(lookback):
            timestamps.append(last_full_hour.timestamp())
            last_full_hour -= timedelta(hours=1)
        # swap the order of the timestamps to get the oldest first and the newest last in the list
        timestamps.reverse()
        # make the entrys to int values
        timestamps = [int(i) for i in timestamps]
        return timestamps

    def fetch_historic_weather_from_timestamp_api_3(self, timestamp_array):
        """ Fetches historic weather data for the given city using the OpenWeatherMap API for the past n full hours."""
        # print(f"Timestamps: {timestamp_array}")
        w_data = {}
        w_data_list = []
        for timestamp in timestamp_array:
            int_timestamp = int(timestamp)
            response_w_data = self.get_weather_data_from_timemachine_api_3(int_timestamp)
            if response_w_data:
                w_data = response_w_data.json()
                w_data_list.append(w_data)
                # json_weather_data = json.dumps(w_data)
                # pprint(json.loads(json_weather_data))
        return w_data_list

    def assign_past_weather_of_n_hours_response_api_3(self):
        timestamps_array_of_past_n_hours = self.return_timestamps_from_n_hours_ago()
        n_past_weather_jsons = self.fetch_historic_weather_from_timestamp_api_3(timestamps_array_of_past_n_hours)
        self.n_past_weather_jsons = n_past_weather_jsons

    def make_dataframe_from_n_past_weather_jsons(self):
        """ Creates a pandas DataFrame from the past n hours of weather data fetched from the OpenWeatherMap API."""
        # weather_one_timestamp_df_full = pandas.DataFrame(columns=['temperature', 'feels_like', 'humidity', 'weather_description'])
        dataframes = []
        for weather_json in self.n_past_weather_jsons:
            json_weather_data = weather_json['data'][0]
            json_weather = json.dumps(json_weather_data)
            # pprint(json.loads(json_weather))

            weather_values = {
                'temperature': json_weather_data['temp'],
                'feels_like': json_weather_data['feels_like'],
                'humidity': json_weather_data['humidity'],
                'weather_description': json_weather_data['weather'][0]['description']
            }
            weather_one_timestamp_df = pandas.DataFrame([weather_values])
            weather_one_timestamp_df['datetime'] = pandas.to_datetime(json_weather_data['dt'], unit='s')
            dataframes.append(weather_one_timestamp_df)
            weather_one_timestamp_df_full = pandas.concat(dataframes, ignore_index=True)
            weather_one_timestamp_df_full.set_index('datetime', inplace=True)

        print(weather_one_timestamp_df_full, "\n")
        return weather_one_timestamp_df_full

    def return_date_str_from_12h_ago(self):
        timestamp_12h_ago = int((datetime.now() - timedelta(hours=12)).timestamp())
        date_str12h_ago = timestamp_12h_ago.strftime('%Y-%m-%d-%H:%M:%S')
        return date_str12h_ago

    def return_feels_like_temp_value_from_openweather_json_api_3(self):
        print(f"Feels like temp: {self.response_api_3['hourly'][0]['feels_like']}")
        return self.response_api_3['hourly'][0]['feels_like']

    def forecast_to_dataframe(self):
        get_weather_data_25 = self.get_weather_data()
        # data_less = self.response_api_25['list']
        data_less = get_weather_data_25['list']
        filtered_data = [
            {'dt_txt': forecast['dt_txt'], 'feels_like': forecast['main']['feels_like'], 'temp': forecast['main']['temp'],
             'temp_min': forecast['main']['temp_min'], 'temp_max': forecast['main']['temp_max']} for forecast in data_less]
        df = pandas.DataFrame(filtered_data)
        df['dt_txt'] = pandas.to_datetime(df['dt_txt'])
        df.set_index('dt_txt', inplace=True)
        return df

    def forecast_to_dataframe_api_3(self):
        data = self.response_api_3['hourly']
        filtered_data = [
            {'dt': forecast['dt'], 'feels_like': forecast['feels_like'], 'temp': forecast['temp'],
             'temp_min': forecast['temp'], 'temp_max': forecast['temp']} for forecast in data]
        df = pandas.DataFrame(filtered_data)
        df['dt'] = pandas.to_datetime(df['dt'], unit='s')
        df.set_index('dt', inplace=True)
        # print(df)
        return df

    def get_next_n_hours_outdoor_temperature_api_3(self, n=12):
        """ Fetches the next n hours of outdoor temperature forecast data for a given city using the OpenWeatherMap API.
        Args:
            n: The number of hours of forecast data to retrieve.
        Returns:
            A pandas DataFrame containing the next n hours of outdoor temperature forecast data.
        """

        current_full_hour_datetime = datetime.now().replace(minute=0, second=0)
        forecast_df_api3 = self.forecast_to_dataframe_api_3()
        # drop first lines if they dont match with current full hour
        while forecast_df_api3.index[0] <= current_full_hour_datetime:
            forecast_df_api3 = forecast_df_api3[1:]

        next_n_hours_outdoor_temperature_df = forecast_df_api3.head(n)
        print(f"Next {n} hours of outdoor temperature forecast data: \n {next_n_hours_outdoor_temperature_df} \n")

        # save to csv
        next_n_hours_outdoor_temperature_df.to_csv('next_n_hours_outdoor_temperature.csv')
        return next_n_hours_outdoor_temperature_df

    def ffill_3h_to_1h_forecasts(self, data):
        df_ffill = data.resample('h').ffill()
        df_ffill['mean_temp'] = df_ffill[['feels_like', 'temp', 'temp_min', 'temp_max']].mean(axis=1)
        # Calculate simple moving average (SMA)
        df_ffill['SMA'] = df_ffill['mean_temp'].rolling(window=3).mean()
        # Calculate exponential moving average (EMA)
        df_ffill['EMA'] = df_ffill['mean_temp'].ewm(span=3, adjust=False).mean()
        return df_ffill

    def get_next_12h_outdoor_temperature(self):
        # data_1 = self.get_weather_data(self.city)
        forecast_df = self.forecast_to_dataframe()  # takes the data from self.request_25
        ffill_forecast_df = self.ffill_3h_to_1h_forecasts(forecast_df)
        next_12h_outdoor_temperature_values_array = ffill_forecast_df['EMA'].head(12).values
        return next_12h_outdoor_temperature_values_array

    def get_next_24h_outdoor_temperature(self):
        """ Fetches the next 24 hours of outdoor temperature forecast data for a given city using the OpenWeatherMap API.
        :param api_key: The API key for the OpenWeatherMap API.
        :param city: The city for which the forecast data should be fetched.
        :return: A numpy array containing the next 24 hours of outdoor temperature forecast data.
        """
        data_1 = self.get_weather_data(self.city)
        forecast_df = self.forecast_to_dataframe()
        ffill_forecast_df = self.ffill_3h_to_1h_forecasts(forecast_df)
        next_24h_outdoor_temperature_values_array = ffill_forecast_df['EMA'].head(24).values
        return next_24h_outdoor_temperature_values_array

    def run_weather_fetcher(self):
        """ Runs the weather fetcher to fetch past and forecast weather data for the given city using the OpenWeatherMap API."""
        print(f"\n Wetterdaten burns weather fetcher with OWM for city {self.city} :")
        self.assign_coords_from_city()
        self.assign_weather_forecast_response_api_3()
        self.assign_past_weather_of_n_hours_response_api_3()
        self.make_dataframe_from_n_past_weather_jsons()   #todo correct the df creation, that hour-1 is the first row of the df and not hour-2
        self.get_next_n_hours_outdoor_temperature_api_3(12)
        return 0

    def init_pyowm_and_get_weather(self):
        # Standort festlegen (z.B. Wien)
        mgr = self.owm.weather_manager()
        # all_stations = mgr.get_stations()
        # pprint(all_stations)
        # station_id_graz = 2778067
        weather = mgr.weather_at_place('Vienna,AT').weather
        # weather_day = mgr.station_day_history(station_id_graz)
        print("\n Wetterdaten pyowm:")
        print(weather, "\n")

    def get_meteostat_weather(self):
        """ Fetches weather data from the Meteostat API for a given city."""
        # Beispiel: Abrufen der Wetterdaten von Meteostat

        # Get nearby weather stations
        # stations = Stations()
        # stations = stations.nearby(lat=self.latitude, lon=self.longitude)
        # station = stations.fetch(1)

        # location = Point(48.2082, 16.3738)  # Standort (Breiten- und Längengrad) Wien, Österreich
        location_graz = Point(lat=self.latitude, lon=self.longitude)
        #todo check if coords are correct or default coords are used for Graz
        start = datetime(2024, 7, 1)  # Startdatum
        end = datetime(2024, 9, 24)  # Enddatum (heutiges Datum)
        # Tagesdaten abrufen
        # data = Daily(location_graz, start, end)
        data = Hourly(location_graz, start, end)
        data = data.fetch()
        # Daten anzeigen
        print("\n Wetterdaten Meteostat:")
        print(data, "\n")

        filename = f"weather_data_{self.city}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        data.to_csv(filename, index=True)
        return data
