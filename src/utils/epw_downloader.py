# get EnergyPlus weather file (.epw) for desired location from here: https://energyplus.net/weather
import os
import requests
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

from .location_handler import LocationHandler


class EPWDownloader:
    def __init__(self, location, base_url="https://energyplus.net/weather-region/europe_wmo_region_6"):
        self.base_url = base_url
        self.location = location

    def fetch_epw_file(self):
        """
        Fetches the .epw file for the given location.

        Args:
            location: The name of the location.

        Returns:
            The name of the .epw file, or None if not found.
        """

        # Set up the WebDriver for Firefox
        driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))

        # Step 1: Open the search results page
        url_to_search = f"https://energyplus.net/weather-search/{self.location}"
        driver.get(url_to_search)
        print(f'Searching for location: {self.location} at {url_to_search}')

        try:
            try:
                # Step 2: Wait for the first '.btn' link and get its href attribute
                search_result = WebDriverWait(driver, 1).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '.btn[href*="IWEC"]'))
                )
                print('Found IWEC button.')
                # search_href = search_result.get_attribute('href')
            except:
                # If the IWEC search result is not found, try the first '.btn' link
                search_result = driver.find_element(By.CSS_SELECTOR, '.btn')
                # search_href = search_result.get_attribute('href')
                print('IWEC button not found, using the first available button.')

            search_href = search_result.get_attribute('href')
            print(f'Found search result href: {search_href}')

            # Step 3: Navigate to the found href page
            driver.get(search_href)

            # Step 4: Wait for the element where the EPW link should be (usually the second button)
            epw_link = WebDriverWait(driver, 1).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'a.btn[href$=".epw"]'))
            )
            epw_href = epw_link.get_attribute('href')
            print(f'Found EPW file href: {epw_href}')

            # Step 5: Download the EPW file
            response = requests.get(epw_href)
            response.raise_for_status()  # Check that the request was successful

            # Step 6: Save the file
            file_name = os.path.basename(epw_href)
            with open(file_name, 'wb') as file:
                file.write(response.content)
            print(f'EPW file downloaded as: {file_name}')
            return file_name

        except Exception as e:
            print(f"An error occurred: {e}")
            # LocationHandler.get_default_location()
            return None

        finally:
            # Close the browser
            driver.quit()
