import requests
from bs4 import BeautifulSoup
import pandas as pd

page = requests.get(
    "https://forecast.weather.gov/MapClick.php?lat=37.777120000000025&lon=-122.41963999999996#.YFw0GeRRXm0")
soup = BeautifulSoup(page.content, 'html.parser')
days_info_html = soup.find(
    'ul', id="seven-day-forecast-list")
days_html = soup.find_all('p', class_="period-name")
short_disc_html = days_info_html.find_all('p', class_="short-desc")
days = []
short_discs = []
for day in days_html:
    days.append(day.get_text())
#################################################################
for short_disc in short_disc_html:
    short_discs.append(short_disc.get_text())
################################################################
temp_low = []
temp_high = []
temp_html = days_info_html.find_all(class_="temp temp-low")
for t in temp_html:
    temp_low.append(t.get_text())
#################################################################
temp_html_h = days_info_html.find_all(class_="temp temp-high")
for t in temp_html_h:
    temp_high.append(t.get_text())
# print(temp_high)
# converting F to C
temp_high_C = []
for i in temp_high:
    for j in i.split():
        print(j)
