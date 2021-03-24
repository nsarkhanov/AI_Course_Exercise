import requests
from bs4 import BeautifulSoup
import pandas as pd
html_txt = requests.get(
    'https://forecast.weather.gov/MapClick.php?lat=37.777120000000025&lon=-122.41963999999996#.YFtJC-RRXm1')
# print(html_txt)
soup = BeautifulSoup(html_txt.content, 'html.parser')
week_info_h = soup.find('div',
                        id="detailed-forecast-body")
week_days_with_info = week_info_h.find_all(
    'div', class_="col-sm-2 forecast-label")
short_info_ofweek = week_info_h.find_all(
    'div', class_="col-sm-10 forecast-text")
days = []
info = []
week_day = ['Today', 'Thursday', 'Friday',
            'Saturday', 'Sunday', 'Monday', 'Tuesday']
for b in week_days_with_info:
    if b.getText() in week_day:
        days.append(b.getText())

for b in short_info_ofweek:

    info.append(b.getText())
# print(info)
week_weather = pd.DataFrame(days)
print(week_weather)
