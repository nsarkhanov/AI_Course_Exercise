from bs4 import BeautifulSoup
import pandas as pd
import requests


url = "https://www.encyclopedia-titanica.org/cabins.html"

page = requests.get(url)
print(page)
soup = BeautifulSoup(page.content, 'html.parser')
post = soup.find(
    'article', class_="post")
table = post.find('tbody')
# for row in table.find_all('tr')[5:]:
#     col = row.find_all("td")


titanic = pd.DataFrame(
    columns=["Cabin", "People"])
cabins = []
names = []
for row in table.find_all('tr')[5:][2:]:
    col = row.find_all("td")
    for i, cell in enumerate(col):
        if len(col) >= 2:
            if i == 0:
                cabins.append(cell.text)
            elif i == 1:
                names.append(cell.text)

        # name = col[1].text
        # cabin = col[0].text

        # print(col[1])


titanic_data = titanic.append(
    {"Cabin": cabins, "People": names}, ignore_index=True)


titanic_data.to_csv("titanic_data.csv")
