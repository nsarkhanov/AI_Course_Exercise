import requests
from bs4 import BeautifulSoup
import pandas as pd
page = requests.get(
    'https://www.imdb.com/search/title/?title_type=feature&num_votes=25000,&genres=war&sort=user_rating,desc')
soup = BeautifulSoup(page.content, 'html.parser')
films_lists = soup.find('div', class_="lister-item-content")
film_names = []  # done
film_year = []  # done
film_starts = []
film_time = []  # done
film_disc = []
film_cat = []  # done
f = []
film_names_s = films_lists.h3.find('a')
film_year_s = films_lists.h3.find(class_="lister-item-year text-muted unbold")
# print(film_names_s)
# for i in film_names_s:
#     print(i)
# for i in film_year_s:
#     print(i)
# film_time_s = films_lists.p.find(class_="runtime")
# for i in film_time_s:
#     print(i)
film_cat_year_time_disc = films_lists.find_all(class_="text-muted")
# for i in film_cat_year_time_disc:
#     # print(i.get_text())
film_direc = films_lists.find('p', class_="")
print(film_direc.get_text())
