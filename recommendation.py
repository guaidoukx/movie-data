
import pandas as pd
import numpy as np
import json
# import enumerate

D_movies = pd.read_csv('input/movie_with_director.csv')
D_columns = ['budget', 'genres', 'homepage', 'id', 'keywords',
             'original_language', 'original_title', 'overview', 'popularity',
             'production_companies', 'production_countries', 'release_date',
             'revenue', 'runtime', 'spoken_languages', 'status', 'tagline',
             'title', 'vote_average', 'vote_count', 'director']
D_info_columns = ['genres', 'homepage', 'keywords', 'original_language', 'original_title',
                  'overview', 'production_companies', 'production_countries',
                  'release_date', 'spoken_languages', 'status', 'tagline', 'title','director']
D_num_columns = ['budget', 'id', 'popularity', 'revenue', 'runtime', 'vote_average','vote_count']



def count_items(object_name):
    L = []
    for i in D_movies[object_name]:
        for j in eval(i):
            L.append(j)
    return L

def count_director():
    L = []
    for i in D_movies['director']:
        L.append(i)
    return L

# m = ['11']
# for i in m:
#     print(i)

print(count_director())
# print(eval(D_movies['genres'][1])[1])
# print(eval(D_movies['director'][1])[0])