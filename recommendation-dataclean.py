
import pandas as pd
import numpy as np
import json
# import enumerate


movies_df = pd.read_csv('input/tmdb_5000_movies.csv')
credit_df = pd.read_csv('input/tmdb_5000_credits.csv')


def get_all_content(object_name):
    def get_content(object):
        L = []
        for i in object:
            L.append(i['name'])
        return L
    movies_df[object_name] = movies_df[object_name].apply(lambda x: get_content(eval(x)))
    return movies_df[object_name]


def get_all_director(object_name):
    def get_director(object):
        L = []
        for i in object:
            if i['job'] == 'Director':
                L.append(i['name'])
                return L
    credit_df[object_name] = credit_df[object_name].apply(lambda x: get_director(eval(x)))
    return credit_df[object_name]


def add_director():
    for i in movies_df['id']:
        for j in credit_df['movie_id']:
            if i == j:
                movies_df['director'] = credit_df['crew']


def to_csv():
    L = ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']
    for a in L:
        get_all_content(a)
    get_all_director('crew')
    add_director()
    pd.DataFrame.to_csv(movies_df, 'movie_with_director1.csv', index=None)

to_csv()
