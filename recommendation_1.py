from sklearn.preprocessing import Imputer,StandardScaler
import pandas as pd
import numpy as np
import json


# using data which is transformed================================================================
D_movies = pd.read_csv('input/movie_with_director.csv')
D_columns = ['budget', 'genres', 'homepage', 'id', 'keywords','original_language', 'original_title', 'overview',
             'popularity', 'production_companies', 'production_countries', 'release_date', 'revenue', 'runtime',
             'spoken_languages', 'status', 'tagline','title', 'vote_average', 'vote_count', 'director']
info_columns = ['genres', 'homepage', 'keywords', 'original_language', 'original_title', 'overview',
                'production_companies', 'production_countries', 'release_date', 'spoken_languages',
                'status', 'tagline', 'title', 'director']  # 14
num_columns = ['budget', 'id', 'popularity', 'revenue', 'runtime','vote_average', 'vote_count']  #8,'year'



use_columns = [ 'id', 'title',
                'budget','revenue',
                'runtime', 'vote_average', 'vote_count','popularity', 'year',
                'keywords',  'genres',
                'production_countries',
                'director']
# delete budget\revenue for too many 0
useless_columns = ['homepage','overview','tagline','status',
                  'spoken_languages','original_language','production_companies','original_title','release_date' ]
change_columns = [  'genres','production_countries', 'director']
numerical_data = D_movies.select_dtypes(exclude=["object"])
information_data = D_movies.select_dtypes(include=["object"])


def print_count_nan():
    def count_nan(object_name):
        objects = D_movies[object_name].fillna('XXXX')
        x = 0
        for object in objects:
            if object != 'XXXX':
                x = x + 1
        print (object_name + ': %d' % x)
    for a in D_columns:
        count_nan(a)


def print_count_zero():
    def count_zero(object_name):
        objects = D_movies[object_name]
        x = 0
        for object in objects:
            if object == 0:
                x = x + 1
        print (object_name + ' zero_num: %d' % x)
    for a in num_columns:
        count_zero(a)


def fill_all_zero():
    def fill_zero(object_name):
        D_movies[object_name] = D_movies[object_name].apply(lambda x: D_movies[object_name].mean() if x == 0 else x)
    for a in num_columns:
        fill_zero(a)


def count_items(object_name):
    L = []
    for i in D_movies[object_name]:
        for j in eval(i):
            L.append(j)
    return set(L)

def count_director():
    L = []
    for i in D_movies['director']:
        L.append(i)
    return set(L)



def word_to_vec(column_name='genres'):
    object_list = []
    object_dict = {}
    objects = D_movies[column_name]
    for object in objects:
        for a in eval(object):
            object_list.append(a)
    object_list = set(object_list)
    print(object_list)
    object_num = len(object_list)
    for object_rank, object in enumerate(object_list):
        zero_vector = np.zeros(shape=object_num)
        zero_vector[object_rank] = 1
        object_dict[object] = zero_vector
    # print(object_list)
    # print(len(object_list))
    # print('---------------------')
    object_to_vectors = []
    for object in objects:
        object_vec = sum(object_dict[o] for o in eval(object))
        object_to_vectors.append(object_vec)
    D_movies[column_name] = object_to_vectors

def split_vector(column_name='genres'):
    objects = D_movies[column_name]
    df = pd.DataFrame()
    for object_rank, object in enumerate(objects):
        df[object_rank] = object
    lenth = len(df)
    df = df.T
    df.columns = [column_name+'{}'.format(i+1) for i in range(lenth)]
    df = pd.concat([D_movies, df], axis=1)
    return df



# print(count_items('genres'))


# change to year====
D_movies['year'] = pd.to_datetime(D_movies['release_date'], errors='coerce').apply(
    lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
# print(D_movies.head())
# fill nan =========

imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
D_movies[num_columns] = imp.fit_transform(D_movies[num_columns])

# fill 0 =========
fill_all_zero()
# ------------------------------------------


# # change colums====
word_to_vec('production_countries')
D_movies = split_vector('production_countries')
del D_movies['production_countries']
word_to_vec('genres')
D_movies = split_vector('genres')
del D_movies['genres']

for i in useless_columns:
    del D_movies[i]

print(D_movies.head())
#
# pd.DataFrame.to_csv(D_movies, 'R_gc.csv', index=None)
