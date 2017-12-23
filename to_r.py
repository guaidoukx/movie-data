from sklearn.preprocessing import Imputer,StandardScaler
import pandas as pd
import numpy as np
import math


movie_data = pd.read_csv('R_gc.csv')
columns = ['budget', 'id', 'keywords', 'popularity', 'revenue', 'runtime',
           'title', 'vote_average', 'vote_count', 'director','year']
num_columns = ['budget', 'id', 'popularity', 'revenue', 'runtime',
               'title', 'vote_average', 'vote_count', 'year']


L1 = ['budget', 'popularity', 'vote_count', 'revenue']
def to_log(object_name):
    movie_data[object_name] = movie_data[object_name].apply(lambda x: math.log(x))
    dif = max(movie_data[object_name]) - min(movie_data[object_name])
    avg = sum(movie_data[object_name]) / len(movie_data[object_name])
    movie_data[object_name] = movie_data[object_name].apply(lambda x: (x - avg) / dif)

L2 = ['runtime', 'vote_average']  #,'year'
def to_one(object_name):
    dif = max(movie_data[object_name]) - min(movie_data[object_name])
    avg = sum(movie_data[object_name])/len(movie_data[object_name])
    movie_data[object_name] = movie_data[object_name].apply(lambda x: (x-avg)/dif)

for a in L1:
    to_log(a)
#
for a in L2:
    to_one(a)



def cos(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)

def return_id_index(id):
    for index, content in enumerate(movie_data['id']):
        if content == id:
            return index


def comp_cos(I):
    input_ = train.iloc[I]
    print(I, movie_data['title'][I])
    L = []
    for ind  in range(len(train)):
        L.append(cos(input_, train.iloc[ind]))
    train['cos'] =L


del movie_data['director']
del movie_data['keywords']
del movie_data['year']
del movie_data['production_countries']
del movie_data['genres']

train = movie_data
train = train.drop('id', axis=1)
train = train.drop('title', axis=1)


def out_():
    AA=train.sort_values('cos', ascending=False)
    ind = AA.index
    l = []
    for i in [1,2,3,4,5]:
        l.append(ind[i])
    print('-------------------')
    print('output:')
    for i in l:
        print(i, movie_data['title'][i])

print('input:')
comp_cos(16)
out_()